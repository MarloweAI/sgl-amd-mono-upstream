# SPDX-License-Identifier: Apache-2.0
"""Muse Spark on gfx950: the monokernel spec and its forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed TEP8 verify
step (8, 16, 32 or 64 rows) runs ``_Built.forward`` here, which takes and returns
what ``MuseSparkForConditionalGeneration.forward`` does. The omni wrapper's
prologue is kept: a batch with media features goes through the model's own
forward, which re-enters this one with the embeddings; a text batch embeds and
normalizes here. The backbone is ``MuseSpark1xModel.forward`` with every decoder
layer as one runner launch; SGLang's logits processor closes the step.

Each MXFP4 layer runs as the K1 / K-attn / K2 / K3 launches (``gfx950/runner.py``);
the bf16-expert layers run SGLang's own layer. ``SGLANG_AMD_MONO_MUSE_KERNELS=0``
keeps every layer on SGLang's modules (stage 0: the output is then identical to
the normal path, which isolates the wiring from the numerics). Check mode
(``SGLANG_AMD_MONO_CHECK=1``, graphs off) runs SGLang's layer on a copy of every
launch's input and logs each layer's rel / cosine on the live rows.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.muse_spark import DRAFT_TOKENS, HF_CONFIG, TP
from sgl_amd_mono.muse_spark.gfx950.config import WIDTHS

logger = logging.getLogger(__name__)

# The SGLang names this glue reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "rows",
        "tokens_per_request",
    ),
    "srt/kernel_plugins/amd/binding.py|srt/monokernels/runtime.py": (
        '"target_verify"',
        '"decode"',
    ),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "page_size",
        "ep_size",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "expert_distribution_recorder_mode",
    ),
    "srt/environ.py": (
        "SGLANG_USE_AITER",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_spec",
        '"tp_size"',
        '"tp_rank"',
        '"tp_group"',
        '"attn_tp_size"',
        '"moe_tp_size"',
        '"moe_ep_size"',
    ),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/model_executor/forward_batch_info.py": (
        "contains_mm_inputs",
        "forward_mode",
        "is_decode",
        "out_cache_loc",
    ),
    "srt/models/muse_spark_omni.py": (
        "MuseSparkForConditionalGeneration",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "get_input_embeddings",
    ),
    "srt/rocm_muse_spark/model.py": (
        "MuseSpark1xModel",
        "model",
        "language_model",
        "layers",
        "norm",
        "_ffn_boundaries",
        "layers_to_capture",
        "residual_dtype",
        "self_attn_input_norm",
        "logits_processor",
        "lm_head",
        "capture_aux_hidden_states",
        "nvfp4_checkpoint",
    ),
    "srt/configs/muse_spark.py": ("text_config", "layer_types", "mlp_layer_types"),
}

# Stage 0 is the normal path; the bounds are the check-mode gate the kernels must
# meet (cosine >= 0.996 per layer on the live rows, ~95k context).
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "layer.stream.rel": 0.05,
        "layer.stream.cosine": 0.996,
        "layer.attn_input.rel": 0.05,
        "layer.attn_input.cosine": 0.996,
    },
)


def _text(hf_config):
    """The text model's config: the released checkpoint nests it."""
    return getattr(hf_config, "text_config", None) or hf_config


def _has_flydsl() -> bool:
    return importlib.util.find_spec("flydsl") is not None


def _same(got, want) -> bool:
    if isinstance(want, float):
        return got is not None and math.isclose(float(got), want, rel_tol=1e-6)
    return got == want


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the nested text config, the layer
    kinds, the draft block, FlyDSL."""
    t = _text(hf_config)
    reasons = [
        f"config {k}={getattr(t, k, None)!r}, the kernels are built for {v!r}"
        for k, v in HF_CONFIG.items()
        if not _same(getattr(t, k, None), v)
    ]
    kinds = set(getattr(t, "mlp_layer_types", None) or ())
    if kinds != {"sparse"}:
        reasons.append(
            f"mlp_layer_types {sorted(kinds)}; the kernels run routed experts in every layer"
        )
    kinds = set(getattr(t, "layer_types", None) or ())
    if not kinds <= {"sliding_attention", "full_attention"}:
        reasons.append(f"layer_types {sorted(kinds)}")
    if (
        cfg.speculative_algorithm is not None
        and cfg.speculative_num_draft_tokens != DRAFT_TOKENS
    ):
        reasons.append(f"{cfg.speculative_num_draft_tokens} draft tokens; the widths are built for {DRAFT_TOKENS} a request")  # fmt: skip
    if not _has_flydsl():
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.rocm_muse_spark.model import MuseSpark1xModel
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.muse_spark.weights import layer_reasons

    lm = model.model.language_model
    if not isinstance(lm, MuseSpark1xModel):
        return [f"{type(lm).__name__}; the kernels mirror the ROCm Muse Spark model"]
    reasons = []
    if lm._ffn_boundaries is None:
        reasons.append(
            "MUSE_FUSE_FFN_BOUNDARY=0; the launches hand on the fused FFN boundary"
        )
    if model.nvfp4_checkpoint:
        reasons.append("NVFP4 experts; the kernels read Quark MXFP4 or bf16 experts")
    p = get_parallel()
    if (p.attn_tp_size, p.moe_tp_size, p.moe_ep_size) != (TP, 1, TP):
        reasons.append(
            f"attention TP {p.attn_tp_size} / MoE TP {p.moe_tp_size} / EP {p.moe_ep_size}: "
            f"the kernels shard attention over TP {TP} and experts over EP {TP}"
        )
    if len(lm.layers) != HF_CONFIG["num_hidden_layers"]:
        reasons.append(
            f"{len(lm.layers)} layers, built for {HF_CONFIG['num_hidden_layers']}"
        )
    for i, layer in enumerate(lm.layers):
        reasons += layer_reasons(layer, i)
    return reasons


class _Built:
    """One rank's monokernels: the runner, the layers, and the forward."""

    def __init__(self, lm, runner, original):
        self.lm, self.model, self.runner = lm, lm.model.language_model, runner
        self.original = original  # the model class's own forward, for media batches
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        *args,
        **kwargs,
    ):
        """``MuseSparkForConditionalGeneration.forward`` for a routed step."""
        lm = self.lm
        if (
            input_ids is not None
            and not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            # the omni prologue embeds the media, then re-enters this forward with
            # input_ids None and the embeddings
            return self.original(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        if input_embeds is None:
            input_embeds = lm.get_input_embeddings()(input_ids)
        hidden, aux = self._backbone(plan, positions, forward_batch, input_embeds)
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(self, plan, positions, forward_batch, input_embeds):
        """``MuseSpark1xModel.forward`` on a decode-phase step (no sequence
        parallelism): every layer one launch, the DFlash aux after the listed ones."""
        m, r = self.model, self.runner
        if r.backend is None:
            self._bind()
        rows, q_len = plan.key.rows, plan.key.tokens_per_request
        stream = input_embeds.to(m.residual_dtype).contiguous()
        attn_input = m.layers[0].self_attn_input_norm(stream).contiguous()
        aux, last = [], len(m.layers) - 1
        check = self.check and not torch.cuda.is_current_stream_capturing()
        live = forward_batch.out_cache_loc[:rows] > 0 if check else None
        for i in range(len(m.layers)):
            pre_norm_dtype = torch.bfloat16 if i == last else None
            ref = self._reference(i, positions, forward_batch, stream, attn_input, pre_norm_dtype) if check else None  # fmt: skip
            kv_ref = self._kv_rows(i, forward_batch, rows) if check else None
            # the layer's q (its K1 ran in the previous launch): the merged launch's next K1
            # overwrites it before the attention check reads it
            q_i = self.runner.q[:rows].clone() if check else None
            stream, attn_input = r.launch(
                i=i, rows=rows, q_len=q_len, positions=positions, forward_batch=forward_batch,
                stream=stream, attn_input=attn_input, pre_norm_dtype=pre_norm_dtype,
            )  # fmt: skip
            if ref is not None:
                _log_check(i, live, (stream, attn_input), ref)
                _log_kv_check(i, live, self._kv_rows(i, forward_batch, rows), kv_ref)
                self._attn_check(i, rows, q_len, forward_batch, q_i)
            if i in m.layers_to_capture:  # set after attach: read per step
                aux.append(stream.to(torch.bfloat16))
        r.end_step()
        return attn_input, (aux if self.lm.capture_aux_hidden_states else None)

    def _attn_check(self, i, rows, q_len, forward_batch, q_i):
        """Check mode, full-attention layers, the first steps: the launch's attention
        output (``runner.o``) against the exact softmax attention recomputed in torch from
        its q (``runner.q``), the served page table and the fp8 cache (P rounded to bf16 as
        the kernel), every row's cosine; rows below 0.99 are listed. Isolates K-attn from
        what follows it in the layer."""
        r = self.runner
        if (
            r.windows[i] != 0
            or getattr(self, "_attn_checks", 0) >= 6 * 16
            or int(forward_batch.seq_lens.max()) < 50000
        ):
            return
        self._attn_checks = getattr(self, "_attn_checks", 0) + 1
        from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM, LOCAL_KV_HEADS, Q_K

        attn = self.model.layers[i].self_attn.attn
        k, v = r.pool.get_kv_buffer(attn.layer_id)
        table = r.backend.forward_metadata.kv_indices
        lens = forward_batch.seq_lens
        extra = q_len if forward_batch.forward_mode.is_target_verify() else 0
        scale = float(attn.scaling)
        q, o = q_i.float(), r.o[:rows].float()
        ref = torch.zeros_like(o)
        GQA = Q_K // HEAD_DIM // LOCAL_KV_HEADS
        for req in range(rows // q_len):
            keys = int(lens[req]) + extra
            ctx = keys - q_len
            pages = table[req, : (keys + 15) // 16].long()
            slots = (
                pages[:, None] * 16 + torch.arange(16, device=q.device)[None, :]
            ).reshape(-1)[:keys]
            K, V = k[slots].float(), v[slots].float()  # [keys, KV heads, HEAD_DIM]
            kidx = torch.arange(keys, device=q.device)
            for qp in range(q_len):
                t = req * q_len + qp
                ok = kidx < ctx + qp + 1
                for h in range(LOCAL_KV_HEADS):
                    qh = q[t].reshape(-1, HEAD_DIM)[h * GQA : (h + 1) * GQA]
                    sc = scale * (qh @ K[:, h].T)
                    sc = torch.where(
                        ok[None, :], sc, torch.full_like(sc, float("-inf"))
                    )
                    p = torch.softmax(sc, dim=-1).to(torch.bfloat16).float()
                    ref[t, h * GQA * HEAD_DIM : (h + 1) * GQA * HEAD_DIM] = (
                        p @ V[:, h]
                    ).reshape(-1)
        cos = torch.nn.functional.cosine_similarity(
            o.reshape(rows, -1, HEAD_DIM), ref.reshape(rows, -1, HEAD_DIM), dim=-1
        )  # [rows, heads]
        low = (cos < 0.99).nonzero().tolist()
        # SGLang's own attention on the same q and the step's K / V rows (from the cache, no write)
        nat = None
        try:
            slots_new = forward_batch.out_cache_loc[:rows].long()
            k_new = k[slots_new].to(torch.bfloat16).reshape(rows, -1)
            v_new = v[slots_new].to(torch.bfloat16).reshape(rows, -1)
            nat_o = (
                attn(
                    q_i.to(torch.bfloat16),
                    k_new,
                    v_new,
                    forward_batch,
                    save_kv_cache=False,
                )
                .float()
                .reshape(rows, -1)
            )
            c_nat = torch.nn.functional.cosine_similarity(
                nat_o.reshape(rows, -1, HEAD_DIM),
                ref.reshape(rows, -1, HEAD_DIM),
                dim=-1,
            )
            c_vs = torch.nn.functional.cosine_similarity(
                nat_o.reshape(rows, -1, HEAD_DIM), o.reshape(rows, -1, HEAD_DIM), dim=-1
            )
            nat = f"native vs exact min {c_nat.min().item():.5f} mean {c_nat.mean().item():.5f}; native vs ours min {c_vs.min().item():.5f}"
        except Exception as e:  # noqa: BLE001 - a diagnostic
            nat = f"native attention not run: {e!r}"[:200]
        dump_dir = os.environ.get("SGLANG_AMD_MONO_ATTN_DUMP", "")
        if dump_dir and cos.min().item() < 0.9 and getattr(self, "_attn_dumps", 0) < 4:
            # the failing case's inputs, for an offline reproduction: the request's K / V rows
            # in key order, its pages, q, our output and the exact one
            self._attn_dumps = getattr(self, "_attn_dumps", 0) + 1
            keys = int(lens[0]) + extra
            pages = table[0, : (keys + 15) // 16].clone()
            slots = (
                pages.long()[:, None] * 16 + torch.arange(16, device=q.device)[None, :]
            ).reshape(-1)[:keys]
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(
                {"layer": i, "rank": r.rank, "q": q_i.cpu(), "k": k[slots].cpu(), "v": v[slots].cpu(), "pages": pages.cpu(),
                 "lens": lens.cpu(), "extra": extra, "scale": scale, "o": r.o[:rows].cpu(), "ref": ref.cpu(), "rows": rows, "q_len": q_len},
                os.path.join(dump_dir, f"attn_r{r.rank}_l{i}_{self._attn_dumps}.pt"),
            )  # fmt: skip
        logger.info(
            "monokernel attn check layer %d: %d rows, lens %s, table %s; ours vs exact head cosine min %.5f mean %.5f; low (row, head) %s; %s",
            i, rows, lens.tolist()[:4], tuple(table.shape), cos.min().item(), cos.mean().item(), low[:24], nat,
        )  # fmt: skip

    def _kv_rows(self, i, forward_batch, rows):
        """Check mode: layer ``i``'s K / V cache rows of this step's slots (through the
        sliding pool's slot table on sliding layers), as fp32 copies."""
        r = self.runner
        if r.pool is None or r.caches is None:
            return None
        slots = forward_batch.out_cache_loc[:rows].long()
        _, _, is_swa = r.caches[i]
        if is_swa:
            slots = r.swa_map[slots.clamp_min(0)].long()
        slots = slots.clamp_min(0)
        k, v = r.pool.get_kv_buffer(self.model.layers[i].self_attn.attn.layer_id)
        return k[slots].float().clone(), v[slots].float().clone()

    def _reference(
        self, i, positions, forward_batch, stream, attn_input, pre_norm_dtype
    ):
        """Check mode: SGLang's layer ``i`` on copies of the launch's inputs, returning
        (stream, next attention input) as the launch does. It writes the same KV
        slots with the same values, which the launch then writes again."""
        m = self.model
        return m.layers[i](
            positions, stream.clone(), forward_batch, attn_input=attn_input.clone(),
            ffn_boundary=m._ffn_boundaries[i], pre_norm_dtype=pre_norm_dtype,
        )  # fmt: skip

    def _bind(self) -> None:
        """At the first routed forward, which every rank runs eagerly together: the
        caches SGLang sized after the build, checked, agreed across ranks, bound."""
        import torch.distributed as dist
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.runtime_context import get_parallel

        backend = get_attn_backend()
        pool = backend.token_to_kv_pool
        reasons = self.runner.cache_reasons(backend, pool)
        group = get_parallel().tp_group
        every = [None] * group.world_size
        dist.all_gather_object(every, reasons, group=group.cpu_group)
        bad = [f"rank {k}: {why}" for k, rs in enumerate(every) for why in rs]
        if bad:
            raise ValueError("Muse Spark monokernels: " + "; ".join(bad))
        self.runner.bind(backend, pool)


def _log_check(lid: int, live: torch.Tensor, got, ref) -> None:
    """Check mode: the launch's outputs against SGLang's on the step's live rows: the
    largest difference over the largest reference value, the least row cosine."""
    if not bool(live.any()):
        return
    stats = []
    for name, o, r in zip(("stream", "attn_input"), got, ref):
        o, r = o[live].float(), r[live].float()
        rel = ((o - r).abs().amax() / r.abs().amax().clamp_min(1e-30)).item()
        cos = torch.nn.functional.cosine_similarity(o, r, dim=-1)
        # the least row (one row's flipped expert shows as ~0.98), the mean, and the rows below 0.99
        stats.append(
            f"{name} rel {rel:.5f} cosine {cos.min().item():.6f} mean {cos.mean().item():.6f} low {int((cos < 0.99).sum())}"
        )
    logger.info("monokernel check layer %d, %d rows: %s", lid, int(live.sum()), "; ".join(stats))  # fmt: skip


def _log_kv_check(lid: int, live: torch.Tensor, got, ref) -> None:
    """Check mode: the K / V rows the launch wrote for the live slots against the
    reference layer's (written just before): the share of equal fp8 values and
    the largest difference."""
    if got is None or ref is None or not bool(live.any()):
        return
    stats = []
    for name, o, r in zip(("K", "V"), got, ref):
        o, r = o[live], r[live]
        same = (o == r).float().mean().item()
        diff = (o - r).abs().amax().item()
        stats.append(
            f"{name} equal {same:.4f} maxdiff {diff:.3f} ref|max| {r.abs().amax().item():.3f}"
        )
    logger.info("monokernel kv check layer %d: %s", lid, "; ".join(stats))


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_parallel, get_spec

    from sgl_amd_mono.muse_spark.gfx950.runner import MuseMonoRunner
    from sgl_amd_mono.muse_spark.weights import LayerViews

    lm = model.model.language_model
    layers = list(lm.layers)
    next_norms = [layer.self_attn.pre_attn_norm for layer in layers[1:]] + [lm.norm]
    views = [
        LayerViews.build(layer, i, next_norms[i], ctx.copies)
        for i, layer in enumerate(layers)
    ]
    p = get_parallel()
    # the peer handshake is collective: every rank gets here (attach agreed first)
    runner = MuseMonoRunner(
        views=views,
        modules=layers,
        boundaries=lm._ffn_boundaries,
        rank=p.tp_rank,
        npes=p.tp_size,
        cpu_group=p.tp_group.cpu_group,
        device=lm.norm.weight.device,
    )
    runner.set_deadline(ctx.deadline_s)
    spec_config = get_spec()
    q_len = (
        int(spec_config.speculative_num_draft_tokens or 1)
        if spec_config.speculative_algorithm is not None
        else 1
    )
    # every rank builds every width; FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share
    widths = list(ctx.widths)
    start = len(widths) * p.tp_rank // max(p.tp_size, 1)
    for rows in widths[start:] + widths[:start]:
        runner.precompile(rows, q_len)
    # the class's forward, bound now: attach replaces the instance's afterwards
    original = type(model).forward.__get__(model, type(model))
    return _Built(model, runner, original)


SPEC = MonoSpec(
    name="muse_spark_hip",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(TP,), resident_blocks=BLOCKS)
    },
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.EXPERT_PARALLEL}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "page_size": frozenset({16}),
            "ep_size": frozenset({TP}),
            "speculative_algorithm": frozenset({None, "DFLASH"}),
            # the launches record no expert distribution
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        hf_config={"model_type": "muse_spark"},
        env={
            "SGLANG_USE_AITER": True,
            # benchmark routing overrides: the kernels route with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
