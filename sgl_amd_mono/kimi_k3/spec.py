# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 on gfx950: the monokernel spec and its forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed decode or
DSpark verify step runs ``_Built.forward`` here, which takes and returns what
``KimiK3ForConditionalGeneration.forward`` does.

One launch a layer. It starts at the attention's TP-local core output, which
is o_proj's input, and ends at the layer's MoE output. It runs, in order:

1. o_proj and the attention-TP all-reduce;
2. the MLP attention-residual seam;
3. the router and its top-16;
4. the latent down and the shared gate_up projections;
5. the MXFP4 experts;
6. the shared down and the latent up projections;
7. the MoE all-reduce.

Both reduces go over a peer buffer. Neither leaves the kernel.

The attention itself stays on SGLang's path. That is where this contract
differs from DeepSeek-V4.1's: the model runs the attention up to
``forward_pre_o_proj``, and the launch owns everything after it. The KDA and
the full-attention layers reach that point the same way. At TP8 both stop at a
``CORE``-wide output. So the launch covers every MoE layer, not one attention
kind.

The kernels read SGLang's weights in place. The dense weights are read as bf16
in their checkpoint layout. The experts are read in whichever MXFP4 layout
``Mxfp4MoEMethod`` left behind. There is no prepared-weight copy, so the KV
cache budget pays nothing.

Two further differences from DeepSeek-V4.1 are deliberate. Each is argued in
the module that holds it. The mailbox epoch is bumped by the host, not advanced
on the device (``gfx950/tag.py``). And the build compiles by running each
launch once on real weights, not through FlyDSL's compile-only mode
(``_warmup``).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import logging
import os
from types import SimpleNamespace
from typing import Any

import msgspec
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
from sgl_amd_mono.kimi_k3 import (
    CORE,
    FRONT_MAX_ROWS,
    HF_CONFIG,
    K2_MAX_ROWS,
    LINEAR_ATTN_CONFIG,
    MAX_BLOCKS,
    MAX_ROWS,
)

logger = logging.getLogger(__name__)

TP_SIZE = 8
# The decode context parallel sizes the launches run at. Under DCP the attention
# hands o_proj the same TP-local 12-head core as without it.
DCP_SIZES = (1, 8)
# K2's OPT_* bits (gfx950/layer.py: sentinel all-reduces, preloads, tagged x rows,
# router partials, batched polls), each bit for bit the launch without it; unset
# is layer.K2_OPT, all of them.
K2_OPT = os.environ.get("SGLANG_AMD_MONO_K3_K2_OPT")
# The MLA (DSA) layers' projection front as one launch (gfx950/attention/mla_pre.py):
# AttnRes, fused_qkv_a, the q / kv norms, q_b, g_proj and the indexer's wk,
# weights_proj and wq_b, handed to SGLang's MLA forward and indexer.
MLA_PRE = os.environ.get("SGLANG_AMD_MONO_K3_MLA_PRE", "1") == "1"
# SGLANG_AMD_MONO_K3_K1=0 builds no K1: every KDA layer keeps SGLang's attention.
K1 = os.environ.get("SGLANG_AMD_MONO_K3_K1", "1") == "1"

# The SGLang names this glue reads, by file under the sglang package (``f(p)``:
# a parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    # the MXFP4 expert layout the launch reads, as Mxfp4MoEMethod chose it
    "srt/layers/quantization/mxfp4.py": ("_aiter_situ_uses_gu_interleaved_weights",),
    "srt/utils/common.py": (
        "BumpAllocator",
        "get_bool_env_var",
    ),
    # mlp-sync padding, which the launch has no handling for
    "srt/layers/dp_attention.py": ("require_mlp_sync",),
    # the model's own forward, which _Built reproduces for a routed step
    "srt/managers/mm_utils.py": ("general_mm_embed_routine",),
    "srt/managers/schedule_batch.py": ("Modality",),
    "srt/layers/layer_boundary/__init__.py": (
        "AttentionInputs",
        "get_attn_tp_context",
    ),
    "srt/layers/layer_boundary/residual/batch.py": (
        "final_norm",
        "set_written",
        "start",
        "written_residual",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    # the DCP front's glue: this step's DCP verify metadata, the KV rows, and the
    # all-to-all, LSE combine and output gate after the launch
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": ("full_attn_backend",),
    "srt/layers/attention/base_attn_backend.py": ("kv_layers",),
    "srt/layers/attention/aiter_backend.py": (
        "cprr_kv_indices",
        "cprr_kv_indptr",
        "global_kv_indptr",
        # the verify plan the front makes for itself (_skip_verify_plan)
        "AiterAttnBackend",
        "AiterAttnBackend.make_mla_meta_data(qo_indptr)",
        "AiterAttnBackend.make_mla_meta_data(max_q_len)",
        "AiterAttnBackend.make_mla_meta_data(is_cp_round_robin)",
        "AiterAttnBackend.make_mla_meta_data(nhead_override)",
        "cprr_kernel_heads",
    ),
    # the attention stops here and the launch runs o_proj; both attention
    # classes expose it. The rest are what the forward and the checks read.
    "srt/models/kimi_k3.py": (
        "AuxHiddenStateList",
        "_cdiv",
        "forward_pre_o_proj",
        "use_latent_moe",
        "routed_expert_norm",
        "routed_expert_up_proj",
        "use_attn_residuals",
        "_sp_moe",
        "all_reduce_fusion",
        "experts",
        "o_proj",
        "layers",
        "language_model",
        # _SglFronts: the MLA attention's forward it stands in for, and the
        # gated o_proj wrapper's gate input it leaves unset
        "KimiK3MLAAttention",
        "_gate_hidden_states",
    ),
    "srt/model_executor/forward_mode.py": (
        "is_target_verify",
    ),
}


def _text_config(hf_config):
    """Kimi-K3's shapes live on the text config; a text-only checkpoint has
    none and is its own text config."""
    return getattr(hf_config, "text_config", None) or hf_config


def _linear_attn_config(hf_config) -> dict:
    return getattr(_text_config(hf_config), "linear_attn_config", None) or {}


def _backbone(model):
    """The K3 language backbone, under the VLM wrapper or on its own."""
    return getattr(model, "language_model", model).model


def _covered_layers(hf_config) -> list[int]:
    """The layers the launch can run: every MoE layer. Which attention a layer
    runs does not matter -- the launch starts after it."""
    t = _text_config(hf_config)
    dense = int(getattr(t, "first_k_dense_replace", 0) or 0)
    freq = int(getattr(t, "moe_layer_freq", 1) or 1)
    return [i for i in range(dense, int(t.num_hidden_layers)) if i % freq == 0]


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the shapes, which live on the text
    config and so are out of ``requires.hf_config``'s reach; aiter's own flags;
    and FlyDSL."""
    from sglang.srt.layers.quantization.mxfp4 import (
        _aiter_situ_uses_gu_interleaved_weights,
    )
    from sglang.srt.utils.common import get_bool_env_var

    t = _text_config(hf_config)
    linear = _linear_attn_config(hf_config)
    reasons = [
        f"text config {k} {getattr(t, k, None)!r} is not {v!r}"
        for k, v in HF_CONFIG.items()
        if getattr(t, k, None) != v
    ]
    reasons += [
        f"linear_attn_config {k} {linear.get(k)!r} is not {v!r}"
        for k, v in LINEAR_ATTN_CONFIG.items()
        if linear.get(k) != v
    ]
    # The bank rows a layer's MLP seam reads grow with the layer index; the
    # launch is built for MAX_BLOCKS of them.
    block = getattr(t, "attn_res_block_size", None)
    if not block:
        reasons.append("needs the attention-residual stream (attn_res_block_size)")
    else:
        last = max(_covered_layers(hf_config), default=0)
        blocks = -(-last // block) + int(last % block == 0)
        if blocks > MAX_BLOCKS:
            reasons.append(
                f"attn_res_block_size {block} gives layer {last} {blocks} "
                f"attention-residual blocks, the launch is built for {MAX_BLOCKS}"
            )
    if not _aiter_situ_uses_gu_interleaved_weights():
        reasons.append(
            "needs the MXFP4 experts in aiter's gate / up interleaved layout; "
            "set AITER_SITUV2_A8W4=1 or leave AITER_SITUV2_A4W4 unset"
        )
    if not get_bool_env_var("SGLANG_AITER_K3_OPT"):
        reasons.append("needs SGLANG_AITER_K3_OPT=1")
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    dcp = getattr(cfg, "dcp_size", 1) or 1
    if dcp not in DCP_SIZES:
        reasons.append(f"--dcp-size {dcp} is not one of {list(DCP_SIZES)}")
    if (getattr(cfg, "attn_cp_size", 1) or 1) > 1 or getattr(
        cfg, "enable_prefill_cp", False
    ):
        # CONTEXT_PARALLEL is accepted for DCP; attention CP reshapes the decode rows
        reasons.append("attention context parallel is on; only --dcp-size is accepted")
    if (getattr(cfg, "kda_dwdp_size", None) or 1) > 1:
        # DWDP shards the KDA in-projection, o_proj and shared experts per rank
        # and gathers them in layer hooks; the launches read them whole
        reasons.append("--kda-dwdp-size is on; the launches read unsharded weights")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """Every per-layer check the config cannot answer. The kernels read each
    weight as a raw address, so a wrong dtype, shape or layout is read as if it
    were the expected one."""
    try:
        from sglang.srt.layers.dp_attention import require_mlp_sync
    except ImportError:  # SGLang before sgl-project/sglang#48781
        from sglang.srt.utils.common import require_mlp_sync

    hidden = HF_CONFIG["hidden_size"]
    covered = _covered_layers(hf_config)
    layers = _backbone(model).layers
    reasons = []
    if require_mlp_sync():
        # mlp-sync pads extend batches to a multiple of attn_tp_size; the
        # launch has no padded-row handling.
        reasons.append("mlp-sync padding is on")
    for i in covered:
        layer = layers[i]
        attn, moe = layer.self_attn, layer.mlp
        experts = getattr(moe, "experts", None)
        up = getattr(moe, "routed_expert_up_proj", None)
        o_proj = getattr(attn, "o_proj", None)
        weight = getattr(o_proj, "weight", None)
        checks = (
            (moe.use_latent_moe, "not a latent MoE layer"),
            (moe.routed_expert_norm is not None, "no latent RMSNorm"),
            (
                up is not None
                and up.weight.shape[0] == hidden
                and up.weight.is_contiguous(),
                f"the latent up projection is not a contiguous [{hidden}, latent]",
            ),
            (
                experts is not None
                and getattr(experts.w13_weight, "is_shuffled", False)
                and getattr(experts.w2_weight, "is_shuffled", False),
                "the experts are not in aiter's preshuffled MXFP4 layout",
            ),
            (layer.use_attn_residuals, "no attention-residual stream"),
            # SP-MoE reduce-scatters o_proj's output and the fused all-reduce
            # completes it out of a symmetric buffer; the launch does neither.
            (not layer._sp_moe, "SP-MoE is on"),
            (not layer.all_reduce_fusion, "the fused attention all-reduce is on"),
            (
                hasattr(attn, "forward_pre_o_proj"),
                "the attention cannot stop before o_proj",
            ),
            # The MLA cores reach o_proj from several places, so stopping
            # before it rides on the gated o_proj wrapper, which only exists
            # with the output gate. The KDA layers have no such attribute and
            # stop before o_proj on their own.
            (
                getattr(attn, "use_output_gate", True),
                "the MLA output gate is off, and the split before o_proj "
                "rides on its wrapper (mla_use_output_gate)",
            ),
            (
                weight is not None
                and weight.dtype == torch.bfloat16
                and tuple(weight.shape) == (hidden, CORE)
                and weight.is_contiguous(),
                f"o_proj is not a contiguous bf16 [{hidden}, {CORE}]",
            ),
            (
                o_proj is not None and getattr(o_proj, "bias", None) is None,
                "o_proj has a bias",
            ),
        )
        reasons += [f"layer {i}: {why}" for ok, why in checks if not ok]
        if reasons and i > covered[0]:
            break  # one layer's reasons say it for all
    return reasons


def _build(model, hf_config, ctx):
    """Pack every covered layer and compile the widths, before graph capture."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950.runner import K3MonoRunner

    backbone = _backbone(model)
    p = get_parallel()
    device = backbone.layers[0].input_layernorm.weight.device
    # the peer handshake is collective: every rank reaches it here
    runner = K3MonoRunner(p.tp_size, p.tp_rank, p.tp_group.cpu_group, device)
    runner.set_deadline(ctx.deadline_s)
    runner.dcp_rank = p.attn_dcp_rank if p.attn_dcp_size > 1 else 0
    if p.attn_dcp_size > 1:
        _fuse_qkv_g(backbone)
    mla_off = _mla_front_disable_reason(p)
    if mla_off is not None:
        logger.info("monokernel: no MLA projection front: %s", mla_off)
    dcp = p.attn_dcp_size > 1
    vcs = _front_vcs(backbone, _covered_layers(hf_config), p.dcp_group) if dcp else {}
    layers, batches = {}, {}
    for rows in ctx.widths:
        for i in _covered_layers(hf_config):
            packed = _weights(
                backbone.layers[i], i, rows, runner, mla_off is None, dcp, vcs.get(i)
            )
            layers.setdefault(rows, {})[i] = packed
        # consecutive DCP fronts take alternate QG copies (mla_front.peer_end)
        fronts = [p for _, p in sorted(layers[rows].items()) if p.front is not None]
        for n, p in enumerate(fronts):
            key, w = p.front
            p.front = (dataclasses.replace(key, qg_par=n % 2), w)
        # FlyDSL compiles a launcher on its first call, and the first real call
        # is inside a CUDA graph capture, where compiling aborts the process.
        # A compile-only call cannot stand in: with zero arguments the buffer
        # loads fold to a pattern LLVM cannot select. So run each launch once,
        # on this layer's real weights and throwaway activations. Every rank
        # runs the same launches in the same order, so the peer waits pair up
        # exactly as they do on a served step.
        #
        # K1's conv rows come from its batch, so that is built first.
        batches[rows] = k1_mod.build_batch(layers[rows], device)
        _warmup(runner, layers[rows], rows)
    sgl = _SglFronts.install(backbone, runner, vcs, layers, ctx.widths) if dcp else None
    if dcp:
        _SglTails.install(backbone, runner, vcs, ctx.widths)
    global _FRONT_ROWS
    _FRONT_ROWS = frozenset(
        rows for rows, packed in layers.items()
        if _every_mla_layer_fronted(backbone, packed)
    ) | (sgl.rows if sgl is not None else frozenset())  # fmt: skip
    return _Built(model, runner, layers, batches)


def _warmup(runner, packed_layers: dict, rows: int) -> None:
    """Launch every layer once at this width, before any graph capture."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950.layer import scratch_region
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    device = runner.device
    core = torch.zeros(rows, CORE, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    banks: dict[int, torch.Tensor] = {}

    def bank(nb: int):
        # One row past the nb blocks the launch reads: a block-write layer's
        # launch (K1 or mla_pre, key.write_idx == nb) stores its updated prefix
        # in block nb, as a served step does into the bank's spare capacity.
        # An nb-row bank put the last token's write one HIDDEN row past the end
        # and zeroed whatever the allocator had placed there (seen: a later
        # layer's indexer k_norm at 8 draft tokens, which silently broke that
        # layer's DSA top-k).
        if nb not in banks:
            banks[nb] = torch.zeros(
                rows, nb + 1, HIDDEN, dtype=torch.bfloat16, device=device
            )
        return banks[nb]

    for packed in packed_layers.values():
        # K1 first: on a served step it runs before this layer's K2, so the
        # waits pair up in the same order here as there.
        if packed.k1 is not None:
            k1_mod.warmup(
                runner, packed.k1, runner.prefix[:rows], bank(packed.k1.key.nblocks)
            )
        if packed.front is not None:
            _front_warmup(runner, packed, rows, bank(packed.front[0].nblocks))
        if packed.mla is not None:
            from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

            key, w = packed.mla
            mp.mla_pre_launch(
                key, prefix=runner.prefix[:rows], blocks=bank(key.nblocks), w=w,
                out=_mla_outputs(rows, device),
                scratch=scratch_region(runner.scratch, "mla_pre"),
                layer=packed.i, epoch=runner.epoch,
            )  # fmt: skip
        _k2_launch(
            runner, packed, core, runner.prefix[:rows], bank(packed.key.nblocks), out
        )
    torch.cuda.synchronize()


# Bounds against SGLang's standard path on the same GPU. They come from what
# test/kernels/kimi_k3 measures at TP8, with a little room. Each number below
# is the worse of the two prefix modes, and for the MoE the worst of widths
# 1, 4 and 8:
#
#   k2.prefix  cos 1.000000            k2.x  cos 1.000000
#   k2.out     cos 0.999994 rel 0.0036
#   moe.out    cos 0.999847 rel 0.0175
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # The updated attention-residual prefix: o_proj, the TP reduce and
        # the add. The reduce sums the same bf16 partials in the same rank
        # order as the standard path, so the result is exact. A bound below
        # 1.0 would let a reordered reduce through.
        "k2.prefix.cosine": 1.0 - 1e-7,
        # The seam: slice, gate, mix, then the FFN input norm. Also exact.
        "k2.x.cosine": 1.0 - 1e-7,
        # The layer output: the MoE on top of the seam. The expert sums land in
        # another order, so this is bf16 rounding, not exact.
        "k2.out.cosine": 0.99999,
        "k2.out.rel": 0.005,
        # The MoE launch, which runs in no forward: the kernel checks use it as
        # the reference for the MoE half of the layer launch. Looser than
        # k2.out because the check feeds it a fresh input at every width, and
        # one row has the least averaging.
        "moe.out.cosine": 0.9998,
        "moe.out.rel": 0.02,
        # SGLANG_AMD_MONO_CHECK, which scores K1 against SGLang's own
        # attention on the real weights at TP8, over 28 steps of one request:
        #
        #   k1.core    cos 0.99999678  rel 0.0026
        #   k1.bank    rel 0 (bit for bit); cos 0.99999982
        #   k1.ssm     cos 0.99999976  rel 0.00033
        #   k1.window  cos 0.99999988  rel 0.00032
        #
        # The gated core output: the in-projection, f_b, the conv and the
        # recurrence. Its reference sums in another order, so this is bf16
        # rounding.
        "k1.core.cosine": 0.99999,
        "k1.core.rel": 0.004,
        # The bank row the launch writes in place of AttnResidual.write. The
        # same values in the same order, so it is exact: rel is what holds it
        # to that, and DSpark's capture reads the bank, so a row that drifts
        # changes the draft's input. The cosine bound is the looser of the two
        # on purpose. Cosine of two 4608-wide bf16 rows carries its own
        # rounding, measured down to 0.99999982 on rows that are bit for bit.
        "k1.bank.cosine": 1.0 - 1e-6,
        "k1.bank.rel": 0.0,
        # The recurrent state staged for the step's verify rows.
        "k1.ssm.cosine": 1.0 - 1e-6,
        "k1.ssm.rel": 0.001,
        # The conv window handed back to SGLang's speculative state, which the
        # accept-time commit reads. Nothing in the serving path reports a wrong
        # one: it shows up as a lower acceptance length and a GSM8K score a few
        # points down. Measured exact, so the bound is tight on purpose.
        "k1.window.cosine": 1.0 - 1e-6,
        "k1.window.rel": 0.001,
    },
)

SPEC = MonoSpec(
    name="kimi_k3_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, K2_MAX_ROWS + 1)),
            tp_sizes=(TP_SIZE,),
            resident_blocks=BLOCKS,
        )
    },
    # The measured win is on DSpark target-verify steps; a plain decode step of
    # at most K2_MAX_ROWS rows takes the same launch. DCP (CONTEXT_PARALLEL)
    # shards the MLA layers' KV.
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.CONTEXT_PARALLEL}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={
            # the launch holds the experts TP sharded, not expert-parallel
            "moe_a2a_backend": frozenset({None, "none"}),
            # DSpark's target steps are uniform verify steps; other drafts differ
            "speculative_algorithm": frozenset({None, "DSPARK"}),
        },
        # The shapes are on the text config, which this cannot reach; they are
        # checked in _config_disable_reasons.
        hf_config={"model_type": "kimi_k3"},
        env={"SGLANG_USE_AITER": True},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)


class _Layer(msgspec.Struct):
    """One layer's launches: their build keys and the weights they read in
    place. ``k1`` is None on the full-attention layers, which keep SGLang's
    attention and enter at ``forward_pre_o_proj``."""

    i: int
    key: Any
    module: Any  # the KimiK3DecoderLayer, for the tensors read per step
    k1: Any = None  # the KDA front launch (gfx950/k1.py), where it applies
    mla: Any = None  # (MlaPreBuild, weights): an MLA layer's projection front
    # (MlaFrontBuild, weights): an MLA layer's attention under DCP8, up to the
    # all-to-all's send records (gfx950/attention/mla_front.py)
    front: Any = None


def _k2_blocks(layer) -> int:
    """Bank rows readable at this layer's MLP seam. A block-write layer
    snapshots its own pre-attention prefix on the attention side, so by the
    seam it has one row more than it inherited."""
    return layer.prev_valid_blocks + int(layer.is_block_write_layer)


def _mla_front_disable_reason(parallel) -> str | None:
    """Why no MLA layer takes the projection front (mla_pre), or None. Under DCP
    the MLA layers take the DCP front (mla_front) at its width instead."""
    if not MLA_PRE:
        return "SGLANG_AMD_MONO_K3_MLA_PRE=0"
    if parallel.attn_dcp_size > 1:
        # SGLang's DCP MLA forward attends every head over this rank's KV shard
        # and computes all their q itself (the replicated q_b); the projection
        # front hands over this rank's 12 heads' q
        return "DCP is on: the projection front computes only this rank's 12 heads' q"
    return None


def _weights(
    layer, i: int, rows: int, runner, mla: bool, dcp: bool, vc=None
) -> _Layer:
    """This layer's build key at one launch width. The weights themselves are
    read in place at launch, so nothing is packed or copied here; ``vc`` is the
    DCP front's all-head W_UV (_front_vcs)."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.layer import K2Build

    moe = layer.mlp
    # K1 and the MLA front take up to MAX_ROWS rows; a wider step keeps
    # SGLang's attention and enters the layer launch at o_proj's input
    narrow = rows <= MAX_ROWS
    return _Layer(
        k1=k1_mod.build(layer, i, rows) if K1 and narrow else None,
        mla=_mla_build(layer, rows) if mla and narrow else None,
        front=_front_build(layer, rows, vc) if dcp else None,
        i=i,
        key=K2Build(
            tokens=rows,
            nblocks=_k2_blocks(layer),
            eps=layer.mlp_res_norm.variance_epsilon,
            out_eps=layer.post_attention_layernorm.variance_epsilon,
            ln_eps=moe.routed_expert_norm.variance_epsilon,
            reset=layer.is_block_write_layer,
            opt=k2.K2_OPT if K2_OPT is None else int(K2_OPT),
            guint=_gu_interleaved(moe),
            fault_site=runner.fault_site,
            fault_after=runner.fault_after,
        ),
        module=layer,
    )


def _mla_build(layer, rows: int):
    """An MLA layer's projection-front key and the weights it reads in place, or
    None (a KDA layer, or weights the launch does not take: SGLang's path)."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    attn = layer.self_attn
    idx = getattr(attn, "indexer", None)
    fused = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    if hasattr(attn, "qkv_conv1d") or idx is None or fused is None:
        return None
    w = {
        "ares_nw": layer.self_attention_res_norm.weight,
        "ares_qk": layer.self_attention_res_proj.weight.view(HIDDEN),
        "in_nw": layer.input_layernorm.weight,
        "w_qkv": fused.weight,
        "w_g": getattr(getattr(attn, "g_proj", None), "weight", None),
        "w_k": getattr(getattr(idx, "wk", None), "weight", None),
        "w_p": getattr(getattr(idx, "weights_proj", None), "weight", None),
        "w_qb": attn.q_b_proj.weight,
        "w_iq": getattr(getattr(idx, "wq_b", None), "weight", None),
        "q_nw": attn.q_a_layernorm.weight,
        "kv_nw": attn.kv_a_layernorm.weight,
    }
    want = {
        "w_qkv": (mp.LATW, HIDDEN), "w_g": (mp.GATE, HIDDEN), "w_k": (mp.IK, HIDDEN),
        "w_p": (mp.IW, HIDDEN), "w_qb": (mp.QB, mp.QL), "w_iq": (mp.IQ, mp.QL),
    }  # fmt: skip
    for n, shp in want.items():
        t = w[n]
        if (
            not isinstance(t, torch.Tensor)
            or t.dtype != torch.bfloat16
            or tuple(t.shape) != shp
            or not t.is_contiguous()
        ):
            logger.info("monokernel: no MLA front for layer %d: %s is %s", layer.layer_id if hasattr(layer, "layer_id") else -1, n, None if t is None else (t.dtype, tuple(t.shape)))  # fmt: skip
            return None
    # the absorb's weight as SGLang's bf16 bmm multiplies it: w_kc itself, unscaled
    wkc, wsc = getattr(attn, "w_kc", None), getattr(attn, "w_scale", None)
    qn = (
        isinstance(wkc, torch.Tensor)
        and wkc.dtype == torch.bfloat16
        and tuple(wkc.shape) == (mp.NHQ, mp.QN, mp.KV)
        and isinstance(wsc, (int, float))
        and wsc == 1.0
    )
    if qn:
        w["w_kc"] = wkc.transpose(1, 2).contiguous().view(mp.NHQ * mp.KV, mp.QN)
    key = mp.MlaPreBuild(
        tokens=rows,
        qn=qn,
        nblocks=layer.prev_valid_blocks,
        eps=layer.self_attention_res_norm.variance_epsilon,
        out_eps=layer.input_layernorm.variance_epsilon,
        q_eps=attn.q_a_layernorm.variance_epsilon,
        kv_eps=attn.kv_a_layernorm.variance_epsilon,
        write_idx=(layer.prev_valid_blocks if layer.is_block_write_layer else -1),
    )
    return (key, w)


def _mla_outputs(rows: int, device) -> dict:
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

    def z(n):
        return torch.zeros(rows, n, dtype=torch.bfloat16, device=device)

    return {
        "q": z(mp.QB), "latent": z(mp.KV + mp.ROPE), "k_nope": z(mp.KV),
        "q_lora": z(mp.QL), "gate": z(mp.GATE), "idx_q": z(mp.IQ),
        "idx_k": z(mp.IK), "idx_w": z(mp.IW),
        "qn": torch.zeros(rows, mp.NHQ, mp.KV, dtype=torch.bfloat16, device=device),
    }  # fmt: skip


def _front_vcs(backbone, layers, group) -> dict:
    """Each MLA layer's W_UV for all 96 heads, [96, 512, 128] bf16: the DCP
    ranks' 12-head shards gathered in rank order. SGLang's own full-head copy
    (dcp_replicate_q_proj) is made only after this build. A collective: every
    rank gathers the same layers in the same order."""
    out = {}
    for i in layers:
        attn = getattr(backbone.layers[i], "self_attn", None)
        w = getattr(attn, "w_vc", None)
        if hasattr(attn, "qkv_conv1d") or not isinstance(w, torch.Tensor):
            continue
        if w.dtype == torch.bfloat16:
            out[i] = group.all_gather(w.contiguous(), dim=0)
    return out


def _front_build(layer, rows: int, vc):
    """An MLA layer's DCP8 front key and the weights it reads in place, or None:
    a KDA layer, a width other than 1 .. 8 requests' verify blocks, or weights
    the launch does not take (SGLang's path). Each width has its own build."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    attn = layer.self_attn
    fused = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    if (
        rows % mf.G_ROWS
        or rows // mf.G_ROWS > mf.MAX_REQS
        or hasattr(attn, "qkv_conv1d")
        or fused is None
    ):
        return None
    mqa = getattr(attn, "attn_mqa", None)
    w = {
        "ares_nw": layer.self_attention_res_norm.weight,
        "ares_qk": layer.self_attention_res_proj.weight.view(HIDDEN),
        "in_nw": layer.input_layernorm.weight,
        "w_qkv": fused.weight,
        "w_g": getattr(getattr(attn, "g_proj", None), "weight", None),
        "q_nw": attn.q_a_layernorm.weight,
        "kv_nw": attn.kv_a_layernorm.weight,
        # this rank's 12 heads' shards (the q all-gather covers the rest)
        "w_qb": getattr(getattr(attn, "q_b_proj", None), "weight", None),
        "w_kc": getattr(attn, "w_kc", None),
        "w_vc": vc,
    }
    for n in mf.WEIGHT_SHAPES:
        t = w[n]
        if not mf.weight_ok(n, t):
            logger.info("monokernel: no DCP front for layer %d: %s is %s", layer.layer_id if hasattr(layer, "layer_id") else -1, n, None if t is None else (t.dtype, tuple(t.shape)))  # fmt: skip
            return None
    if mqa is None or getattr(attn, "rotary_emb", None) is not None:
        return None
    key = mf.MlaFrontBuild(
        tokens=rows,
        nblocks=layer.prev_valid_blocks,
        scale=float(mqa.scaling),
        eps=layer.self_attention_res_norm.variance_epsilon,
        out_eps=layer.input_layernorm.variance_epsilon,
        q_eps=attn.q_a_layernorm.variance_epsilon,
        kv_eps=attn.kv_a_layernorm.variance_epsilon,
        write_idx=(layer.prev_valid_blocks if layer.is_block_write_layer else -1),
    )
    return (key, w)


# The step widths at which every MLA layer runs the DCP front (the spec's
# layer launches and _SglFronts), set by _build, and the target's MLA backends
# the front has run on: there SGLang's verify attention plan
# (make_mla_meta_data, aiter's get_mla_metadata_v1) has no reader. The plan's
# skip is installed at the front's first launch (_front_plan_skip).
_FRONT_ROWS: frozenset = frozenset()
_FRONT_BACKENDS = None


def _front_plan_skip(backend) -> None:
    """This module's skip of SGLang's verify plan at ``_FRONT_ROWS``;
    ``backend`` among the backends it applies to."""
    import weakref

    global _FRONT_BACKENDS
    if _FRONT_BACKENDS is None:
        _FRONT_BACKENDS = weakref.WeakSet()
        _skip_verify_plan()
    _FRONT_BACKENDS.add(backend)


def _every_mla_layer_fronted(backbone, packed_layers: dict) -> bool:
    """Whether every MLA (full-attention) layer this process runs has the DCP
    front at this width (a layer it does not hold has no attention)."""
    mla = [
        i for i, layer in enumerate(backbone.layers)
        if hasattr(layer, "self_attn") and not hasattr(layer.self_attn, "qkv_conv1d")
    ]  # fmt: skip
    return bool(mla) and all(
        i in packed_layers and packed_layers[i].front is not None for i in mla
    )


def _skip_verify_plan() -> None:
    """SGLang's DCP verify plan is skipped for the target backends the front
    runs on, at its widths: the cp round-robin call with the asm verify's
    head count, whose step is ``_FRONT_ROWS`` rows. Every other call (the
    drafter's backend, decode, extend, other widths) runs as SGLang's, and so
    does every call while another copy of this module is the loaded one (a
    hot swap's other arm)."""
    import functools
    import inspect
    import sys

    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

    mod = sys.modules[__name__]
    orig = AiterAttnBackend.make_mla_meta_data
    sig = inspect.signature(getattr(orig, "__wrapped__", orig))

    @functools.wraps(orig)
    def make_mla_meta_data(self, *args, **kwargs):
        if sys.modules.get(__name__) is mod and self in _FRONT_BACKENDS:
            a = sig.bind(self, *args, **kwargs).arguments
            rows = (a["qo_indptr"].numel() - 1) * int(a["max_q_len"])
            if (
                a.get("is_cp_round_robin", False)
                and a.get("nhead_override") is not None
                and a["nhead_override"] == self.cprr_kernel_heads
                and rows in _FRONT_ROWS
            ):
                return None
        return orig(self, *args, **kwargs)

    AiterAttnBackend.make_mla_meta_data = make_mla_meta_data


def _front_core_buffer(rows: int, device) -> torch.Tensor:
    """The DCP front's gated core, a width."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    return torch.zeros(rows, mf.GATE, dtype=torch.bfloat16, device=device)


def _front_warmup(runner, packed, rows: int, blocks) -> None:
    """The DCP front once, on throwaway KV and every request at P = 0."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf
    from sgl_amd_mono.kimi_k3.gfx950.layer import scratch_region

    dev = runner.device
    key, w = packed.front
    nreq = rows // mf.G_ROWS
    mf.mla_front_launch(
        key, prefix=runner.prefix[:rows], blocks=blocks, w=w,
        kv=torch.zeros(64, mf.DQK, dtype=torch.uint8, device=dev),
        kv_idx=torch.zeros(8, dtype=torch.int32, device=dev),
        kv_ptr=torch.tensor([0] + [1] * nreq, dtype=torch.int32, device=dev),
        g_kv_ptr=torch.arange(nreq + 1, dtype=torch.int32, device=dev) * mf.G_ROWS,
        loc=torch.zeros(rows, dtype=torch.int64, device=dev),
        core=_front_core_buffer(rows, dev),
        scratch=scratch_region(runner.scratch, "front"),
        peers=runner.peer.addresses, rank=runner.dcp_rank, layer=packed.i,
        epoch=runner.epoch,
    )  # fmt: skip


class _SglFronts:
    """The DCP front inside SGLang's own MLA attention, at the verify widths the
    spec builds no layer launch for, up to ``FRONT_MAX_ROWS``. There SGLang
    runs the whole layer: its seam and input_layernorm, then this front on
    their x rows (an ``x_in`` build: the q path, the attention, the
    all-to-all, LSE combine and output gate, and this rank's KV record), then
    SGLang's o_proj on the gated core, its all-reduce and MoE. Every other
    step of these layers (other widths, decode, extend) is SGLang's own
    forward. One static rule a width: no step chooses."""

    def __init__(self, runner, entries: list, rows: frozenset) -> None:
        self.runner, self.entries, self.rows = runner, entries, rows
        self.core = {r: _front_core_buffer(r, runner.device) for r in rows}

    @classmethod
    def install(cls, backbone, runner, vcs: dict, by_width: dict, widths):
        """Every MLA layer's ``x_in`` front at the verify widths past the spec's
        own, built and run once (every rank the same launches in the same
        order, so the peer waits pair up), then each layer's attention routed
        through it. None where no width or no layer qualifies."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

        rows = frozenset(
            r for r in range(mf.G_ROWS, FRONT_MAX_ROWS + 1, mf.G_ROWS)
            if r not in widths and r // mf.G_ROWS <= mf.MAX_REQS
        )  # fmt: skip
        if not rows:
            return None
        mla = [
            i for i, layer in enumerate(backbone.layers)
            if hasattr(layer, "self_attn") and not hasattr(layer.self_attn, "qkv_conv1d")
        ]  # fmt: skip
        entries = []
        for n, i in enumerate(mla):
            layer = backbone.layers[i]
            keys, w = {}, None
            for r in sorted(rows):
                built = _front_build(layer, r, vcs.get(i))
                if built is None:
                    return None  # a layer the front does not take: none of them
                key, w = built
                # consecutive fronts take alternate QG copies (mla_front.peer_end)
                keys[r] = dataclasses.replace(
                    key, nblocks=0, write_idx=-1, x_in=True, qg_par=n % 2
                )
            entries.append(SimpleNamespace(i=i, attn=layer.self_attn, keys=keys, w=w))
        self = cls(runner, entries, rows)
        for r in sorted(rows):
            self._warmup(r)
        for n, e in enumerate(entries):
            e.first = n == 0
            e.attn.forward = functools.partial(self.forward, e)
        logger.info(
            "monokernel: DCP front in SGLang's MLA attention at %s rows (%d layers)",
            sorted(rows), len(entries),
        )  # fmt: skip
        return self

    def _warmup(self, rows: int) -> None:
        """Both QG copies' builds at this width once, on two layers' weights,
        throwaway KV and every request at P = 0, in an epoch of their own."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf
        from sgl_amd_mono.kimi_k3.gfx950.layer import scratch_region

        dev, nreq = self.runner.device, rows // mf.G_ROWS
        self.runner.step_begin()
        for e in self.entries[:2]:
            mf.mla_front_launch(
                e.keys[rows],
                prefix=torch.zeros(rows, mf.HIDDEN, dtype=torch.bfloat16, device=dev),
                blocks=None, w=e.w,
                kv=torch.zeros(64, mf.DQK, dtype=torch.uint8, device=dev),
                kv_idx=torch.zeros(8, dtype=torch.int32, device=dev),
                kv_ptr=torch.tensor([0] + [1] * nreq, dtype=torch.int32, device=dev),
                g_kv_ptr=torch.arange(nreq + 1, dtype=torch.int32, device=dev) * mf.G_ROWS,
                loc=torch.zeros(rows, dtype=torch.int64, device=dev),
                core=self.core[rows],
                scratch=scratch_region(self.runner.scratch, "front"),
                peers=self.runner.peer.addresses, rank=self.runner.dcp_rank,
                layer=e.i, epoch=self.runner.epoch,
            )  # fmt: skip
        torch.cuda.synchronize()

    def forward(self, e, positions=None, hidden_states=None, forward_batch=None,
                zero_allocator=None, **kwargs):  # fmt: skip
        """``KimiK3MLAAttention.forward`` of layer ``e``: at a front width's
        target verify, the front and SGLang's o_proj on its gated core."""
        rows = hidden_states.shape[0]
        if not (forward_batch.forward_mode.is_target_verify() and rows in self.rows):
            return type(e.attn).forward(
                e.attn, positions, hidden_states, forward_batch, zero_allocator, **kwargs
            )
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf
        from sgl_amd_mono.kimi_k3.gfx950.layer import scratch_region

        if e.first:
            # once a forward, before its first front: the steps' mailbox tags
            self.runner.step_begin()
        backend = get_attn_backend()
        backend = getattr(backend, "full_attn_backend", backend)
        _front_plan_skip(backend)
        fm = backend.forward_metadata
        kv = backend.kv_layers.read(e.attn.attn_mqa.layer_id).key()
        core = self.core[rows]
        mf.mla_front_launch(
            e.keys[rows], prefix=hidden_states, blocks=None, w=e.w,
            kv=kv.view(-1, mf.DQK), kv_idx=fm.cprr_kv_indices,
            kv_ptr=fm.cprr_kv_indptr, g_kv_ptr=fm.global_kv_indptr,
            loc=forward_batch.out_cache_loc, core=core,
            scratch=scratch_region(self.runner.scratch, "front"),
            peers=self.runner.peer.addresses,
            rank=self.runner.dcp_rank, layer=e.i, epoch=self.runner.epoch,
        )  # fmt: skip
        # the gate is in the core: o_proj's wrapper has no gate input to apply
        out, _ = e.attn.o_proj(core)
        return out


# The widest steps' fused_qkv_a and g_proj run as one GEMM on their
# concatenated weights: from 24 requests it is faster than the two (5.6 us at
# 144 rows, 8 at 180); below, the two separate tuned GEMMs are.
QKV_G_MIN_ROWS = 144


def _fp8_dtype():
    """SGLang's e4m3 for aiter's MLA decode (its q's dtype)."""
    from sglang.srt.layers.attention import aiter_backend as ab

    return ab.fp8_dtype


def _fuse_qkv_g(backbone) -> None:
    """Each MLA layer's fused_qkv_a and g_proj weights (both on the layer's
    input x, bf16) as row blocks of one [2112 + 1536, 7168] tensor, the
    modules' weights views of it (``self_attn._qkv_g_weight``): the wide path
    runs them as one GEMM. Every other reader sees the same weights."""
    for layer in backbone.layers:
        attn = getattr(layer, "self_attn", None)
        fused = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
        g = getattr(attn, "g_proj", None)
        if fused is None or g is None or hasattr(attn, "qkv_conv1d"):
            continue
        wq, wg = fused.weight, g.weight
        if wq.dtype != torch.bfloat16 or wg.dtype != torch.bfloat16 or wq.shape[1] != wg.shape[1]:
            continue
        cat = torch.cat([wq.data, wg.data]).contiguous()
        wq.data, wg.data = cat[: wq.shape[0]], cat[wq.shape[0] :]
        attn._qkv_g_weight = cat


# The widths whose absorb runs in the prologue (SGLang's prepare stopped after
# q_b): every wide width (tier 0: -6 to -11 us an MLA layer on the mixes,
# -3 to -5 at 100K).
ABSORB_MIN_ROWS = 36


class _SglTails:
    """The MLA prologue, the decode and the tail inside SGLang's own MLA
    attention at the verify widths past the front's, ``mla_tail.MIN_ROWS`` ..
    ``MAX_ROWS``. SGLang's ROCm MLA prepare runs (fused_qkv_a, the q / kv
    norms, q_b over all 96 heads, the absorb); then, in place of its core:
    ``mla_prologue`` (the decode's e4m3 q and this rank's KV records),
    ``mla_decode_b`` (its split plan once a step, from SGLang's DCP KV
    indices), ``mla_tail`` on its split partials (the reduce, W_UV, the DCP
    exchange, the LSE combine and the output gate), and SGLang's o_proj on
    the gated core. Every other step of these layers is SGLang's own
    forward. One static rule a width."""

    def __init__(self, runner, entries: list, rows: frozenset) -> None:
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode_b as mdb
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt

        dev = runner.device
        self.runner, self.entries, self.rows = runner, entries, rows
        self.core = {r: _front_core_buffer(r, dev) for r in rows}
        self.q8 = {r: torch.empty(r, mt.NHA, mt.mf.DQK, dtype=_fp8_dtype(), device=dev) for r in rows}
        self.out = {r: torch.empty(r, mt.NHA, mt.KV, dtype=torch.bfloat16, device=dev) for r in rows}
        self.lse = {r: torch.empty(r, mt.NHA, dtype=torch.float32, device=dev) for r in rows}
        # the decode's split partials and the reduce's maps, one set for every layer
        self.bufs = mdb.DecodeBuffers(dev)
        # the softmax scale of every MLA layer's gathered-head attention
        scales = {e.attn.attn_mqa_for_dcp_decode.scaling for e in entries}
        assert len(scales) == 1, scales
        self.dkey = mdb.MlaDecodeBuild(scale=scales.pop())

    @classmethod
    def install(cls, backbone, runner, vcs: dict, widths):
        """Every MLA layer's wide path at the widths the spec and the front
        leave to SGLang, its builds run once (every rank the same launches in
        the same order), then each layer's ROCm MLA core routed through it.
        None where no width or no layer qualifies."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt

        rows = frozenset(
            r for r in range(mt.MIN_ROWS, mt.MAX_ROWS + 1, mt.G_ROWS)
            if r not in widths and r > FRONT_MAX_ROWS
        )  # fmt: skip
        mla = [
            i for i, layer in enumerate(backbone.layers)
            if hasattr(layer, "self_attn") and not hasattr(layer.self_attn, "qkv_conv1d")
        ]  # fmt: skip
        if not rows or not mla or any(vcs.get(i) is None for i in mla):
            return None
        entries = [
            # consecutive layers take alternate A2A copies (mla_tail.peer_end)
            SimpleNamespace(
                i=i, attn=backbone.layers[i].self_attn, w_vc=vcs[i],
                key=mt.MlaTailBuild(par=n % 2), first=n == 0,
            )
            for n, i in enumerate(mla)
        ]  # fmt: skip
        self = cls(runner, entries, rows)
        self._warmup()
        for e in entries:
            e.attn.forward_absorb_rocm_core = functools.partial(self.core_forward, e)
            e.attn.forward_absorb_rocm_prepare = functools.partial(self.prepare, e)
            if getattr(e.attn, "_qkv_g_weight", None) is not None:
                e.gate = None
                e.qkvg = torch.nn.Module()
                e.qkvg.weight = torch.nn.Parameter(e.attn._qkv_g_weight, requires_grad=False)
                e.attn.prepare_qkv_latent = functools.partial(self.qkv_latent, e)
                e.attn._compute_output_gate = functools.partial(self.output_gate, e)
        logger.info(
            "monokernel: MLA prologue and tail in SGLang's MLA attention at %d .. %d rows (%d layers)",
            min(rows), max(rows), len(entries),
        )  # fmt: skip
        return self

    def _warmup(self) -> None:
        """Every build once: the prologue on no owned row, the decode's plan and
        decode on requests with no key, both A2A copies' tails on no reduce
        tile and every row empty, in an epoch of their own."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode_b as mdb
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_prologue as mp
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt
        from sgl_amd_mono.kimi_k3.gfx950.layer import scratch_region

        dev, s = self.runner.device, min(self.rows)
        f32, i32, bf = torch.float32, torch.int32, torch.bfloat16
        qb = torch.zeros(s, mt.NHA, mp.QHD, dtype=bf, device=dev)
        lat = torch.zeros(s, mt.mf.LATW, dtype=bf, device=dev)
        w_kc = self.entries[0].attn.w_kc_qrep
        # the builds the widths take
        for absorb in {r >= ABSORB_MIN_ROWS for r in self.rows}:
            mp.mla_prologue_launch(
                mp.MlaPrologueBuild(absorb=absorb), q_pe=qb[..., mp.QHD - mp.ROPE :],
                k_nope=lat[:, : mt.KV], k_pe=lat[:, mt.KV : mp.DQK],
                # a slot of another rank: no record written
                loc=torch.full((s,), self.runner.dcp_rank + 1, dtype=torch.int64, device=dev) % mt.DCP,
                q8=self.q8[s], kv=torch.zeros(8, mp.DQK, dtype=torch.uint8, device=dev),
                rank=self.runner.dcp_rank,
                **(
                    dict(q_b=qb, w_kc=w_kc) if absorb
                    else dict(q_nope_out=torch.zeros(s, mt.NHA, mt.KV, dtype=bf, device=dev))
                ),
            )  # fmt: skip
        nreq = s // mt.G_ROWS
        ptr = torch.zeros(nreq + 1, dtype=i32, device=dev)
        mdb.mla_decode_plan_launch(kv_ptr=ptr, g_ptr=ptr, nreq=nreq, bufs=self.bufs)
        mdb.mla_decode_launch(
            self.dkey, q8=self.q8[s], kv=torch.zeros(8, mp.DQK, dtype=torch.uint8, device=dev),
            kv_idx=torch.zeros(1, dtype=i32, device=dev), kv_ptr=ptr, g_ptr=ptr, rank=self.runner.dcp_rank,
            o=self.out[s], lse=self.lse[s], bufs=self.bufs,
        )  # fmt: skip
        self.runner.step_begin()
        for e in self.entries[:2]:
            mt.mla_tail_launch(
                e.key, s=s, logits=torch.zeros(1, mt.NHA, mt.KV, dtype=f32, device=dev),
                plse=torch.zeros(1, mt.NHA, dtype=f32, device=dev),
                indptr=torch.zeros(2, dtype=i32, device=dev),
                fmap=torch.zeros(2, dtype=i32, device=dev),
                pmap=torch.zeros(1, dtype=i32, device=dev),
                o=torch.zeros(s, mt.NHA, mt.KV, dtype=bf, device=dev),
                lse=torch.zeros(s, mt.NHA, dtype=f32, device=dev),
                empty=torch.ones(s, dtype=torch.bool, device=dev), w_vc=e.w_vc,
                gate=torch.zeros(s, mt.GATE, dtype=bf, device=dev),
                core=self.core[s], scratch=scratch_region(self.runner.scratch, "mla_tail"),
                peers=self.runner.peer.addresses, rank=self.runner.dcp_rank,
                layer=e.i, epoch=self.runner.epoch,
            )  # fmt: skip
        torch.cuda.synchronize()

    def prepare(self, e, positions, hidden_states, forward_batch, zero_allocator,
                llama_4_scaling=None, prev_topk_indices=None):  # fmt: skip
        """``forward_absorb_rocm_prepare`` of layer ``e``: at a wide width's
        target verify from ABSORB_MIN_ROWS rows, SGLang's prepare up to q_b
        (the latent, the fused q / kv norm, the replicated q_b), the absorb
        left to the prologue (``q_nope_out`` None, q_b's output in its place);
        else SGLang's."""
        attn = e.attn
        rows = hidden_states.shape[0]
        if not (
            forward_batch.forward_mode.is_target_verify() and rows in self.rows and rows >= ABSORB_MIN_ROWS
        ):
            return type(attn).forward_absorb_rocm_prepare(
                attn, positions, hidden_states, forward_batch, zero_allocator,
                llama_4_scaling, prev_topk_indices,
            )  # fmt: skip
        from sglang.srt.layers.layer_boundary import get_attn_tp_context
        from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm as frm

        assert attn.is_dcp_q_replicate_phase(forward_batch) and attn.q_b_proj_qrep_weight is not None
        q, latent = get_attn_tp_context().fetch_qkv_latent().split(
            [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim], dim=-1
        )
        k_nope = latent[..., : attn.kv_lora_rank]
        q, k_nope = frm.fused_qk_rmsnorm_bf16(
            q, attn.q_a_layernorm.weight, attn.q_a_layernorm.variance_epsilon,
            k_nope, attn.kv_a_layernorm.weight, attn.kv_a_layernorm.variance_epsilon,
        )  # fmt: skip
        k_nope = k_nope.unsqueeze(1)
        q = frm.a16w16_linear(q, attn.q_b_proj_qrep_weight).view(-1, attn.num_local_heads * attn.dcp_size, attn.qk_head_dim)
        _, q_pe, k_pe = attn._split_q_nope_pe(q, latent)
        return (
            q_pe, k_pe, None, k_nope, forward_batch, zero_allocator, positions, None,
            llama_4_scaling, q,
        )  # fmt: skip

    def qkv_latent(self, e, hidden_states, forward_batch):
        """``prepare_qkv_latent`` of layer ``e``: at a wide width's target
        verify from QKV_G_MIN_ROWS rows, fused_qkv_a and g_proj as one GEMM (the
        latent returned, g kept for ``output_gate``), else SGLang's."""
        rows = hidden_states.shape[0]
        attn = e.attn
        if not (
            forward_batch.forward_mode.is_target_verify() and rows in self.rows and rows >= QKV_G_MIN_ROWS
        ):
            return type(attn).prepare_qkv_latent(attn, hidden_states, forward_batch)
        fused = attn.fused_qkv_a_proj_with_mqa
        n = fused.weight.shape[0]
        # fused_qkv_a's own linear method on the concatenated weight
        out = fused.quant_method.apply(e.qkvg, hidden_states)
        e.gate = out[:, n:]
        return out[:, :n]

    def output_gate(self, e, hidden_states):
        """``_compute_output_gate`` of layer ``e``: g from the fused GEMM when
        this step's latent came from it (its gate branch, if forked, joined),
        else SGLang's."""
        attn = e.attn
        g, e.gate = e.gate, None
        if g is None:
            return type(attn)._compute_output_gate(attn, hidden_states)
        alt, attn._gate_pending_stream = attn._gate_pending_stream, None
        if alt is not None:
            torch.cuda.current_stream().wait_stream(alt)
        return g

    def core_forward(self, e, q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator,
                     positions, topk_indices, llama_4_scaling, q_nope_unabsorbed=None):  # fmt: skip
        """``forward_absorb_rocm_core`` of layer ``e``: at a wide width's
        target verify, the prologue, the decode (its plan before the first
        layer's), the tail and SGLang's o_proj on the gated core."""
        attn = e.attn
        rows = q_pe.shape[0]
        if not (forward_batch.forward_mode.is_target_verify() and rows in self.rows):
            return type(attn).forward_absorb_rocm_core(
                attn, q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator,
                positions, topk_indices, llama_4_scaling, q_nope_unabsorbed,
            )  # fmt: skip
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode_b as mdb
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_prologue as mp
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt
        from sgl_amd_mono.kimi_k3.gfx950.layer import scratch_region

        absorb = rows >= ABSORB_MIN_ROWS
        assert topk_indices is None and llama_4_scaling is None
        assert (q_nope_out is None) == absorb and (q_nope_unabsorbed is not None) == absorb
        if e.first:
            # once a forward, before its first tail: the steps' mailbox tags
            self.runner.step_begin()
        backend = get_attn_backend()
        backend = getattr(backend, "full_attn_backend", backend)
        fm = backend.forward_metadata
        # the gathered-head attention SGLang's DCP verify calls
        layer = attn.attn_mqa_for_dcp_decode
        assert fm.use_asm_cprr_verify and fm.cprr_kernel_heads == layer.tp_q_head_num == mt.NHA
        assert backend._resolve_fp8_kv_scale_float(layer, None) == 1.0
        kv = backend.kv_layers.read(layer.layer_id).key()
        q8 = self.q8[rows]
        mp.mla_prologue_launch(
            mp.MlaPrologueBuild(absorb=absorb), q_nope_out=q_nope_out, q_pe=q_pe,
            k_nope=k_nope, k_pe=k_pe, loc=forward_batch.out_cache_loc, q8=q8,
            kv=kv.view(torch.uint8).view(-1, mp.DQK), rank=self.runner.dcp_rank,
            **(dict(q_b=q_nope_unabsorbed, w_kc=attn.w_kc_qrep) if absorb else {}),
        )  # fmt: skip
        out, lse, bufs = self.out[rows], self.lse[rows], self.bufs
        assert layer.scaling == self.dkey.scale
        if e.first:
            # once a step: the split plan and the reduce's maps, from this rank's key counts
            mdb.mla_decode_plan_launch(
                kv_ptr=fm.cprr_kv_indptr, g_ptr=fm.global_kv_indptr, nreq=rows // mt.G_ROWS, bufs=bufs
            )  # fmt: skip
        mdb.mla_decode_launch(
            self.dkey, q8=q8, kv=kv.view(torch.uint8).view(-1, mp.DQK), kv_idx=fm.cprr_kv_indices,
            kv_ptr=fm.cprr_kv_indptr, g_ptr=fm.global_kv_indptr, rank=self.runner.dcp_rank,
            o=out, lse=lse, bufs=bufs,
        )  # fmt: skip
        gate = attn._compute_output_gate(attn._gate_hidden_states)
        attn._gate_hidden_states = None
        core = self.core[rows]
        mt.mla_tail_launch(
            e.key, s=rows, logits=bufs.logits, plse=bufs.plse, indptr=bufs.indptr, fmap=bufs.fmap,
            pmap=bufs.pmap, o=out, lse=lse, empty=fm.cprr_empty_rows, w_vc=e.w_vc,
            gate=gate, core=core, scratch=scratch_region(self.runner.scratch, "mla_tail"),
            peers=self.runner.peer.addresses, rank=self.runner.dcp_rank, layer=e.i,
            epoch=self.runner.epoch,
        )  # fmt: skip
        # the gate is in the core: o_proj's wrapper has no gate input to apply
        output, _ = attn.o_proj(core)
        if attn.next_skip_topk is None:
            return output
        return output, None


def _gu_interleaved(moe) -> bool:
    """Whether the experts carry the gate / up interleaved MXFP4 layout.

    ``is_shuffled`` alone cannot answer this: Mxfp4MoEMethod sets it on both of
    its branches and only the SiTU A8W4 branch interleaves. Reading those bytes
    with the wrong layout does not fail, it returns wrong numbers, so ask the
    same question Mxfp4MoEMethod asked.
    """
    from sglang.srt.layers.quantization.mxfp4 import (
        _aiter_situ_uses_gu_interleaved_weights,
    )

    if getattr(moe.experts.moe_runner_config, "activation", None) != "situ":
        return False
    return _aiter_situ_uses_gu_interleaved_weights()


def _up_shard(moe, rank: int):
    """This rank's rows of the replicated latent up projection. Slicing the
    first dimension of a contiguous weight keeps it contiguous: a view."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import UP_N

    return moe.routed_expert_up_proj.weight.narrow(0, rank * UP_N, UP_N)


def _k2_launch(
    runner, packed: _Layer, core, prefix, blocks, out, prefix_in=None
) -> None:
    """The launch itself: this layer's weights read in place, the activations
    the caller owns."""
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    module = packed.module
    moe, experts = module.mlp, module.mlp.experts
    k2.k2_launch(
        packed.key,
        core=core,
        w_o=module.self_attn.o_proj.weight,
        prefix=prefix,
        blocks=blocks,
        ares_nw=module.mlp_res_norm.weight,
        ares_qk=module.mlp_res_proj.weight.view(HIDDEN),
        in_nw=module.post_attention_layernorm.weight,
        w_gate=moe.gate.weight,
        bias=moe.gate.e_score_correction_bias,
        w_ld=moe.routed_expert_down_proj.weight,
        w_sgu=moe.shared_experts.gate_up_proj.weight,
        w_sd=moe.shared_experts.down_proj.weight,
        w13=experts.w13_weight,
        w13s=experts.w13_weight_scale,
        w2=experts.w2_weight,
        w2s=experts.w2_weight_scale,
        ln_w=moe.routed_expert_norm.weight,
        w_up=_up_shard(moe, runner.rank),
        out=out,
        prefix_in=prefix_in,
        **runner.launch_args(),
        # One tag a layer: a layer takes at most one launch in a step.
        layer=128 + packed.i,
    )


def _launch(runner, packed: _Layer, attn_res, core, forward_batch):
    """One layer, from the attention's TP-local core to the layer's output."""
    from sglang.srt.layers.layer_boundary.residual.batch import written_residual

    module = packed.module
    rows = core.shape[0]
    core = core.contiguous()
    # A block-write layer restarts the prefix, so it takes none.
    prefix_sum = (
        None if module.is_block_write_layer else written_residual(forward_batch)
    )
    _check(packed, core, prefix_sum, attn_res, rows)
    prefix = runner.prefix[:rows]
    out = torch.empty_like(prefix)
    # The kernel reads the stream from prefix_sum (lent out read-only), updates its
    # own prefix and writes out + prefix itself.
    _k2_launch(runner, packed, core, prefix, attn_res.block_residual, out, prefix_sum)
    return out


def _check(packed: _Layer, core, prefix_sum, attn_res, rows: int) -> None:
    """What the routing decision could not know, fail-loud.

    Every tensor reaches the kernel as a raw address, so a wrong dtype or
    stride is read as if it were the expected one. The step was routed for
    every rank at once, so falling back here is not an option.
    """
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    module = packed.module
    assert core.shape == (rows, CORE) and core.dtype == torch.bfloat16, (
        tuple(core.shape),
        core.dtype,
    )
    assert attn_res is not None, "the attention-residual stream is not open"
    # The kernel's two prefix modes are chosen at build time, so the caller's
    # mode has to be the one this layer was built for.
    assert (prefix_sum is None) == module.is_block_write_layer
    if prefix_sum is not None:
        assert (
            prefix_sum.shape == (rows, HIDDEN)
            and prefix_sum.dtype == torch.bfloat16
            and prefix_sum.is_contiguous()
        ), (tuple(prefix_sum.shape), prefix_sum.dtype)
    assert attn_res.num_valid_blocks == packed.key.nblocks, (
        attn_res.num_valid_blocks,
        packed.key.nblocks,
    )
    bank = attn_res.block_residual
    assert (
        bank.dtype == torch.bfloat16
        and bank.size(0) >= rows
        and bank.size(2) == HIDDEN
        and bank.stride(2) == 1
    ), (bank.dtype, tuple(bank.shape), bank.stride())


def _mla_covered(front, hidden, bank) -> bool:
    """An MLA front (``front``: its key and weights) takes a layer whose stream
    is a written tensor (not an owed contribution), with its bank open at the
    row count it was built for."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    key, _ = front
    return (
        bank is not None
        and isinstance(hidden, torch.Tensor)
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and hidden.dim() == 2
        and hidden.shape[1] == HIDDEN
        and bank.num_valid_blocks == key.nblocks
        and bank.block_residual.dtype == torch.bfloat16
    )


class _LanguageModelShim:
    """``general_mm_embed_routine``'s language model for a routed step.

    The routine reads the real model (``get_input_embeddings``, ``pp_group``)
    and then calls it; this hands the reads through and takes the call.
    """

    def __init__(self, built: _Built) -> None:
        self._built, self._lm = built, built.lm

    def __getattr__(self, name):
        return getattr(self._lm, name)

    def __call__(self, input_ids=None, forward_batch=None, input_embeds=None, **kw):
        return self._built.language_forward(
            input_ids, forward_batch, input_embeds, **kw
        )


class _Built:
    """One rank's monokernels: the runner, each layer's weights, and the forward."""

    def __init__(self, model, runner, by_width: dict, k1_by_width: dict) -> None:
        self.model = model
        self.lm = getattr(model, "language_model", model)
        self.backbone = self.lm.model
        # {launch width: {layer id: its packed launch}}; a layer's build
        # key carries the width, so each width has its own compiled kernel
        self.runner, self._by_width = runner, by_width
        # {launch width: that width's K1 layers as one batch}, None where K1
        # took no layer
        self._k1_by_width = k1_by_width
        # This step's shared K1 state and the layers that took it, set by
        # _backbone before the first layer and read by _core and finish().
        self._k1_state, self._k1_engaged = None, []
        # Read once: _core asks this for every layer of every step, and a
        # served step must not pay for a check that is off.
        self._check = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
        # The conv windows waiting for finish(), under that flag only.
        self._check_handoff = []
        self._shim = _LanguageModelShim(self)
        # The MLA front's outputs, one set a width (the layers run in sequence)
        dev = runner.device
        self._mla_out = {
            rows: _mla_outputs(rows, dev)
            for rows, packed in by_width.items()
            if any(p.mla is not None for p in packed.values())
        }
        self._front_buf = {
            rows: _front_core_buffer(rows, dev)
            for rows, packed in by_width.items()
            if any(p.front is not None for p in packed.values())
        }

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        """``KimiK3ForConditionalGeneration.forward`` for a routed step: SGLang's
        multimodal routine around the backbone, which it calls through the shim."""
        from sglang.srt.managers.mm_utils import general_mm_embed_routine
        from sglang.srt.managers.schedule_batch import Modality

        self._plan, self._positions = plan, positions
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self._shim,
            data_embedding_funcs={Modality.IMAGE: self.model.get_image_feature},
            positions=positions,
            pp_proxy_tensors=kwargs.get("pp_proxy_tensors"),
        )

    def language_forward(self, input_ids, forward_batch, input_embeds, **kw):
        """``KimiK3LinearForCausalLM.forward``: the backbone, then the head."""
        lm = self.lm
        hidden = self._backbone(input_ids, forward_batch, input_embeds)
        aux = None
        if lm.capture_aux_hidden_states:
            hidden, aux = hidden
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(self, input_ids, forward_batch, input_embeds):
        """``KimiK3LinearModel.forward``: the embedding, every layer, the final
        norm. Pipeline and DP attention are refused, so this rank is the first
        and the last, and the bank is always open."""
        from sglang.srt.layers.layer_boundary.residual.batch import final_norm, start
        from sglang.srt.models.kimi_k3 import AuxHiddenStateList, _cdiv
        from sglang.srt.utils.common import BumpAllocator

        from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

        m = self.backbone
        self.runner.step_begin()
        # Once a step, before the first layer: every K1 layer's conv window
        # staged in one write. _k1_engaged collects the layers that take it,
        # which is what finish() hands back for.
        k1_batch = self._k1_by_width[self._plan.key.rows]
        self._k1_state = k1_mod.prepare(k1_batch, forward_batch)
        self._k1_engaged = []
        self._check_handoff = []
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        start(forward_batch)
        zero_allocator = BumpAllocator(
            buffer_size=2 * (m.end_layer - m.start_layer),
            dtype=torch.float32,
            device=hidden.device,
        )
        m.attn_bank.open(hidden, _cdiv(m.end_layer, m.config.attn_res_block_size), None)
        # The bank's stream is captured after a layer, not at the next layer's
        # input (``captures_at_input`` is False whenever the bank is open).
        aux = AuxHiddenStateList()
        captures = m.dspark_layers_to_capture or ()
        for i in range(m.start_layer, m.end_layer):
            hidden = self._layer(i, hidden, forward_batch, zero_allocator)
            if i in captures:
                aux.append(m._dspark_capture_stream(i, hidden, forward_batch))
        # Once a step, after the last layer: the conv inputs handed back to
        # SGLang's speculative window, which only reads them at accept.
        k1_mod.finish(k1_batch, self._k1_state, self._k1_engaged)
        if self._check:
            from sgl_amd_mono.kimi_k3 import check as check_mod

            # The conv window is handed back once a step, so it is scored here
            # and not in the layer that produced it.
            check_mod.compare_handoff(self)
        hidden = final_norm(
            hidden, forward_batch, m._final_read, capture=None, skip_empty=True
        )
        m.attn_bank.close()
        if m.dspark_layers_to_capture is not None:
            return hidden, aux
        return hidden

    def _layer(self, i, hidden, forward_batch, zero_allocator):
        """One layer: the launch where it is covered, else the model's own."""
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        layer = self.backbone.layers[i]
        packed = self._by_width[self._plan.key.rows].get(i)
        if packed is None:  # the dense prefix keeps the model's forward
            return layer(
                positions=self._positions,
                hidden_states=hidden,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                capture_gathered=None,
            )
        bank = self.backbone.attn_bank.current
        core = self._core(layer, packed, hidden, bank, forward_batch, zero_allocator)
        out = _launch(self.runner, packed, bank, core, forward_batch)
        # The launch folded the residual in, so the stream is written rather
        # than finished through ffn_boundary.
        return set_written(out, forward_batch)

    def _core(self, layer, packed, hidden, bank, forward_batch, zero_allocator):
        """This layer's TP-local attention output, which the layer launch takes.

        A KDA layer takes the K1 launch where it is covered: K1 runs the
        attention-side seam, ``input_layernorm`` and the whole KDA attention in
        one launch, so it replaces both the boundary's prepare and the model's
        attention. K1 reads the residual stream and does not add to it -- the
        attention's contribution is added at the MLP seam, which is K2's -- so
        the stream is handed on unchanged, which is what the boundary's prepare
        would have left for K2's ``written_residual``.

        Everything else -- the full-attention layers, and a KDA layer K1
        declines -- keeps SGLang's path. K1 writes nothing before it declines,
        so that fallback is clean.
        """
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

        state = self._k1_state
        if packed.front is not None and _mla_covered(packed.front, hidden, bank):
            return self._front_core(layer, packed, hidden, bank, forward_batch)
        # At these widths SGLang's verify plan is not made: an MLA layer must
        # take the front
        assert packed.front is None or hidden.shape[0] not in _FRONT_ROWS, (
            "an MLA layer declined the DCP front at a width without SGLang's plan"
        )
        if packed.mla is not None and _mla_covered(packed.mla, hidden, bank):
            return self._mla_core(
                layer, packed, hidden, bank, forward_batch, zero_allocator
            )
        if packed.k1 is not None and k1_mod.covered(packed.k1, hidden, bank, state):
            ref = None
            if self._check:
                from sgl_amd_mono.kimi_k3 import check as check_mod

                # Before K1, on the same input and the caches as they were.
                ref = check_mod.reference(
                    self, layer, packed, hidden, bank, forward_batch, zero_allocator
                )
            set_written(hidden, forward_batch)
            self._k1_engaged.append(packed.i)
            core = k1_mod.launch(self.runner, packed.k1, hidden, bank, state)
            if self._check:
                from sgl_amd_mono.kimi_k3 import check as check_mod

                check_mod.compare(self, packed.i, core, ref, bank)
            return core
        x = layer.attn_boundary.prepare(hidden, forward_batch, capture_gathered=None)
        return self._pre_o_proj(layer, x, forward_batch, zero_allocator)

    def _front_core(self, layer, packed, hidden, bank, forward_batch):
        """An MLA layer under DCP8: the front launch, through its all-to-all,
        LSE combine and output gate, to the gated core, o_proj's input."""
        from sglang.srt.layers.layer_boundary.residual.batch import set_written
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf
        from sgl_amd_mono.kimi_k3.gfx950.layer import scratch_region

        key, w = packed.front
        rows = hidden.shape[0]
        core = self._front_buf[rows]
        attn = layer.self_attn
        # the MLA layers' backend: the hybrid model's full-attention one
        backend = get_attn_backend()
        backend = getattr(backend, "full_attn_backend", backend)
        _front_plan_skip(backend)
        fm = backend.forward_metadata
        kv = backend.kv_layers.read(attn.attn_mqa.layer_id).key()
        # The launch reads the stream and does not add to it, as K1 does.
        set_written(hidden, forward_batch)
        mf.mla_front_launch(
            key, prefix=hidden, blocks=bank.block_residual, w=w,
            kv=kv.view(-1, mf.DQK), kv_idx=fm.cprr_kv_indices,
            kv_ptr=fm.cprr_kv_indptr, g_kv_ptr=fm.global_kv_indptr,
            loc=forward_batch.out_cache_loc, core=core,
            scratch=scratch_region(self.runner.scratch, "front"),
            peers=self.runner.peer.addresses,
            rank=self.runner.dcp_rank, layer=packed.i, epoch=self.runner.epoch,
        )  # fmt: skip
        if key.write_idx >= 0:
            bank._count_write(None)  # the launch wrote the bank row
        return core

    def _mla_core(self, layer, packed, hidden, bank, forward_batch, zero_allocator):
        """An MLA layer: the projection front as one launch, then SGLang's MLA
        forward (RoPE, the KV write, the indexer's norms / quantization / logits /
        top-k, the sparse attention) on its outputs, up to o_proj's input."""
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
        from sgl_amd_mono.kimi_k3.gfx950.layer import scratch_region
        from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

        key, w = packed.mla
        rows = hidden.shape[0]
        out = self._mla_out[rows]
        # The launch reads the stream and does not add to it, as K1 does.
        set_written(hidden, forward_batch)
        mp.mla_pre_launch(
            key, prefix=hidden, blocks=bank.block_residual, w=w, out=out,
            scratch=scratch_region(self.runner.scratch, "mla_pre"),
            layer=packed.i, epoch=self.runner.epoch,
        )  # fmt: skip
        if key.write_idx >= 0:
            bank._count_write(None)  # the launch wrote the bank row
        off, nbytes = mp.scratch_layout(key)["x"]
        x = (
            scratch_region(self.runner.scratch, "mla_pre")[off : off + nbytes]
            .view(torch.bfloat16)
            .view(rows, HIDDEN)
        )
        attn = layer.self_attn
        attn.__dict__["_mono_pre"] = {
            "q": out["q"], "latent": out["latent"], "k_nope": out["k_nope"],
            "q_lora": out["q_lora"], "gate": out["gate"],
            "q_nope_out": out["qn"] if key.qn else None,
        }  # fmt: skip
        attn.indexer.__dict__["_mono_pre"] = {
            "q": out["idx_q"], "k": out["idx_k"], "w": out["idx_w"],
        }  # fmt: skip
        try:
            return self._pre_o_proj(layer, x, forward_batch, zero_allocator)
        finally:
            attn.__dict__.pop("_mono_pre", None)
            attn.indexer.__dict__.pop("_mono_pre", None)

    def _pre_o_proj(self, layer, x, forward_batch, zero_allocator):
        """``KimiK3DecoderLayer._run_self_attn`` stopped before o_proj: the
        attention's TP-local output, which is where the launch starts."""
        from sglang.srt.layers.layer_boundary import (
            AttentionInputs,
            get_attn_tp_context,
        )

        attn = layer.self_attn
        latent = getattr(attn, "prepare_qkv_latent", None)
        if latent is not None:
            # MLA layers with q_lora_rank read their latent inputs from here.
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(x, forward_batch, latent)
            )
        try:
            return attn.forward_pre_o_proj(
                hidden_states=x,
                positions=self._positions,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )
        finally:
            if latent is not None:
                get_attn_tp_context().clear_attn_inputs()
