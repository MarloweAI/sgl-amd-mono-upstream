# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-Flash-Next launches on gfx950: the buffers, the per-step pointer
tables and the launch sequence of a verify (and of an MTP draft step).

A verify of R requests x L tokens is 13 launches: layer 0 behind the prelude (the
token embedding and the n-gram history), layers 1-3 with layer 1's PLE in front,
then 11 launches of 3 GDN + 1 QSA layers, the last one followed by the model-level
hyper-connection mixer. Launch i reads the residual streams launch i - 1 wrote, so
the streams (and their sum-of-squares partials) alternate between two buffers.

Every buffer but the caches is allocated at construction, before SGLang sizes the
KV cache. A launch reads its weights, caches and the step's tensors through an
int64 pointer table, a row a layer plus a global row (``layout.P_*``,
``layout.G_*``); the tables of a batch size are built at its first, eager forward
(the decode graph runner's warmup) and kept, and a graph replays them. The cache
strides are compile-time constants of the kernel (``config.MonoBuild``), so a
batch size's launches compile at that forward too, not at build: the pools do
not exist yet then.

Word 0 of the epoch buffer is the step: the host moves it on after a step's last
launch, and every mailbox tag is ``step * LAUNCH_SLOTS + launch + 1``; words 1-5
are the guard block (``common/guard.py``).
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Optional

import torch

from sgl_amd_mono.common import guard

from . import layout as LY
from . import packing
from .config import GDN_CONV_DIM, HC_DIM, HIDDEN, MAX_T, ConvStrides, MonoBuild

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/model_executor/forward_context.py": (
        "get_req_to_token_pool",
        "get_token_to_kv_pool",
    ),
    "srt/mem_cache/memory_pool.py": (
        "req_to_token",
        "req_index_to_mamba_index_mapping",
        "layer_router",
        "mamba2_layer_cache",
        "conv",
        "temporal",
        "intermediate_ssm",
        "intermediate_conv_window",
        "ngram_pool",
        "get_key_buffer",
        "get_value_buffer",
    ),
    "srt/mem_cache/kv/router.py": ("read", "view", "view_layer_id", '"state"'),
    "srt/mem_cache/ple_state_pool.py": (
        "context",
        "intermediate_context",
        "layer_cache",
        "layer_intermediate_cache",
        '"short_conv"',
    ),
    "srt/mem_cache/qsa_kv_pool.py": (
        "get_qsa_key_state_buffer",
        "get_qsa_compressed_k_buffer",
        "qsa_rope_position_buffer",
        "qsa_num_request_slots",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "is_draft_extend_v2",
        "is_decode",
        "batch_size",
        "input_ids",
        "positions",
        "out_cache_loc",
        "req_pool_indices",
        "seq_lens",
    ),
}

# eager steps' tables kept besides the captured ones (eager steps of new tensors
# would otherwise grow the cache)
EAGER_TABLES = 64


def no_capture(what: str) -> None:
    """Raise inside a CUDA graph capture: builds, binds and allocations run eagerly."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"Qwen3.8-Flash-Next monokernel: {what} inside a graph capture; the "
            "decode graph runner's eager warmup prepares every batch size first"
        )


def compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    from flydsl.utils.env import CompileEnvManager

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    var = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(var)
    os.environ[var] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def launch_plans(kinds: list[str]) -> list[tuple[str, list[int]]]:
    """The verify's launches: (layer kinds, layer ids). Layer 0 alone behind the
    prelude, layers 1-3 behind layer 1's PLE, then 3 GDN + 1 QSA a launch."""
    n = len(kinds)
    plans = [(kinds[0], [0]), (",".join(kinds[1:4]), [1, 2, 3])]
    plans += [(",".join(kinds[b : b + 4]), list(range(b, b + 4))) for b in range(4, n, 4)]  # fmt: skip
    return plans


PLE_LAUNCH = 1  # the launch whose first layer (1) carries the PLE


class _Base:
    """The buffers both runners share: scratch mailboxes, the epoch, the tables."""

    def __init__(self, device, eos_token_id: int, fault):
        ensure_writable_flydsl_cache()
        self.device = device
        self.eos_token_id = int(eos_token_id)
        self.fault_site, self.fault_after = fault or (0, 0)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        self._kernels: dict = {}  # MonoBuild -> launcher
        self._tables: dict = {}  # step tensors -> (tables, keys)
        self._captured: set = set()  # table keys a graph replays: never dropped
        # every RoPE table a table points at: SGLang replaces its cached bf16 cast
        # when the table grows, and a graph still reads the old one
        self._held: dict = {}

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline (below the server's watchdog timeout)."""
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def kernel(self, key: MonoBuild):
        """The launcher of ``key``, built once (FlyDSL compiles it at its first call
        and keeps it in its disk cache)."""
        launch = self._kernels.get(key)
        if launch is None:
            no_capture(f"building {key}")
            from .kernel import build_qwen38_flash_next_kernel

            launch = self._kernels[key] = build_qwen38_flash_next_kernel(key)
        return launch

    def _cached(self, key):
        """The tables built for ``key``, kept for good once a graph captures them."""
        got = self._tables.get(key)
        if torch.cuda.is_current_stream_capturing():
            if got is None:
                no_capture("building the pointer tables of a new batch")
            self._captured.add(key)
        return got

    def _hold(self, t: torch.Tensor) -> None:
        self._held[t.data_ptr()] = t

    def _keep(self, key, value):
        if len(self._tables) >= EAGER_TABLES + len(self._captured):
            for k in [k for k in self._tables if k not in self._captured]:
                del self._tables[k]
        self._tables[key] = value
        return value

    def _upload(self, rows: list) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.int64).to(self.device)

    def _build_key(self, **kw) -> MonoBuild:
        return MonoBuild(
            eos_token_id=self.eos_token_id,
            fault_site=self.fault_site,
            fault_after=self.fault_after,
            **kw,
        )


def _step_reasons(fb, pos, R: int, L: int) -> list[str]:
    reasons = []
    for name in ("input_ids", "out_cache_loc", "req_pool_indices", "seq_lens"):
        t = getattr(fb, name)
        if t.dtype != torch.int64:
            reasons.append(f"forward_batch.{name} is {t.dtype}, the kernel reads int64")
    if pos.dtype != torch.int64 or pos.dim() not in (1, 2) or pos.stride(-1) != 1:
        reasons.append(f"positions {pos.dtype} of shape {tuple(pos.shape)}")
    if fb.req_pool_indices.numel() != R:
        reasons.append(f"{fb.req_pool_indices.numel()} request slots for {R} requests")
    if fb.input_ids.numel() != R * L:
        reasons.append(f"{fb.input_ids.numel()} rows for {R} requests x {L} tokens")
    return reasons


def _qsa_buffers(kvp, layer_id: int):
    return (
        kvp.get_key_buffer(layer_id),
        kvp.get_value_buffer(layer_id),
        kvp.get_qsa_key_state_buffer(layer_id),
        kvp.get_qsa_compressed_k_buffer(layer_id),
    )


def gdn_cache(kvp, layer_id: int):
    """A GDN layer's conv / SSM states and their verify intermediates."""
    r = kvp.layer_router.read(layer_id, "state")
    return r.view.mamba2_layer_cache(r.view_layer_id)


def ple_states(kvp, layer_id: int):
    """The PLE short-conv state and its verify intermediate."""
    r = kvp.layer_router.read(layer_id, "short_conv")
    return r.view.layer_cache(r.view_layer_id), r.view.layer_intermediate_cache(
        r.view_layer_id
    )


def ring_width(kvp) -> int:
    """Index-key ring slots a request in SGLang's QSA pool."""
    return kvp.qsa_rope_position_buffer.shape[0] // kvp.qsa_num_request_slots


def qsa_cache_reasons(kvp, layer_id: int) -> list[str]:
    k_buf, v_buf, ring, comp = _qsa_buffers(kvp, layer_id)
    reasons = []
    # 4 slots (one group): a verify row past the pending group overwrites one of its
    # members, and a rejected row's key stays there, so the group can later compress
    # with it (as in SGLang's own verify). 8 slots (two groups, sgl-project/sglang#47051)
    # keep every pending member.
    if ring_width(kvp) not in (4, 8):
        reasons.append(f"an index-key ring of {ring_width(kvp)} slots a request")
    for name, b in (("K", k_buf), ("V", v_buf)):  # 64-bit addressed (any pool size)
        if b.dtype != torch.bfloat16 or not b.is_contiguous() or b.shape[-2:] != (2, 256):  # fmt: skip
            reasons.append(
                f"layer {layer_id}: the {name} cache is {b.dtype} "
                f"{tuple(b.shape)}, the kernel writes contiguous bf16 [slots, 2, 256]"
            )
    for name, b in (("index-key ring", ring), ("compressed index keys", comp)):
        if not b.is_contiguous() or b.numel() * b.element_size() >= 2**32:
            reasons.append(
                f"layer {layer_id}: the {name} must be contiguous and under 4 GiB "
                "(32-bit buffer offsets)"
            )
    return reasons


class VerifyRunner(_Base):
    """Target verify of R <= ``max_reqs`` requests x L tokens (topk 1): the whole
    model, the embedding to the final mixer, in 13 launches."""

    def __init__(
        self,
        *,
        kinds: list[str],
        layers: dict,
        final: dict,
        ngram: dict,
        embed_w: torch.Tensor,
        cos_sin,
        ntok: int,
        max_reqs: int,
        eos_token_id: int,
        device,
        fault=None,
    ):
        super().__init__(device, eos_token_id, fault)
        self.kinds, self.layers, self.final, self.ngram = kinds, layers, final, ngram
        self.embed_w = embed_w
        self.cos_sin = cos_sin  # () -> SGLang's bf16 RoPE table (a cached cast)
        self.L, self.max_reqs = int(ntok), int(max_reqs)
        self.plans = launch_plans(kinds)
        S = self.max_reqs * self.L
        # one scratch for every R: a launch only accepts pairs tagged with its own
        # (step, launch) epoch
        nbytes = max(
            LY.scratch_layout(4 + 2, r * self.L, r)[1]
            for r in range(1, self.max_reqs + 1)
        )
        self.scratch = torch.zeros(nbytes // 4, dtype=torch.int32, device=device)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.h_bufs = [torch.empty(S, HC_DIM, **bf16) for _ in range(2)]
        self.ssq_bufs = [
            torch.zeros(S, 4, LY.NPART, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self.emb_buf = torch.empty(S, HIDDEN, **bf16)
        self.x_buf = torch.empty(S, HIDDEN, **bf16)

    def keys(self, R: int, meta: dict) -> list[MonoBuild]:
        """The build keys of a verify of R requests on caches of ``meta``'s strides."""
        last = len(self.plans) - 1
        return [
            self._build_key(
                plan=plan,
                nreq=R,
                ntok=self.L,
                ple_first=i == PLE_LAUNCH,
                final_mix=i == last,
                prelude=i == 0,
                first_src="embed" if i == 0 else "global",
                seq_p0=True,
                **meta,
            )
            for i, (plan, _) in enumerate(self.plans)
        ]

    def prepare_step(self, fb, pos):
        """The step's pointer tables and launch keys (built at the batch size's first,
        eager forward, and kept); ``pos``: the RoPE positions."""
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        rtp, kvp = get_req_to_token_pool(), get_token_to_kv_pool()
        R = int(fb.batch_size)
        cos = self.cos_sin()
        key = (
            R,
            fb.input_ids.data_ptr(),
            pos.data_ptr(),
            fb.out_cache_loc.data_ptr(),
            fb.req_pool_indices.data_ptr(),
            fb.seq_lens.data_ptr(),
            cos.data_ptr(),
            rtp.req_to_token.data_ptr(),
        )
        got = self._cached(key)
        if got is not None:
            return got
        no_capture("building the pointer tables")
        self._hold(cos)
        reasons = _step_reasons(fb, pos, R, self.L) + self.cache_reasons(rtp, kvp, R)
        if reasons:
            raise ValueError("Qwen3.8-Flash-Next monokernel: " + "; ".join(reasons))
        tables, meta = self._tables_for(fb, pos, rtp, kvp, cos, R)
        return self._keep(key, (tables, self.keys(R, meta)))

    def cache_reasons(self, rtp, kvp, R: int) -> list[str]:
        """What the kernel assumes of SGLang's caches for a verify of R requests:
        dtypes, layouts, and request slot r owning row r of every verify scratch."""
        L, reasons = self.L, []
        mamba_map = rtp.req_index_to_mamba_index_mapping
        if mamba_map.dtype not in (torch.int32, torch.int64):
            reasons.append(f"the mamba slot map is {mamba_map.dtype}")
        if rtp.req_to_token.dtype != torch.int32:
            reasons.append(f"req_to_token is {rtp.req_to_token.dtype}, not int32")
        ngram = rtp.ngram_pool
        ctx, ictx = ngram.context, ngram.intermediate_context
        if ctx is None or ctx.dtype != torch.int64 or not ctx.is_contiguous() or ctx.shape[1] != 2:  # fmt: skip
            reasons.append("the n-gram context must be contiguous int64 [slots, 2]")
        if (
            ictx is None
            or ictx.dtype != torch.int64
            or ictx.shape[0] < R
            or ictx.shape[1] < L
            or ictx.stride()[1:] != (2, 1)
        ):
            reasons.append(
                f"the n-gram intermediate context must be int64 [>= {R}, >= {L}, 2] "
                "(the verify scratch of speculative decoding)"
            )
        ssm_dtypes = set()
        for layer_id, kind in enumerate(self.kinds):
            if kind == "qsa":
                reasons += qsa_cache_reasons(kvp, layer_id)
                continue
            cache = gdn_cache(kvp, layer_id)
            conv, ssm = cache.conv[0], cache.temporal
            ssm_dtypes.add(ssm.dtype)
            if (
                conv.dtype != torch.bfloat16
                or not conv.is_contiguous()
                or conv[0].numel() * 2 != LY.CONV_SLOT_BYTES
            ):
                reasons.append(
                    f"layer {layer_id}: the conv state is {conv.dtype} "
                    f"{tuple(conv.shape)}, the kernel reads contiguous bf16 "
                    f"[slots, {GDN_CONV_DIM}, 3]"
                )
            if (
                ssm.dtype not in (torch.float32, torch.bfloat16)
                or not ssm.is_contiguous()
                or ssm[0].numel() * 4 != LY.SSM_SLOT_BYTES
            ):
                reasons.append(
                    f"layer {layer_id}: the SSM state is {ssm.dtype} "
                    f"{tuple(ssm.shape)}, the kernel reads contiguous f32 or bf16 "
                    "[slots, 48, 128, 128]"
                )
            issm = cache.intermediate_ssm
            icw = cache.intermediate_conv_window
            icw = icw[0] if icw else None
            if (
                issm is None
                or icw is None
                or issm.dtype != ssm.dtype
                or not issm[0].is_contiguous()
                or issm.shape[0] < R
                or issm.shape[1] < L
                or icw.dtype != torch.bfloat16
                or icw.shape[0] < R
                or icw.shape[1] < L
            ):
                # under --enable-linear-replayssm-spec SGLang keeps no per-draft state
                reasons.append(
                    f"layer {layer_id}: needs the per-draft-token intermediate SSM "
                    f"and conv states of {R} requests x {L} tokens, which the "
                    "kernel writes"
                )
            if layer_id == 1:
                state, istate = ple_states(kvp, layer_id)
                if (
                    istate is None
                    or istate.dtype != torch.bfloat16
                    or not istate[0].is_contiguous()
                    or istate.shape[0] < R
                    or istate.shape[1] < L
                    or istate.stride(1) != HC_DIM * LY.PLE_STATE_LEN
                ):
                    reasons.append(
                        "layer 1: needs the PLE's intermediate short-conv state, bf16 "
                        f"[>= {R}, >= {L}, {HC_DIM}, {LY.PLE_STATE_LEN}]"
                    )
                if (
                    state.dtype != torch.bfloat16
                    or not state.is_contiguous()
                    or state[0].numel() * 2 != LY.PLE_STATE_SLOT_BYTES
                ):
                    reasons.append(
                        "layer 1: the PLE short-conv state must be contiguous bf16 "
                        f"[slots, {HC_DIM}, {LY.PLE_STATE_LEN}]"
                    )
        if len(ssm_dtypes) > 1:
            reasons.append(f"the GDN layers' SSM states differ in dtype: {ssm_dtypes}")
        return reasons

    def _tables_for(self, fb, pos, rtp, kvp, cos, R: int):
        """One int64 table a launch: a row a layer, the global row, and the final
        mixer's row in the last launch."""
        mamba_map = rtp.req_index_to_mamba_index_mapping
        ngram_ctx = rtp.ngram_pool.context
        ngram_ictx = rtp.ngram_pool.intermediate_context

        def global_row(launch: int) -> list:
            g = packing.global_ptr_row(
                pos,
                fb.out_cache_loc,
                fb.req_pool_indices,
                fb.seq_lens,
                rtp.req_to_token,
                cos,
                kvp.qsa_rope_position_buffer,
                mamba_map,
            )
            g[LY.G_PLE_EMB] = self.emb_buf.data_ptr()
            g[LY.G_FIN_OUT] = self.x_buf.data_ptr()
            g[LY.G_INPUT_IDS] = fb.input_ids.data_ptr()
            g[LY.G_EMBED_W] = self.embed_w.data_ptr()
            g[LY.G_NGRAM_W] = self.ngram["table"].data_ptr()
            g[LY.G_NGRAM_SCALE] = self.ngram["scale"].data_ptr()
            g[LY.G_NGRAM_CTX] = ngram_ctx.data_ptr()
            g[LY.G_NGRAM_META] = self.ngram["meta"].data_ptr()
            g[LY.G_NGRAM_ICTX] = ngram_ictx[0].data_ptr()
            # launch i's streams: h_bufs[(i - 1) % 2] -> h_bufs[i % 2], with their
            # sum-of-squares partials
            g[LY.G_HIN_SSQ] = self.ssq_bufs[(launch - 1) % 2].data_ptr()
            g[LY.G_HOUT_SSQ] = self.ssq_bufs[launch % 2].data_ptr()
            return g

        last = self.layers[len(self.kinds) - 1]
        frow = [0] * LY.N_PTR
        frow[LY.P_HCA_NORM] = self.final["norm"].data_ptr()
        frow[LY.P_HCA_DOWN] = self.final["down"].data_ptr()
        frow[LY.P_HCA_UP] = self.final["up"].data_ptr()
        frow[LY.P_HCA_INJ] = last["hca"]["inj"].data_ptr()  # its combine gate is unused
        iconv = issm_row = ple_ist_row = ssm_bf16 = None
        tables = []
        for li, (_, layer_ids) in enumerate(self.plans):
            rows = []
            for layer_id in layer_ids:
                t = self.layers[layer_id]
                if self.kinds[layer_id] == "qsa":
                    rows.append(packing.qsa_ptr_row(t, *_qsa_buffers(kvp, layer_id)))
                    continue
                cache = gdn_cache(kvp, layer_id)
                conv, ssm = cache.conv[0], cache.temporal
                ssm_bf16 = ssm.dtype == torch.bfloat16
                row = packing.gdn_ptr_row(t, conv, ssm)
                issm = cache.intermediate_ssm
                icw = cache.intermediate_conv_window[0]
                # request slot r owns row r (SGLang's verify intermediate state indices)
                row[LY.P_ISSM] = issm.data_ptr()
                row[LY.P_ICONV] = icw.data_ptr()
                st = (icw.stride(1), icw.stride(2), icw.stride(3), icw.stride(0))
                if iconv not in (None, st) or issm_row not in (None, issm.stride(0)):
                    raise ValueError(
                        "Qwen3.8-Flash-Next monokernel: the GDN layers' verify "
                        "scratch strides differ"
                    )
                iconv, issm_row = st, issm.stride(0)
                if li == PLE_LAUNCH and layer_id == layer_ids[0]:
                    ple_state, ple_ist = ple_states(kvp, layer_id)
                    packing.add_ple_ptrs(row, t["ple"], ple_state)
                    row[LY.P_PLE_ISTATE] = ple_ist.data_ptr()
                    ple_ist_row = ple_ist.stride(0)
                rows.append(row)
            rows.append(global_row(li))
            if li == len(self.plans) - 1:
                rows.append(frow)
            tables.append(self._upload(rows))
        meta = dict(
            rtt_stride=rtp.req_to_token.stride(0),
            # 1-D text positions: all three MRoPE axes read the same element
            pos_stride=pos.stride(0) if pos.dim() == 2 else 0,
            map_i64=mamba_map.dtype == torch.int64,
            iconv=ConvStrides(*iconv),
            issm_row=issm_row,
            ple_ist_row=ple_ist_row,
            ictx_row=ngram_ictx.stride(0),
            ssm_bf16=bool(ssm_bf16),
            ring=ring_width(kvp),
        )
        return tables, meta

    def launch(self, fb, pos) -> tuple[torch.Tensor, torch.Tensor]:
        """A verify step with RoPE positions ``pos`` (SGLang's MRoPE positions where
        the model enables them) -> (the final mixer's output x [S, HIDDEN], the
        residual streams after the last layer [S, HC_DIM])."""
        S = int(fb.batch_size) * self.L
        if S > MAX_T:
            raise ValueError(f"Qwen3.8-Flash-Next monokernel: {S} rows past {MAX_T}")
        tables, keys = self.prepare_step(fb, pos)
        stream = torch.cuda.current_stream()
        h = self.h_bufs[1]  # unused by the prelude launch, which embeds the tokens
        for i, key in enumerate(keys):
            out = self.h_bufs[i % 2]
            self.kernel(key)(
                h.data_ptr(),
                out.data_ptr(),
                tables[i].data_ptr(),
                self.scratch.data_ptr(),
                self.epoch.data_ptr(),
                i,
                stream=stream,
            )
            h = out
        self.epoch[:1].add_(1)
        return self.x_buf[:S], h[:S]


class DraftRunner(_Base):
    """MTP (NEXTN) draft steps of R <= ``max_reqs`` requests, one launch each: the
    input fusion, the QSA layer and the draft's mixer. A draft extend runs R x
    ``ntok_extend`` rows, a draft decode step R x 1."""

    def __init__(
        self,
        *,
        views: dict,
        embed_w: torch.Tensor,
        cos_sin,
        layer_id: int,
        ntok_extend: int,
        max_reqs: int,
        eos_token_id: int,
        device,
        fault=None,
    ):
        super().__init__(device, eos_token_id, fault)
        self.views, self.embed_w, self.cos_sin = views, embed_w, cos_sin
        self.layer_id = layer_id
        self.L_ext, self.max_reqs = int(ntok_extend), int(max_reqs)
        S = self.max_reqs * self.L_ext
        nbytes = max(
            LY.scratch_layout(1 + 2, r * t, r)[1]
            for r in range(1, self.max_reqs + 1)
            for t in (1, self.L_ext)
        )
        self.scratch = torch.zeros(nbytes // 4, dtype=torch.int32, device=device)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.x_buf = torch.empty(S, HIDDEN, **bf16)
        # draft step i + 1 reads step i's output: a step writes the buffer its
        # input does not overlap, chosen by address so a capture repeats warmup
        self.hc_bufs = [torch.empty(S, HC_DIM, **bf16) for _ in range(2)]
        # a chain step's input_ids is a new tensor each step (topk_index.flatten()):
        # the launch reads a copy at a fixed address, so the tables a warmup built
        # are the ones its graph capture finds
        self.ids_buf = torch.zeros(S, dtype=torch.int64, device=device)
        # the same for hidden states that are not a previous step's output: with
        # rejection sampling select_top_k_tokens makes a new tensor each step
        self.hin_buf = torch.zeros(S, HC_DIM, **bf16)
        self.ssq_buf = torch.zeros(S, 4, LY.NPART, dtype=torch.float32, device=device)
        # a QSA-only launch reads no mamba slot
        self.mamba_map = torch.zeros(4096, dtype=torch.int32, device=device)

    def prepare_step(self, fb, hidden_states: torch.Tensor):
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        rtp, kvp = get_req_to_token_pool(), get_token_to_kv_pool()
        R, S = int(fb.batch_size), int(fb.input_ids.numel())
        cos = self.cos_sin()
        key = (
            R,
            S,
            fb.positions.data_ptr(),
            fb.out_cache_loc.data_ptr(),
            fb.req_pool_indices.data_ptr(),
            fb.seq_lens.data_ptr(),
            hidden_states.data_ptr(),
            cos.data_ptr(),
        )
        got = self._cached(key)
        if got is not None:
            return got
        no_capture("building the pointer tables")
        self._hold(cos)
        reasons = _step_reasons(fb, fb.positions, R, S // R)
        reasons += qsa_cache_reasons(kvp, self.layer_id)
        if reasons:
            raise ValueError("Qwen3.8-Flash-Next monokernel: " + "; ".join(reasons))
        g = packing.global_ptr_row(
            fb.positions,
            fb.out_cache_loc,
            fb.req_pool_indices,
            fb.seq_lens,
            rtp.req_to_token,
            cos,
            kvp.qsa_rope_position_buffer,
            self.mamba_map,
        )
        g[LY.G_INPUT_IDS] = self.ids_buf.data_ptr()
        g[LY.G_EMBED_W] = self.embed_w.data_ptr()
        g[LY.G_MTP_HIN] = hidden_states.data_ptr()
        g[LY.G_FIN_OUT] = self.x_buf.data_ptr()
        g[LY.G_HOUT_SSQ] = self.ssq_buf.data_ptr()
        rows = packing.mtp_rows(self.views, *_qsa_buffers(kvp, self.layer_id), g)
        pos = fb.positions
        build = self._build_key(
            plan="qsa",
            nreq=R,
            ntok=S // R,
            rtt_stride=rtp.req_to_token.stride(0),
            pos_stride=pos.stride(0) if pos.dim() == 2 else 0,
            final_mix=True,
            first_src="mtp",
            ring=ring_width(kvp),
        )
        return self._keep(key, (self._upload(rows), build))

    def launch(self, fb, hidden_states: torch.Tensor):
        """A draft step -> (the draft mixer's output x [S, HIDDEN], the
        hyper-connection hidden states [S, HC_DIM])."""
        S = int(fb.input_ids.numel())
        if not any(_overlaps(b, hidden_states) for b in self.hc_bufs):
            self.hin_buf[:S].copy_(hidden_states)
            hidden_states = self.hin_buf[:S]
        table, build = self.prepare_step(fb, hidden_states)
        hc = self.hc_bufs[1 if _overlaps(self.hc_bufs[0], hidden_states) else 0]
        self.ids_buf[:S].copy_(fb.input_ids.view(-1))
        self.kernel(build)(
            0,
            hc.data_ptr(),
            table.data_ptr(),
            self.scratch.data_ptr(),
            self.epoch.data_ptr(),
            0,
            stream=torch.cuda.current_stream(),
        )
        self.epoch[:1].add_(1)
        return self.x_buf[:S], hc[:S]


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors' memory ranges overlap."""
    a0, b0 = a.data_ptr(), b.data_ptr()
    a1 = a0 + a.numel() * a.element_size()
    b1 = b0 + b.numel() * b.element_size()
    return a0 < b1 and b0 < a1


def draft_eligible(fb, hidden_states: Optional[torch.Tensor], L_ext: int, max_reqs: int) -> bool:  # fmt: skip
    """A draft step the launch takes: R <= ``max_reqs`` whole requests, a draft
    extend of ``L_ext`` rows each or a draft decode of one, on the target's bf16
    hyper-connection hidden states."""
    R = int(fb.batch_size)
    hs = hidden_states
    if not 1 <= R <= max_reqs or hs is None:
        return False
    if hs.dtype != torch.bfloat16 or hs.shape[-1] != HC_DIM or not hs.is_contiguous():
        return False
    n = int(fb.input_ids.numel())
    if fb.forward_mode.is_draft_extend_v2():
        return n == R * L_ext and hs.shape[0] == n
    return fb.forward_mode.is_decode() and n == R and hs.shape[0] == R
