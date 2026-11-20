# SPDX-License-Identifier: Apache-2.0
"""The KDA layers' in_proj, one copy a layer: packed in place.

K1 (both forms) reads in_proj in ``kda_wide.pack_w_in``'s layout, AITER's 16x16
preshuffle. The layer's merged in-projection is rewritten in that layout once
at build, so no second copy of it is kept (90 MB a layer a rank); only its
f_a | b | pad tail (144 rows, 2 MB) is kept plain beside it, for SGLang's own
tail GEMM. SGLang's own KDA attention on a packed layer:

* target verify: one ``kda_wide`` launch for the step's requests (its GEMM
  form past 8), on SGLang's x (after ``input_layernorm``) and its ReplaySSM
  state. A request's outputs are the bits K1's wide form gives it in any batch
  (the launch is batch invariant), so the bits do not change across the routed
  widths and the wider ones.
* every other mode (extend, decode): SGLang's own ``forward_pre_o_proj``, its
  q, k, v, g GEMM on the packed rows (AITER's preshuffled-weight GEMMs,
  ``Preshuffled``), its f_a | b GEMM on the plain tail.

The first covered layer is not packed: its prefix is owed by the dense layer in
front, so K1 never takes it and SGLang's attention reads its in_proj as loaded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

# verify requests SGLang's path takes on a packed layer, one launch
MAX_REQS = 64

# the SGLang names this reads
SGLANG_READS = {
    "srt/models/kimi_k3.py": (
        "forward_pre_o_proj",
        "fused_qkvg_proj",
        "f_a_proj",
        "b_proj",
        "_bfa_w",
        "_qkvgbfa_layer",
        "_qkvgbfa_sizes",
    ),
    "srt/layers/linear.py": ("quant_method",),
    "srt/model_executor/forward_mode.py": ("is_target_verify",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend",
        "req_to_token_pool",
        "state_layer",
        "forward_metadata",
    ),
    "srt/layers/attention/linear/kda_backend.py": (
        "verify_intermediate_state_indices",
    ),
    "srt/layers/attention/linear/utils.py": (
        "select_verify_intermediate_state_indices",
    ),
    "srt/mem_cache/memory_pool.py": (
        "intermediate_conv_window",
        "replayssm_rawv",
        "replayssm_rawk",
        "replayssm_g",
        "replayssm_beta",
    ),
    "srt/layers/radix_linear_attention.py": ("conv_weights",),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    "srt/mem_cache/kv/router.py": ("view_layer_id",),
}


# ---------------------------------------------------------------- layouts
@triton.jit
def _pack_chunks(x, xp, S, CROWS: tl.constexpr, XROWS: tl.constexpr):
    """x [S, 7168] bf16 -> chunk c's x rows (CROWS a chunk) in ``pack_x``'s layout
    for its own token tiles, at row c XROWS of ``xp``; rows past S zero.
    Program (chunk, K step, tile)."""
    c, ks, tt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows_c = tl.minimum(S - c * CROWS, CROWS)
    nt = (rows_c + 15) // 16
    if tt < nt:
        e = tl.arange(0, 1024)
        q, kg, row, k8 = e // 512, (e // 128) % 4, (e // 8) % 16, e % 8
        r = tt * 16 + row
        k = ks * 64 + q * 32 + kg * 8 + k8
        v = tl.load(x + (c * CROWS + r) * 7168 + k, mask=r < rows_c, other=0.0)
        tl.store(xp + c * XROWS * 7168 + (ks * nt + tt) * 1024 + e, v)


def pack_chunks(x: torch.Tensor, xp: torch.Tensor, crows: int) -> None:
    """x [S, HIDDEN] -> ``xp`` [chunks, >= 16 tiles, HIDDEN], chunk c's ``crows``
    rows as ``pack_x`` lays out its own tiles."""
    s, xrows = x.shape[0], xp.shape[1]
    assert x.is_contiguous() and x.shape[1] == HIDDEN and x.dtype == torch.bfloat16
    assert -(-s // crows) <= xp.shape[0] and -(-crows // 16) * 16 <= xrows
    _pack_chunks[(-(-s // crows), HIDDEN // 64, -(-crows // 16))](
        x, xp, s, CROWS=crows, XROWS=xrows
    )


# ---------------------------------------------------------------- prefill GEMM
# The q, k, v, g GEMM on the preshuffled rows, by M: (largest M, AITER kernel).
# ("asm", name): AITER's preshuffled-weight asm GEMM; ("fly", tile_m, tile_n,
# tile_k, waves_per_eu): AITER's FlyDSL preshuffle GEMM.
_ASM = "_ZN5aiter{}bf16gemm_{}E"
PRE_TILES = (
    (384, ("asm", _ASM.format(43, "fp32bf16_tn_128x64_bshuffle_splitk"))),
    (640, ("asm", _ASM.format(43, "fp32bf16_tn_128x64_bshuffle_splitk"))),
    (896, ("asm", _ASM.format(43, "fp32bf16_tn_160x64_bshuffle_splitk"))),
    (1280, ("fly", 64, 192, 64, 2)),
    # the 256 x 256 tile while its tiles fit one wave of CUs (6144 / 256 = 24 a row)
    (2730, ("asm", _ASM.format(36, "bf16_tn_256x256_bpreshuffle"))),
    (3584, ("fly", 96, 192, 128, 0)),
    (1 << 30, ("asm", _ASM.format(36, "bf16_tn_256x256_bpreshuffle"))),
)


class Preshuffled:
    """``fused_qkvg_proj``'s method on a packed layer: its weight is the packed
    buffer's q, k, v, g rows, AITER's 16x16 preshuffle (``is_shuffled``)."""

    def __init__(self) -> None:
        self._exe: dict = {}
        self._dev: dict = {}

    def _fly(self, n, k, tm, tn, tk, wpe):
        key = (n, k, tm, tn, tk, wpe)
        if key not in self._exe:
            from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm

            self._exe[key] = compile_preshuffle_gemm(
                N=n, K=k, tile_m=tm, tile_n=tn, tile_k=tk, in_dtype="bf16",
                out_dtype="bf16", waves_per_eu=wpe or None,
            )  # fmt: skip
        return self._exe[key]

    def apply(self, layer, x, bias=None):
        w = layer.weight
        assert bias is None and getattr(w, "is_shuffled", False) and x.dtype == torch.bfloat16
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        m, (n, k) = x2.shape[0], w.shape
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
        if m == 0:
            return out.view(*x.shape[:-1], n)
        cfg = next(c for top, c in PRE_TILES if m <= top)
        if cfg[0] == "asm":
            from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm

            gemm_a16w16_asm(x2, w, out, None, 1, cfg[1], True)
        else:
            import flydsl.expr as fx
            from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

            dev = x.device
            if dev not in self._dev:
                self._dev[dev] = (
                    torch.empty(0, dtype=torch.int32, device=dev),
                    torch.empty(1, dtype=torch.float32, device=dev),
                    torch.empty(0, dtype=torch.bfloat16, device=dev),
                )
            sem, sc, nobias = self._dev[dev]
            exe = self._fly(n, k, *cfg[1:])
            _run_compiled(
                exe, out.view(-1), out.view(-1), sem, x2.view(-1), w.view(-1), sc, sc,
                nobias, m, n, fx.Stream(torch.cuda.current_stream()),
            )  # fmt: skip
        return out.view(*x.shape[:-1], n)


PRESHUFFLED = Preshuffled()


# ---------------------------------------------------------------- the layers
class InProj:
    """The process's packed KDA layers and SGLang's path on them."""

    def __init__(self, device) -> None:
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda, kda_wide as kw

        self.device = device
        self.layers: dict[int, Any] = {}  # id(attn) -> the layer's packed in_proj
        self.first: Optional[int] = None  # the lowest packed layer id
        # the verify launches: one scratch and epoch word, x in pack_x's layout
        self.scratch = torch.zeros(
            kw.scratch_bytes(kw.gemm_key(MAX_REQS, kw.WT)), dtype=torch.uint8, device=device
        )
        self.epochs = torch.zeros(1, 8, dtype=torch.int32, device=device)
        self.xp = torch.zeros(1, MAX_REQS * kw.WT, HIDDEN, dtype=torch.bfloat16, device=device)
        self.step: dict = {}

    def pack(self, layer, i: int) -> torch.Tensor:
        """This layer's in_proj packed in place (once); SGLang's q, k, v, g GEMM
        pointed at the packed rows, its f_a | b views at a plain copy of the
        tail; SGLang's attention routed through ``pre_o_proj``."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda, kda_wide as kw

        attn = layer.self_attn
        if id(attn) in self.layers:
            return self.layers[id(attn)].w_in
        w = attn._qkvgbfa_layer.weight
        qkvg, tail = self._views(attn, w)
        with torch.no_grad():
            # SGLang's f_a | b GEMM reads the tail rows plain
            plain = w[kda.FA_0 :].clone()
            for obj, name, r0, n in tail:
                t = plain[r0 - kda.FA_0 : r0 - kda.FA_0 + n]
                if isinstance(getattr(obj, name), torch.nn.Parameter):
                    getattr(obj, name).data = t
                else:
                    setattr(obj, name, t)
            w.copy_(kw.pack_w_in(w))
        qkvg.is_shuffled = True
        attn.fused_qkvg_proj.quant_method = PRESHUFFLED
        # SGLang's one GEMM over the whole merged buffer reads it plain
        attn._qkvgbfa_sizes = None
        entry = layer_entry(
            i,
            {
                "w_in": w,
                "w_fb": attn.f_b_proj.weight,
                "conv_w": attn.attn.conv_weights,
                "a_log": attn.A_log.detach().reshape(-1),
                "dt_bias": attn.dt_bias,
                "on_w": attn.o_norm.weight.data.to(torch.bfloat16).contiguous(),
            },
            float(attn.o_norm.eps),
            float(attn.attn.lower_bound),
        )
        entry.attn = attn
        self.layers[id(attn)] = entry
        self.first = i if self.first is None else min(self.first, i)
        attn.forward_pre_o_proj = lambda **kw_: self.pre_o_proj(entry, **kw_)
        return w

    @staticmethod
    def _views(attn, w):
        """SGLang's views into the merged in_proj: the q, k, v, g weight (rows
        0 .. FA_0 - 1), and the tail views (owner, attribute, first row, rows)."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda

        row_bytes = w.stride(0) * w.element_size()
        tail = []
        for obj, name in (
            (attn.fused_qkvg_proj, "weight"),
            (attn.f_a_proj, "weight"),
            (attn.b_proj, "weight"),
            (attn, "_bfa_w"),
        ):
            t = getattr(obj, name, None)
            if t is None:
                continue
            off = t.data_ptr() - w.data_ptr()
            assert t.untyped_storage().data_ptr() == w.untyped_storage().data_ptr(), (
                f"{name} is not a view of the merged in_proj"
            )
            assert t.is_contiguous() and t.shape[1] == HIDDEN and off % row_bytes == 0
            r0 = off // row_bytes
            if obj is attn.fused_qkvg_proj:
                assert (r0, t.shape[0]) == (0, kda.FA_0)
            else:
                assert r0 >= kda.FA_0
                tail.append((obj, name, r0, t.shape[0]))
        return attn.fused_qkvg_proj.weight, tail

    # ------------------------------------------------------------ SGLang's path
    def pre_o_proj(self, entry, *, hidden_states, positions, forward_batch, zero_allocator):
        if forward_batch.forward_mode.is_target_verify():
            return self.verify(entry, hidden_states, forward_batch)
        attn = entry.attn
        return type(attn).forward_pre_o_proj(
            attn,
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

    def _step_state(self, forward_batch) -> dict:
        """The step's requests: slots and verify-window rows (int32), the pools."""
        from sglang.srt.layers.attention.linear.utils import (
            select_verify_intermediate_state_indices,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend().linear_attn_backend
        pool = backend.req_to_token_pool
        cache = pool.mamba_pool.mamba_cache
        assert getattr(cache, "replayssm_rawv", None) is not None, "needs the ReplaySSM ring"
        r = forward_batch.req_pool_indices.shape[0]
        assert r <= MAX_REQS, f"{r} requests, SGLang's path on a packed layer takes {MAX_REQS}"
        slots = backend.forward_metadata.mamba_cache_indices[:r]
        rows = select_verify_intermediate_state_indices(
            backend.verify_intermediate_state_indices,
            forward_batch.req_pool_indices,
            slots >= 0,
            pool.size,
        )
        return {
            "backend": backend,
            "cache": cache,
            "reqs": r,
            "slots": slots.to(torch.int32),
            "rows": rows[:r].to(torch.int32),
        }

    def verify(self, entry, x, forward_batch) -> torch.Tensor:
        """kda_wide over the step's requests on SGLang's pools."""
        if entry.i == self.first:
            # once a step, in the step's first packed layer: a step's layers
            # differ by slot, the steps by the epoch
            self.step = self._step_state(forward_batch)
            self.epochs[:, 0].add_(1)
        st = self.step
        if entry.mamba_idx is None:
            entry.mamba_idx = st["backend"].state_layer(
                entry.attn.attn.layer_id
            ).view_layer_id
        mi, cache = entry.mamba_idx, st["cache"]
        pools = SimpleNamespace(
            conv_state=cache.conv[0][mi].mT,
            rstate=cache.temporal[mi],
            conv_win=cache.intermediate_conv_window[0][mi],
            ring=tuple(
                t[mi]
                for t in (
                    cache.replayssm_rawk,
                    cache.replayssm_rawv,
                    cache.replayssm_g,
                    cache.replayssm_beta,
                )
            ),
            slots=st["slots"],
            rows=st["rows"],
        )
        return self.launch(entry, x, st["reqs"], pools)

    def launch(self, entry, x, reqs: int, pools) -> torch.Tensor:
        """kda_wide on requests 0 .. reqs - 1 of x [reqs L, HIDDEN]: the gated
        core output [reqs L, PROJ]."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda, kda_wide as kw

        rows = x.shape[0]
        qlen = rows // reqs
        assert rows == reqs * qlen and 1 <= qlen <= kw.WT and reqs <= MAX_REQS
        key = entry.key(reqs, qlen)
        nt = kw.tiles(key)
        pack_chunks(x, self.xp, rows)
        core = torch.empty(rows, kda.PROJ, dtype=torch.bfloat16, device=x.device)
        kw.kda_wide(
            key,
            x=self.xp[0, : 16 * nt],
            **entry.w,
            conv_state=pools.conv_state,
            rstate=pools.rstate,
            st_idx=pools.slots[:reqs],
            conv_win=pools.conv_win,
            win_row=pools.rows[:reqs],
            ring=pools.ring,
            core_out=core,
            scratch=self.scratch,
            layer=entry.i,
            epoch=self.epochs[0],
        )
        return core

    def warmup(self, qlen: int) -> None:
        """Compile every width's launch and the x layout kernel, on a packed
        layer's weights and throwaway state, before any capture."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda, kda_wide as kw

        if not self.layers or qlen < 1 or qlen > kw.WT:
            return
        f32, bf16, dev = torch.float32, torch.bfloat16, self.device

        def z(*shape, dt=bf16):
            return torch.zeros(*shape, dtype=dt, device=dev)

        chans = 3 * kda.PROJ
        pack_chunks(z(MAX_REQS * qlen, HIDDEN), self.xp, MAX_REQS * qlen)
        for entry in self.layers.values():
            for r in range(1, MAX_REQS + 1):
                key = entry.key(r, qlen)
                if key in _WARM:
                    continue
                _WARM.add(key)
                nt = kw.tiles(key)
                # a step a launch: one layer's widths share its slot
                self.epochs[:, 0].add_(1)
                kw.kda_wide(
                    key,
                    x=z(16 * nt, HIDDEN),
                    **entry.w,
                    conv_state=z(1, kda.CONV_W - 1, chans).mT,
                    rstate=z(1, kda.NH, kda.HD, kda.HD, dt=f32),
                    st_idx=z(r, dt=torch.int32),
                    conv_win=z(1, qlen, kda.CONV_W - 1, chans),
                    win_row=z(r, dt=torch.int32),
                    ring=(
                        z(1, kda.NH, qlen, kda.HD),
                        z(1, kda.NH, qlen, kda.HD),
                        z(1, kda.NH, qlen, kda.HD, dt=f32),
                        z(1, kda.NH, qlen, dt=f32),
                    ),
                    core_out=z(r * qlen, kda.PROJ),
                    scratch=self.scratch,
                    layer=entry.i,
                    epoch=self.epochs[0],
                )
        # the throwaway launches' tags are below every served step's
        self.scratch.zero_()
        torch.cuda.synchronize()


def layer_entry(i: int, w: dict, onorm_eps: float, lower_bound: float):
    """A packed layer: its launch slot, the tensors kda_wide reads (``w``), its
    build key a chunk width."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

    keys: dict = {}

    def key(reqs: int, qlen: int):
        if (reqs, qlen) not in keys:
            kw_ = dict(onorm_eps=onorm_eps, lower_bound=lower_bound)
            keys[(reqs, qlen)] = (
                kw.KdaWideBuild(tokens=reqs * qlen, qlen=qlen, **kw_)
                if reqs <= kw.WIDE_REQS
                else kw.gemm_key(reqs, qlen, **kw_)
            )
        return keys[(reqs, qlen)]

    return SimpleNamespace(i=i, w=w, w_in=w["w_in"], key=key, mamba_idx=None, attn=None)


_WARM: set = set()
_INPROJ: dict = {}


def for_device(device) -> InProj:
    """The process's one InProj (a rank has one device)."""
    if "p" not in _INPROJ:
        _INPROJ["p"] = InProj(device)
    return _INPROJ["p"]


def stock_pre_o_proj(attn):
    """SGLang's own ``forward_pre_o_proj`` on this layer (on a packed layer it
    reads the packed buffer as it is)."""
    p = _INPROJ.get("p")
    if p is None or id(attn) not in p.layers:
        return attn.forward_pre_o_proj
    return type(attn).forward_pre_o_proj.__get__(attn)
