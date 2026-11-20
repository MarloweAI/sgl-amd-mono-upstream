# SPDX-License-Identifier: Apache-2.0
"""The KDA layers' in_proj, one copy a layer: packed in place.

K1 (both forms) reads in_proj in ``kda_wide.pack_w_in``'s layout. The layer's
merged in-projection is rewritten in that layout once at build, so no second
copy is kept (90 MB a layer a rank). SGLang's own KDA attention on a packed
layer then takes one of two paths, by forward mode:

* target verify: ``kda_wide`` in chunks of at most ``CHUNK`` requests, on
  SGLang's x (after ``input_layernorm``) and its ReplaySSM state. A request's
  outputs are the bits K1's wide form gives it in any batch (the launch is
  batch invariant), so the bits do not change across the routed widths and the
  wider ones.
* every other mode (extend, decode): in_proj unpacked into one buffer the
  process shares, which every packed layer's SGLang weights are views of; then
  SGLang's own ``forward_pre_o_proj``.

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

# requests a verify chunk: kda_wide's widest
CHUNK = 8
# verify requests SGLang's path takes on a packed layer: (MAX_REQS / CHUNK)
# chunks, each with its own epoch word (tags stay distinct across the chunks)
MAX_REQS = 64
EPOCH_STRIDE = MAX_REQS // CHUNK

# the SGLang names this reads
SGLANG_READS = {
    "srt/models/kimi_k3.py": (
        "forward_pre_o_proj",
        "fused_qkvg_proj",
        "f_a_proj",
        "b_proj",
        "_bfa_w",
        "_qkvgbfa_layer",
    ),
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
def _unpack(src, dst, HSTEPS: tl.constexpr, WAVES: tl.constexpr, G: tl.constexpr):
    """One (set, K step, wave slice, group) run of ``pack_w_in``'s layout (128
    16-B units: K half, K group, row) -> its 16 rows x 128 B of in_proj."""
    p = tl.program_id(0)
    g = p % G
    ws = (p // G) % WAVES
    hs = (p // (G * WAVES)) % HSTEPS
    s = p // (G * WAVES * HSTEPS)
    u = tl.arange(0, 128)
    q, kg, row = u // 64, (u // 16) % 4, u % 16
    w = tl.arange(0, 4)
    rows_k = 896  # 16-B units a row (HIDDEN / 8)
    o = ((s * G + g) * 16 + row) * rows_k + (ws * HSTEPS + hs) * 8 + q * 4 + kg
    v = tl.load(src + (p * 128 + u)[:, None] * 4 + w[None, :])
    tl.store(dst + o[:, None] * 4 + w[None, :], v)


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


def unpack_w_in(packed: torch.Tensor, out: torch.Tensor) -> None:
    """``pack_w_in``'s inverse, into ``out`` (both [NPROJ, HIDDEN] bf16)."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

    _unpack[(kw.SETS * kw.HSTEPS * kw.WAVES * kw.G,)](
        packed.view(torch.int32), out.view(torch.int32),
        HSTEPS=kw.HSTEPS, WAVES=kw.WAVES, G=kw.G,
    )  # fmt: skip


def pack_chunks(x: torch.Tensor, xp: torch.Tensor, crows: int) -> None:
    """x [S, HIDDEN] -> ``xp`` [chunks, >= 16 tiles, HIDDEN], chunk c's ``crows``
    rows as ``pack_x`` lays out its own tiles."""
    s, xrows = x.shape[0], xp.shape[1]
    assert x.is_contiguous() and x.shape[1] == HIDDEN and x.dtype == torch.bfloat16
    assert -(-s // crows) <= xp.shape[0] and -(-crows // 16) * 16 <= xrows
    _pack_chunks[(-(-s // crows), HIDDEN // 64, -(-crows // 16))](
        x, xp, s, CROWS=crows, XROWS=xrows
    )


# ---------------------------------------------------------------- the layers
class InProj:
    """The process's packed KDA layers and SGLang's path on them."""

    def __init__(self, device) -> None:
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda, kda_wide as kw

        self.device = device
        # every packed layer's SGLang in_proj views, unpacked by the layer that runs
        self.plain = torch.empty(kda.NPROJ, HIDDEN, dtype=torch.bfloat16, device=device)
        self.layers: dict[int, Any] = {}  # id(attn) -> the layer's packed in_proj
        self.first: Optional[int] = None  # the lowest packed layer id
        # the verify chunks: one scratch, chunk c's epoch word c
        crows = CHUNK * kw.WT
        self.crows = crows
        self.scratch = torch.zeros(
            kw.scratch_bytes(kw.KdaWideBuild(tokens=crows, qlen=kw.WT)),
            dtype=torch.uint8,
            device=device,
        )
        self.epochs = torch.zeros(EPOCH_STRIDE, 8, dtype=torch.int32, device=device)
        self.epochs[:, 0] = torch.arange(EPOCH_STRIDE, dtype=torch.int32, device=device)
        # chunk c's x at row c * crows, in its own tiles' pack_x layout
        self.xp = torch.zeros(
            MAX_REQS // CHUNK, crows, HIDDEN, dtype=torch.bfloat16, device=device
        )
        self.step: dict = {}

    def pack(self, layer, i: int) -> torch.Tensor:
        """This layer's in_proj packed in place (once); its SGLang weights made
        views of ``plain``; SGLang's attention routed through ``pre_o_proj``."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

        attn = layer.self_attn
        if id(attn) in self.layers:
            return self.layers[id(attn)].w_in
        w = attn._qkvgbfa_layer.weight
        views = self._views(attn, w)
        with torch.no_grad():
            w.copy_(kw.pack_w_in(w))
        for obj, name, off, n in views:
            plain = self.plain[off : off + n]
            if isinstance(getattr(obj, name), torch.nn.Parameter):
                getattr(obj, name).data = plain
            else:
                setattr(obj, name, plain)
        attn._qkvgbfa_layer = SimpleNamespace(weight=self.plain)
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
        """SGLang's views into the merged in_proj: (owner, attribute, first row, rows)."""
        row_bytes = w.stride(0) * w.element_size()
        out = []
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
            out.append((obj, name, off // row_bytes, t.shape[0]))
        return out

    # ------------------------------------------------------------ SGLang's path
    def stock(self, entry, **kw_):
        """SGLang's own attention on this layer, in_proj unpacked first."""
        unpack_w_in(entry.w_in, self.plain)
        attn = entry.attn
        return type(attn).forward_pre_o_proj(attn, **kw_)

    def pre_o_proj(self, entry, *, hidden_states, positions, forward_batch, zero_allocator):
        if forward_batch.forward_mode.is_target_verify():
            return self.verify(entry, hidden_states, forward_batch)
        return self.stock(
            entry,
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
            # once a step, in the step's first packed layer: a step's chunks and
            # layers differ by epoch word and slot, the steps by the epoch
            self.step = self._step_state(forward_batch)
            self.epochs[:, 0].add_(EPOCH_STRIDE)
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
        return self.chunks(entry, x, st["reqs"], pools)

    def chunks(self, entry, x, reqs: int, pools) -> torch.Tensor:
        """kda_wide on requests 0 .. reqs - 1 of x [reqs L, HIDDEN], CHUNK of them a
        launch: the gated core output [reqs L, PROJ]."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda, kda_wide as kw

        rows = x.shape[0]
        qlen = rows // reqs
        assert rows == reqs * qlen and 1 <= qlen <= kw.WT and reqs <= MAX_REQS
        pack_chunks(x, self.xp, CHUNK * qlen)
        core = torch.empty(rows, kda.PROJ, dtype=torch.bfloat16, device=x.device)
        for c, r0 in enumerate(range(0, reqs, CHUNK)):
            r = min(CHUNK, reqs - r0)
            key = entry.key(r, qlen)
            nt = kw.tiles(key)
            kw.kda_wide(
                key,
                x=self.xp[c, : 16 * nt],
                **entry.w,
                conv_state=pools.conv_state,
                rstate=pools.rstate,
                st_idx=pools.slots[r0 : r0 + r],
                conv_win=pools.conv_win,
                win_row=pools.rows[r0 : r0 + r],
                ring=pools.ring,
                core_out=core[r0 * qlen : (r0 + r) * qlen],
                scratch=self.scratch,
                layer=entry.i,
                epoch=self.epochs[c],
            )
        return core

    def warmup(self, qlen: int) -> None:
        """Compile every chunk width's launch and the two layout kernels, on a
        packed layer's weights and throwaway state, before any capture."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda, kda_wide as kw

        if not self.layers or qlen < 1 or qlen > kw.WT:
            return
        unpack_w_in(next(iter(self.layers.values())).w_in, self.plain)
        f32, bf16, dev = torch.float32, torch.bfloat16, self.device

        def z(*shape, dt=bf16):
            return torch.zeros(*shape, dtype=dt, device=dev)

        chans = 3 * kda.PROJ
        pack_chunks(z(MAX_REQS * qlen, HIDDEN), self.xp, CHUNK * qlen)
        for entry in self.layers.values():
            for r in range(1, CHUNK + 1):
                key = entry.key(r, qlen)
                if key in _WARM:
                    continue
                _WARM.add(key)
                nt = kw.tiles(key)
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
            keys[(reqs, qlen)] = kw.KdaWideBuild(
                tokens=reqs * qlen,
                qlen=qlen,
                onorm_eps=onorm_eps,
                lower_bound=lower_bound,
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
    """SGLang's own ``forward_pre_o_proj`` on this layer: in_proj unpacked first
    on a packed layer."""
    p = _INPROJ.get("p")
    entry = None if p is None else p.layers.get(id(attn))
    if entry is None:
        return attn.forward_pre_o_proj
    return lambda **kw_: p.stock(entry, **kw_)
