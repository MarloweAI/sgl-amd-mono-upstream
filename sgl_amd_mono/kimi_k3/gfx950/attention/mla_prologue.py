# SPDX-License-Identifier: Apache-2.0
"""``mla_prologue``: the inputs of aiter's DCP8 verify decode for one Kimi-K3
MLA layer at the tail's widths (``mla_tail.MIN_ROWS`` .. ``MAX_ROWS`` rows),
from SGLang's absorb and q_b outputs and the normed KV latent, in one launch
(``BLOCKS`` x ``THREADS``) in place of SGLang's q concatenation, its e4m3
casts, the KV record write and the window concatenation:

    q8   e4m3 [S, 96, 576]: head h of row t is the absorb output q_nope_out
         [t, h, 0 .. 511], then q_b's rope columns q_pe [t, h, 0 .. 63] (q_b
         [t, h, 128 .. 191])
    KV   e4m3 [pool rows, 576]: the record of each row this rank owns (its KV
         slot ``loc`` = local row x 8 + rank): the normed latent k_nope [t,
         0 .. 511], then the rope part k_pe [t, 0 .. 63], at local row loc / 8

Every cast is torch's bf16 -> float8_e4m3fn (round to nearest even, no
saturation: v_cvt_pk_fp8_f32 on the exact fp32 value), so both outputs are
SGLang's bytes. A thread takes 16 output bytes at a time, grid-strided.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.ops import rsrc
from sgl_amd_mono.common.plan import BLOCKS, THREADS, kernel_sources, key_tuple
from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

SOURCES = kernel_sources(__name__)

NHA = mf.NHA  # 96
KV = mf.KV  # 512
ROPE = mf.ROPE  # 64
DQK = mf.DQK  # 576
QHD = mf.QHD  # 192: q_b's columns a head (128 nope, 64 rope)
DCP = mf.DCP
PIECE = 16  # output bytes a thread a step: 16 values
Q_PIECES = DQK // PIECE  # 36 a (row, head)
K_PIECES = DQK // PIECE  # 36 a record

_BUILDS: dict = {}
_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class MlaPrologueBuild:
    """One build serves every width (S a launch argument)."""


def _bf16x8(base, i):
    """Eight bf16 values (16 bytes at bf16 index i) as fp32."""
    w = fx.Vector(bo.buffer_load(rsrc(base), i // 2, vec_width=4, dtype=T.i32))
    out = []
    for k in range(4):
        x = fx.Int32(w[k])
        out += [(x << 16).bitcast(fx.Float32), (x & fx.Int32(-65536)).bitcast(fx.Float32)]
    return out


def _e4m3x16(vals):
    return fx.Vector.from_elements([mf.e4m3x4(vals[4 * k : 4 * k + 4]) for k in range(4)], fx.Int32)


def build(key: MlaPrologueBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    keyed = key_tuple(key, SOURCES)
    nthreads = BLOCKS * THREADS

    @flyc.kernel(name="k3_mono_mla_prologue", known_block_size=[THREADS, 1, 1])
    def mla_prologue(
        qn: Int64, qp: Int64, kn: Int64, kn_rs: Int32, kp: Int64, kp_rs: Int32, loc: Int64,
        q8: Int64, kv: Int64, s: Int32, rank: Int32,
    ):  # fmt: skip
        _ = keyed
        gid = fx.block_idx.x * THREADS + fx.thread_idx.x
        nq = s * NHA * Q_PIECES
        # q: a (row, head, 16 columns) piece a thread a step
        for k in range((nq + nthreads - 1) // nthreads):
            p = gid + fx.Int32(k) * nthreads
            live = p < nq
            pc = fx.min(p, nq - 1)
            th = pc // Q_PIECES
            j = pc % Q_PIECES
            nope = j < KV // PIECE
            # both sources' pieces loaded (clamped), the right one kept
            a = _bf16x8(qn, th * KV + fx.min(j, KV // PIECE - 1) * PIECE)
            b = _bf16x8(qn, th * KV + fx.min(j, KV // PIECE - 1) * PIECE + 8)
            jr = fx.max(j - KV // PIECE, 0)
            ra = _bf16x8(qp, th * QHD + jr * PIECE)
            rb = _bf16x8(qp, th * QHD + jr * PIECE + 8)
            vals = [nope.select(x, y) for x, y in zip(a + b, ra + rb)]
            if live:
                bo.buffer_store(_e4m3x16(vals), rsrc(q8), pc * (PIECE // 4))
        # the records of the rows this rank owns: a (row, 16 bytes) piece a thread
        nk = s * K_PIECES
        for k in range((nk + nthreads - 1) // nthreads):
            p = gid + fx.Int32(k) * nthreads
            pc = fx.min(p, nk - 1)
            t = pc // K_PIECES
            j = pc % K_PIECES
            slot = fx.Int64(bo.buffer_load(rsrc(loc), t * 2, vec_width=1, dtype=T.i32))
            owned = (p < nk) & ((slot & fx.Int64(DCP - 1)) == fx.Int64(rank))
            nope = j < KV // PIECE
            a = _bf16x8(kn, t * kn_rs + fx.min(j, KV // PIECE - 1) * PIECE)
            b = _bf16x8(kn, t * kn_rs + fx.min(j, KV // PIECE - 1) * PIECE + 8)
            jr = fx.max(j - KV // PIECE, 0)
            ra = _bf16x8(kp, t * kp_rs + jr * PIECE)
            rb = _bf16x8(kp, t * kp_rs + jr * PIECE + 8)
            vals = [nope.select(x, y) for x, y in zip(a + b, ra + rb)]
            if owned:
                row = slot >> 3
                ptr = fx.inttoptr(
                    fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16),
                    kv + row * fx.Int64(DQK) + fx.Int64(j * PIECE),
                )
                fx.generic_store(ptr, _e4m3x16(vals))

    @flyc.jit
    def launch(
        qn: Int64, qp: Int64, kn: Int64, kn_rs: Int32, kp: Int64, kp_rs: Int32, loc: Int64,
        q8: Int64, kv: Int64, s: Int32, rank: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        mla_prologue(qn, qp, kn, kn_rs, kp, kp_rs, loc, q8, kv, s, rank).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


def mla_prologue_launch(key: MlaPrologueBuild, *, q_nope_out, q_pe, k_nope, k_pe, loc, q8, kv, rank):
    """Host launcher. ``q_nope_out`` bf16 [S, 96, 512] (contiguous: SGLang's
    absorb output) and ``q_pe`` [S, 96, 64] (the rope columns of q_b's
    contiguous [S, 96, 192] output, a view); ``k_nope`` / ``k_pe``:
    the rows' normed latent [S, 512] and rope part [S, 64] (bf16, unit column
    stride, any row stride); ``loc``: the rows' KV slots (int64); ``q8`` [S, 96,
    576] out (e4m3 or uint8); ``kv``: this rank's KV pool rows [., 576]."""
    s = q_nope_out.shape[0]
    bf = torch.bfloat16
    assert q_nope_out.dtype == bf and q_nope_out.is_contiguous() and tuple(q_nope_out.shape) == (s, NHA, KV)
    assert q_pe.dtype == bf and tuple(q_pe.shape) == (s, NHA, ROPE) and q_pe.stride() == (NHA * QHD, QHD, 1)
    k_nope, k_pe = k_nope.reshape(s, -1), k_pe.reshape(s, -1)
    assert k_nope.dtype == bf and k_nope.shape[1] == KV and k_nope.stride(1) == 1
    assert k_pe.dtype == bf and k_pe.shape[1] == ROPE and k_pe.stride(1) == 1
    assert k_nope.stride(0) % 8 == 0 and k_pe.stride(0) % 8 == 0, (k_nope.stride(), k_pe.stride())
    assert loc.dtype == torch.int64 and loc.numel() >= s
    assert q8.numel() == s * NHA * DQK and q8.element_size() == 1 and q8.is_contiguous()
    assert kv.element_size() == 1 and kv.stride(-2) == DQK
    build(key)(
        q_nope_out.data_ptr(), q_pe.data_ptr(), k_nope.data_ptr(), k_nope.stride(0),
        k_pe.data_ptr(), k_pe.stride(0), loc.data_ptr(), q8.data_ptr(), kv.data_ptr(), s, rank,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
