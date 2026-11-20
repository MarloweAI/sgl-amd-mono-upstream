# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 DSpark drafter's attention: the draft block's query rows over the
request's whole FP8 draft KV, plus the block itself (non-causal), at TP8.

A rank holds 8 query heads and 2 KV heads of 64 (4 query heads a KV head). A
request's block has ``rows`` query rows (5: the anchor and 4 masks); the draft
KV holds its full context, not split by DCP.

Two launches:

    split   (TARGET + bs) units x 2 KV heads, one wave each. The batch's context
            is cut into units of ``chunk`` tokens, one ``chunk`` for every
            request: the smallest multiple of BLOCK_N that keeps the unit count
            at most TARGET + bs. A request so gets programs in proportion to its
            tokens: a long row is spread over the whole GPU, a short one takes
            as few as one. A program runs its unit for the KV head's 4 query
            heads x rows (32 rows with padding): bf16 dots over the dequantized
            FP8 keys and values, fp32 online softmax, the next tile's loads in
            flight while a tile computes. It writes the unit's normalized
            output and base-2 LSE (fp32). A unit's two KV-head programs run on
            one XCD, so each token's 128-byte K and V rows are fetched once.
    merge   bs x 8 heads x rows programs, one output row each: the row's unit
            partials (LSE merge), the block's own keys and values (bf16, every
            row sees every block key), and the final LSE merge.

Both launches compute the plan (chunk, units, the unit-to-request map) from
``kv_indptr`` on the device, so a captured graph serves any contexts.
"""

import torch
import triton
import triton.language as tl

H_Q = 8  # query heads a rank at TP8
H_KV = 2  # KV heads a rank at TP8
HEAD_DIM = 64
GROUP = H_Q // H_KV

TARGET = 1024  # units at most TARGET + bs
BLOCK_N = 64  # keys a tile
MERGE_BLOCK = 128  # unit partials a merge step
MERGE_WARPS = 4
XCDS = 8  # workgroups are dispatched round-robin over the XCDs

LOG2E = 1.4426950408889634


@triton.jit
def _plan(
    kv_indptr, bs, BS_PAD: tl.constexpr, TARGET: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Per request (BS_PAD lanes): KV start, length, units and inclusive unit
    prefix sum; and the unit length in tokens."""
    offs_b = tl.arange(0, BS_PAD)
    bmask = offs_b < bs
    lo = tl.load(kv_indptr + offs_b, mask=bmask, other=0)
    hi = tl.load(kv_indptr + offs_b + 1, mask=bmask, other=0)
    lens = hi - lo
    chunk = tl.cdiv(tl.cdiv(tl.sum(lens, 0), BLOCK_N), TARGET) * BLOCK_N
    chunk = tl.maximum(chunk, BLOCK_N)
    units = tl.cdiv(lens, chunk)
    cum = tl.cumsum(units, 0)
    return offs_b, lo, lens, units, cum, chunk


@triton.jit
def _slots(kv_indices, kv_lo, start, s1, BLOCK_N: tl.constexpr):
    offs_n = start + tl.arange(0, BLOCK_N)
    return tl.load(kv_indices + kv_lo + offs_n, mask=offs_n < s1, other=0)


@triton.jit
def _kv(
    k_base,
    v_base,
    loc,
    start,
    s1,
    stride_kt,
    stride_vt,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """A tile's FP8 keys and values, row-major (each token's 64 bytes contiguous)."""
    nmask = (start + tl.arange(0, BLOCK_N)) < s1
    offs_d = tl.arange(0, D)
    k = tl.load(
        k_base + loc[:, None] * stride_kt + offs_d[None, :],
        mask=nmask[:, None],
        other=0.0,
    )
    v = tl.load(
        v_base + loc[:, None] * stride_vt + offs_d[None, :],
        mask=nmask[:, None],
        other=0.0,
    )
    return k, v


@triton.jit
def _split(
    Q,
    K_Buf,
    V_Buf,
    kv_indptr,
    kv_indices,
    qo_indptr,
    P_O,
    P_L,
    bs,
    qk_scale,
    v_scale,
    nu,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    L: tl.constexpr,
    L_PAD: tl.constexpr,
    GROUP: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    BS_PAD: tl.constexpr,
    TARGET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    XCDS: tl.constexpr,
):
    # workgroups p and p + XCDS share an XCD: they take one unit's two KV heads
    p = tl.program_id(0)
    pid = (p // (2 * XCDS)) * XCDS + p % XCDS
    kvh = (p // XCDS) % 2
    offs_b, lo, lens, units, cum, chunk = _plan(kv_indptr, bs, BS_PAD, TARGET, BLOCK_N)
    if pid < tl.sum(units, 0):
        b = tl.sum((cum <= pid).to(tl.int32), 0)
        sel = offs_b == b
        u = pid - tl.sum(tl.where(sel, cum - units, 0), 0)
        kv_lo = tl.sum(tl.where(sel, lo, 0), 0)
        seq = tl.sum(tl.where(sel, lens, 0), 0)
        s0 = u * chunk
        s1 = tl.minimum(s0 + chunk, seq)
        k_base = K_Buf + kvh * stride_kh
        v_base = V_Buf + kvh * stride_vh
        loc = _slots(kv_indices, kv_lo, s0, s1, BLOCK_N)
        k_n, v_n = _kv(k_base, v_base, loc, s0, s1, stride_kt, stride_vt, BLOCK_N, D)
        loc = _slots(kv_indices, kv_lo, s0 + BLOCK_N, s1, BLOCK_N)

        R: tl.constexpr = GROUP * L_PAD
        offs_r = tl.arange(0, R)
        g = offs_r // L_PAD
        l = offs_r % L_PAD
        offs_d = tl.arange(0, D)
        q_lo = tl.load(qo_indptr + b)
        q = tl.load(
            Q
            + ((q_lo + l) * stride_qt + (kvh * GROUP + g) * stride_qh)[:, None]
            + offs_d[None, :],
            mask=(l < L)[:, None],
            other=0.0,
        )
        m = tl.zeros([R], dtype=tl.float32) - float("inf")
        s = tl.zeros([R], dtype=tl.float32)
        acc = tl.zeros([R, D], dtype=tl.float32)
        for start in tl.range(s0, s1, BLOCK_N):
            k, v = k_n, v_n
            k_n, v_n = _kv(
                k_base,
                v_base,
                loc,
                start + BLOCK_N,
                s1,
                stride_kt,
                stride_vt,
                BLOCK_N,
                D,
            )
            loc = _slots(kv_indices, kv_lo, start + 2 * BLOCK_N, s1, BLOCK_N)
            qk = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * qk_scale
            qk = tl.where(
                ((start + tl.arange(0, BLOCK_N)) < s1)[None, :], qk, float("-inf")
            )
            m_new = tl.maximum(m, tl.max(qk, 1))
            alpha = tl.exp2(m - m_new)
            pr = tl.exp2(qk - m_new[:, None])
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), v.to(tl.bfloat16))
            s = s * alpha + tl.sum(pr, 1)
            m = m_new
        row = (kvh * R + offs_r) * nu + pid
        tl.store(P_O + row[:, None] * D + offs_d[None, :], acc * (v_scale / s)[:, None])
        tl.store(P_L + row, m + tl.log2(s))


@triton.jit
def _merge(
    Q,
    K_Ext,
    V_Ext,
    O,
    kv_indptr,
    qo_indptr,
    P_O,
    P_L,
    bs,
    qk_scale,
    nu,
    stride_qt,
    stride_qh,
    stride_ket,
    stride_keh,
    stride_vet,
    stride_veh,
    stride_ot,
    stride_oh,
    L: tl.constexpr,
    L_PAD: tl.constexpr,
    GROUP: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    BS_PAD: tl.constexpr,
    TARGET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    hl = tl.program_id(1)
    h = hl // L
    i = hl % L
    kvh = h // GROUP
    R: tl.constexpr = GROUP * L_PAD
    # the row's partials are contiguous over units
    base = (kvh * R + (h % GROUP) * L_PAD + i) * nu
    offs_b, lo, lens, units, cum, chunk = _plan(kv_indptr, bs, BS_PAD, TARGET, BLOCK_N)
    sel = offs_b == b
    n = tl.sum(tl.where(sel, units, 0), 0)
    u0 = tl.sum(tl.where(sel, cum - units, 0), 0)
    offs_d = tl.arange(0, D)

    # the context: the unit partials, LSE-merged
    m = tl.zeros([1], dtype=tl.float32) - float("inf")
    s = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    for j0 in tl.range(0, n, MERGE_BLOCK, num_stages=2):
        offs_j = j0 + tl.arange(0, MERGE_BLOCK)
        jmask = offs_j < n
        row = base + u0 + offs_j
        lse = tl.load(P_L + row, mask=jmask, other=float("-inf"))
        po = tl.load(
            P_O + row[:, None] * D + offs_d[None, :], mask=jmask[:, None], other=0.0
        )
        m_new = tl.maximum(m, tl.max(lse, 0))
        alpha = tl.exp2(m - m_new)
        w = tl.exp2(lse - m_new)
        acc = acc * alpha + tl.sum(w[:, None] * po, 0)
        s = s * alpha + tl.sum(w, 0)
        m = m_new

    # the block: row i sees all L block keys
    q_lo = tl.load(qo_indptr + b)
    q = tl.load(Q + (q_lo + i) * stride_qt + h * stride_qh + offs_d).to(tl.float32)
    offs_l = tl.arange(0, L_PAD)
    lmask = offs_l < L
    ke = tl.load(
        K_Ext
        + ((q_lo + offs_l) * stride_ket + kvh * stride_keh)[:, None]
        + offs_d[None, :],
        mask=lmask[:, None],
        other=0.0,
    ).to(tl.float32)
    ve = tl.load(
        V_Ext
        + ((q_lo + offs_l) * stride_vet + kvh * stride_veh)[:, None]
        + offs_d[None, :],
        mask=lmask[:, None],
        other=0.0,
    ).to(tl.float32)
    qk = tl.sum(q[None, :] * ke, 1) * qk_scale
    qk = tl.where(lmask, qk, float("-inf"))
    m_b = tl.max(qk, 0)
    p = tl.exp2(qk - m_b)
    acc_b = tl.sum(p[:, None] * ve, 0)
    s_b = tl.sum(p, 0)

    # the context part is normalized: weigh it by exp2(m) s against the block's sum
    # (a request with no context has n = 0, s = 0)
    mm = tl.maximum(m, m_b)
    wc = tl.where(s > 0, tl.exp2(m - mm), 0.0)
    wb = tl.exp2(m_b - mm)
    out = (acc * wc + acc_b * wb) / (s * wc + s_b * wb)
    tl.store(
        O + (q_lo + i) * stride_ot + h * stride_oh + offs_d, out.to(O.dtype.element_ty)
    )


class DraftAttention:
    """The drafter's attention for graph batches of up to ``max_bs`` requests of
    ``rows`` block rows: ``__call__`` launches the split and the merge; the
    unit partials live in buffers allocated here."""

    def __init__(self, max_bs, rows, device):
        self.rows = rows
        self.max_bs = max_bs
        # tl.dot needs at least 16 rows: GROUP x L_PAD
        self.l_pad = max(triton.next_power_of_2(rows), 16 // GROUP)
        r = GROUP * self.l_pad
        # unit partials by (KV head, row, unit): a row's are contiguous for the merge
        self.nu = TARGET + max_bs
        self.p_o = torch.empty(
            (H_KV, r, self.nu, HEAD_DIM), dtype=torch.float32, device=device
        )
        self.p_l = torch.empty((H_KV, r, self.nu), dtype=torch.float32, device=device)

    def _common(self, bs):
        return dict(
            L=self.rows,
            L_PAD=self.l_pad,
            GROUP=GROUP,
            H_KV=H_KV,
            D=HEAD_DIM,
            BS_PAD=max(2, triton.next_power_of_2(bs)),
            TARGET=TARGET,
            BLOCK_N=BLOCK_N,
        )

    def split(
        self,
        q,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
    ):
        bs = qo_indptr.shape[0] - 1
        assert bs <= self.max_bs
        units = triton.cdiv(TARGET + bs, XCDS) * XCDS
        self.split_kernel = _split[(H_KV * units,)](
            q,
            k_buf,
            v_buf,
            kv_indptr,
            kv_indices,
            qo_indptr,
            self.p_o,
            self.p_l,
            bs,
            float(sm_scale) * LOG2E * float(k_scale),
            float(v_scale),
            self.nu,
            q.stride(0),
            q.stride(1),
            k_buf.stride(0),
            k_buf.stride(1),
            v_buf.stride(0),
            v_buf.stride(1),
            XCDS=XCDS,
            num_warps=1,
            num_stages=1,
            **self._common(bs),
        )

    def merge(self, q, k_ext, v_ext, o, qo_indptr, kv_indptr, sm_scale):
        bs = qo_indptr.shape[0] - 1
        self.merge_kernel = _merge[(bs, H_Q * self.rows)](
            q,
            k_ext,
            v_ext,
            o,
            kv_indptr,
            qo_indptr,
            self.p_o,
            self.p_l,
            bs,
            float(sm_scale) * LOG2E,
            self.nu,
            q.stride(0),
            q.stride(1),
            k_ext.stride(0),
            k_ext.stride(1),
            v_ext.stride(0),
            v_ext.stride(1),
            o.stride(0),
            o.stride(1),
            MERGE_BLOCK=MERGE_BLOCK,
            num_warps=MERGE_WARPS,
            num_stages=1,
            **self._common(bs),
        )

    def __call__(
        self,
        q,
        k_ext,
        v_ext,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
    ):
        self.split(
            q,
            k_buf,
            v_buf,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
        )
        self.merge(q, k_ext, v_ext, o, qo_indptr, kv_indptr, sm_scale)
        return o


_IMPLS = {}


def draft_verify_fwd(
    q,
    k,
    v,
    o,
    k_buf,
    v_buf,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_extend_len,
    k_scale,
    v_scale,
    sm_scale,
    *,
    logit_cap=0.0,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    max_bs=None,
    **unused,
):
    """The drafter's verify attention with the argument list of SGLang's Triton
    backend verify hook (``verify_shared_kv_fwd``, ``verify_mla_draft_fwd``). It
    serves the K3 drafter's shape only and refuses anything else."""
    assert (
        q.shape[1:] == (H_Q, HEAD_DIM)
        and k.shape[1:] == (H_KV, HEAD_DIM)
        and v.shape[1:] == (H_KV, HEAD_DIM)
    )
    assert not is_causal and custom_mask is None and sinks is None and not logit_cap
    assert (sliding_window_size or -1) <= 0 and (xai_temperature_len or -1) <= 0
    rows = int(max_extend_len)
    bs = qo_indptr.shape[0] - 1
    assert q.shape[0] == bs * rows
    key = (q.device, rows)
    impl = _IMPLS.get(key)
    if impl is None or impl.max_bs < bs:
        impl = _IMPLS[key] = DraftAttention(max(bs, max_bs or bs), rows, q.device)
    impl(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
    )
    return True
