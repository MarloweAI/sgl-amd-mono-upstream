# SPDX-License-Identifier: Apache-2.0
"""KC: an index KV source's compressor past its projection -- SGLang's fused compression
(``c2_decode_norm_rope_store`` at ratio 2, ``c1_decode_norm_rope_store`` at ratio 1), the
indexer's ``wk`` and ``index_k_norm_rope_pack_store_split``, a row a CTA: at the end of the K1
that computes the projection (``index_query.run_cp``, ``index_query.run_kc``), or alone in a
launch (``build_kc``). A row:

- ratio 2: pools its kv with its partner's by their scores -- the previous row of its request
  in a verify block, else the pending row the state ring holds -- or, at an even position,
  parks its kv and score in the ring and stops (it completes no group); ratio 1 takes its kv;
- RMSNorm (``compressor.norm``) -> the bf16 pre-RoPE latent; RoPE of its tail at the group's
  position (``pos - 1`` at ratio 2); the FP4 fake quant of 16-element blocks; the V4 record
  (448 e4m3 with a ue8m0 scale a 64, then 64 bf16) at its compressed slot;
- the index key: ``wk`` of the latent, ``k_norm``, RoPE of its tail, the indexer's two FP4
  stages, into the split FP4 plane at its index slot.

The arithmetic is SGLang's kernels' step for step; sums run in this kernel's own order. A row
with no compressed slot (a padded row) writes nothing, nor does a row past the step's live
rows (``live``)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import (
    bf16_round,
    butterfly,
    fp8_roundtrip,
    global_load,
    global_ptr,
    hw_rsq,
    ld_i32,
)
from sgl_amd_mono.common.plan import THREADS, WAVES, kernel_sources

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

DIM = 512  # the compressed latent (head_dim)
ROPE = 64  # its RoPE tail
NOPE = DIM - ROPE
KEY = 128  # an index key (index_head_dim)
KEY_ROPE = 64
RECORD_DATA = 576  # V4: 448 e4m3 + 64 bf16
RECORD_SCALES = 8  # V4: 7 ue8m0 + a pad byte
PLANE_ROWS = 64  # the FP4 index plane's page
# the row index arrays, each int32 or int64 (``wide``: a bit each, in this order)
INDEX_ARRAYS = ("pos", "req", "raw_loc", "iloc")
# a launch's arguments: the projection (``kv``), then the rest (``K2iRunner.kc_args``)
KC_ARGS = (
    "kv", "per_req", "state", "ring", "norm", "eps", "freqs", "pos", "req", "raw_loc",
    "iloc", "cache", "page_bits", "page_bytes", "wk", "knorm", "keps", "payload", "pscale",
    "wide", "live",
)  # fmt: skip
# LDS words: the latent, the index key's K-quarter partials, the wave sums
LAT_AT, PART_AT, RED_AT = 0, DIM, DIM + 4 * KEY
LDS_WORDS = RED_AT + 2 * WAVES
# ``row_inputs``' loads, in order
ROW_INPUTS = ("pos", "loc", "islot", "wb", "wbk", "f", "fk", "kvr", "scr", "own")


def _f32_bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _bits_f32(x):
    return fx.Int32(x).bitcast(fx.Float32)


def ue8m0(x):
    """SGLang's ``cast_to_ue8m0``: the exponent byte of the smallest power of two >= x."""
    u = _f32_bits(x)
    return ((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))


def inv_ue8m0(e):
    return _bits_f32((254 - e) << 23)


def e2m1_code(x):
    """SGLang's ``fp4::e2m1_code``: RNE onto {0, .5, 1, 1.5, 2, 3, 4, 6}, saturated, sign kept."""
    mag = fx.min(abs(x), fx.Float32(6.0))
    step = (mag < 2.0).select(
        fx.Float32(0.5), (mag < 4.0).select(fx.Float32(1.0), fx.Float32(2.0))
    )
    q = fx.Float32(fmath.roundeven(mag / step)) * step
    idx = (q < 2.0).select(
        fx.Int32(q * 2.0), (q < 4.0).select(fx.Int32(q) + 2, fx.Int32(q * 0.5) + 4)
    )
    return idx | ((_f32_bits(x) >> 31) & 1) << 3


def e2m1_value(code):
    idx = code & 7
    fi = fx.Float32(idx)
    mag = (idx < 4).select(fi * 0.5, (idx < 6).select(fi - 2.0, (fi - 4.0) * 2.0))
    return ((code & 8) != 0).select(-mag, mag)


def e4m3_round(x):
    return fp8_roundtrip(x, x)[0]


def _amax(v, lanes):
    """Max of |v| over aligned runs of ``lanes`` lanes (xor butterfly)."""
    offs = [1 << k for k in range(lanes.bit_length() - 1)]
    return butterfly(abs(v), offs, lambda a, b: fx.max(a, b))


@flyc.jit
def _block_sum(c, v, slot):
    """Sum of f32 v over the CTA: waves' butterflies, then the waves in order."""
    w = butterfly(v, (1, 2, 4, 8, 16, 32))
    if c["lane"] == 0:
        fx.ptr_store(w.bitcast(fx.Int32), c["lds"] + RED_AT + slot * WAVES + c["wave"])
    gpu.barrier()
    total = fx.ptr_load(c["lds"] + RED_AT + slot * WAVES).bitcast(fx.Float32)
    for k in range_constexpr(1, WAVES):
        total = total + fx.ptr_load(c["lds"] + RED_AT + slot * WAVES + k).bitcast(
            fx.Float32
        )
    return total


def _st8(addr, v):
    ptr = fx.inttoptr(fx.Pointer[fx.Int8, fx.AddressSpace.Global, 1].ir_type, addr)
    fx.generic_store(ptr, fx.Int8(v))


def _st16(addr, v):
    ptr = fx.inttoptr(fx.Pointer[fx.Int16, fx.AddressSpace.Global, 2].ir_type, addr)
    fx.generic_store(ptr, fx.Int16(v))


def _rope_pair(v, f, e):
    """Element e's RoPE (pairs (2i, 2i + 1) as (real, imag), ``f`` this element's table
    entry): the partner from lane e ^ 1, rounded to bf16."""
    o = butterfly(v, (1,), lambda a, b: b)
    fo = butterfly(f, (1,), lambda a, b: b)
    even = (e & 1) == 0
    real, imag = even.select(v, o), even.select(o, v)
    fr, fi = even.select(f, fo), even.select(fo, f)
    return bf16_round(even.select(real * fr - imag * fi, real * fi + imag * fr))


@flyc.jit
def stage_record(c, a, v, e, slot):
    """Element e of the latent after RoPE (``v``): the FP4 fake quant of its 16-block, then
    its byte(s) of the V4 record at compressed slot ``slot``."""
    amax = _amax(v, 16)
    scale = e4m3_round(
        fx.min(fx.max(amax * (1.0 / 6.0), fx.Float32(2.0**-9)), fx.Float32(448.0))
    )
    q = e2m1_value(e2m1_code(v / scale + 0.0)) * scale
    page = slot >> a["page_bits"]
    sl = slot & ((fx.Int32(1) << a["page_bits"]) - 1)
    base = a["cache"] + fx.Int64(page) * fx.Int64(a["page_bytes"])
    data = base + fx.Int64(sl * RECORD_DATA)
    if e < NOPE:
        code = ue8m0(fx.max(fx.Float32(1e-4), _amax(q, 64)) / 448.0)
        x = fx.min(fx.max(q * inv_ue8m0(code), fx.Float32(-448.0)), fx.Float32(448.0))
        byte = rocdl.cvt_pk_fp8_f32(T.i32, x, x, fx.Int32(0), False)
        _st8(data + fx.Int64(e), byte & 0xFF)
        if (e % 64) == 0:
            scales = (
                base
                + fx.Int64(RECORD_DATA << a["page_bits"])
                + fx.Int64(sl * RECORD_SCALES)
            )
            _st8(scales + fx.Int64(e // 64), code)
    else:
        bits = (
            fx.Vector.from_elements([q, q], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        _st16(data + fx.Int64(NOPE + 2 * (e - NOPE)), bits & 0xFFFF)


def wk_order(wk):
    """The indexer's ``wk`` [128, 512] bf16 in the order KC's threads load it
    (``wk_loads``): thread t's (output t % 128, K quarter t / 128) k-th 8 elements at
    16-byte chunk k * THREADS + t, so a wave's loads are contiguous."""
    order = wk.view(KEY, 4, DIM // 4 // 8, 8).permute(2, 1, 0, 3)
    return order.reshape(DIM // 4 // 8, THREADS, 8).contiguous()


def wk_loads(a, tid):
    """This thread's ``wk`` words of the key (output tid % 128, K quarter tid / 128), from
    ``wk_order``: nothing they wait on."""
    return [
        global_load(a["wk"] + fx.Int64((k8 * THREADS + tid) * 16), 4)
        for k8 in range(DIM // 4 // 8)
    ]


@flyc.jit
def stage_key(c, a, slot, ws, wb, f):
    """The row's index key from its latent in LDS -> the FP4 plane at index slot ``slot``;
    this thread's ``wk`` words ``ws``, ``k_norm`` word ``wb`` and RoPE table entry ``f``
    were loaded before."""
    tid, lds = c["tid"], c["lds"]
    # wk: output o on threads o + 128 q, K quarter q each (a wave reads one latent word at
    # a time), then their sum (q0 + q1) + (q2 + q3)
    o, part = tid % KEY, tid // KEY
    acc = fx.Float32(0.0)
    k0 = part * (DIM // 4)
    for k8 in range_constexpr(DIM // 4 // 8):
        w = ws[k8]
        for j in range_constexpr(4):
            lo = (w[j] << 16).bitcast(fx.Float32)
            hi = (w[j] & fx.Int32(-65536)).bitcast(fx.Float32)
            acc = (
                acc
                + fx.ptr_load(lds + LAT_AT + k0 + 8 * k8 + 2 * j).bitcast(fx.Float32)
                * lo
            )
            acc = (
                acc
                + fx.ptr_load(lds + LAT_AT + k0 + 8 * k8 + 2 * j + 1).bitcast(
                    fx.Float32
                )
                * hi
            )
    fx.ptr_store(acc.bitcast(fx.Int32), lds + PART_AT + part * KEY + o)
    gpu.barrier()
    # k_norm, RoPE, the two FP4 stages: element e on thread e of waves 0, 1
    e = tid % KEY
    q = [fx.ptr_load(lds + PART_AT + k * KEY + e).bitcast(fx.Float32) for k in range(4)]
    y = bf16_round((q[0] + q[1]) + (q[2] + q[3]))
    sq = _block_sum(c, (tid < KEY).select(y * y, fx.Float32(0.0)), 1)
    inv = hw_rsq(sq * (1.0 / KEY) + _bits_f32(a["keps"]))
    wv = ((e & 1) == 0).select(
        (wb << 16).bitcast(fx.Float32), (wb & fx.Int32(-65536)).bitcast(fx.Float32)
    )
    v = bf16_round(wv * (y * inv))
    r = _rope_pair(v, f, e)
    v = (e >= KEY - KEY_ROPE).select(r, v)
    # fake quant: a ue8m0 scale a 32-block
    amax = _amax(v, 32)
    ex = ue8m0(fx.max(amax, fx.Float32(6.0 * 2.0**-126)) * (1.0 / 6.0))
    v = e2m1_value(e2m1_code(fmath.fma(v, inv_ue8m0(ex), fx.Float32(0.0)))) * _bits_f32(
        ex << 23
    )
    # the packer: its own scale floor, then the codes, -0 as 0
    amax = _amax(v, 32)
    ex = fx.min(
        fx.max(ue8m0(fx.max(amax / 6.0, fx.Float32(1.0e-4))), fx.Int32(1)),
        fx.Int32(254),
    )
    code = e2m1_code(v * inv_ue8m0(ex))
    code = ((code & 7) == 0).select(fx.Int32(0), code)
    pair = code | (butterfly(code, (1,), lambda x, y_: y_) << 4)
    if (tid < KEY) & (slot > 0) & ((e & 1) == 0):
        page, s = slot // PLANE_ROWS, slot % PLANE_ROWS
        j = e // 2  # payload byte: chunk j / 16, byte j % 16
        at = fx.Int64(page) * (4 * PLANE_ROWS * 16) + fx.Int64(
            (j // 16) * PLANE_ROWS * 16 + s * 16 + j % 16
        )
        _st8(a["payload"] + at, pair & 0xFF)
        if (e % 32) == 0:
            at = fx.Int64(page) * (4 * PLANE_ROWS) + fx.Int64(
                (e // 32) * PLANE_ROWS + (s % 16) * 4 + s // 16
            )
            _st8(a["pscale"] + at, ex)


def ld_index(a, name, i):
    """Entry i of index array ``name``, int32 or int64 by its bit of ``a["wide"]`` (word 0
    of an int64: a slot, a position or a request below 2 ** 31)."""
    wide = (a["wide"] >> INDEX_ARRAYS.index(name)) & 1
    return global_load(a[name] + fx.Int64(i) * fx.Int64(4 << wide), 1)


def kv_word(a, i):
    """Word i of the projection, at cache policy ``a["cm"]`` (K1's device-scope stores,
    where K1 runs the row)."""
    return ld_i32(a["kv"], i, a["cm"])


def row_inputs(c, a, row):
    """Row ``row``'s loads the projection does not address (``ROW_INPUTS``): its indices,
    the norms' words, then the RoPE entries and the ring's pending row at its position."""
    tid = c["tid"]
    e, ek = tid, tid % KEY
    two = c["r"] == 2
    i = {
        "pos": fx.Int32(ld_index(a, "pos", row)),
        "loc": fx.Int32(ld_index(a, "raw_loc", row)),
        "islot": fx.Int32(ld_index(a, "iloc", row)),
        "wb": global_load(a["norm"] + fx.Int64((e // 2) * 4), 1),
        "wbk": global_load(a["knorm"] + fx.Int64((ek // 2) * 4), 1),
    }
    rid = fx.Int32(ld_index(a, "req", row)) if two else fx.Int32(0)
    # at the row's position (an even ratio-2 row completes no group: its entry is never used)
    pos = i["pos"]
    rope_pos = fx.max(pos - 1, 0) if two else pos
    first = pos & ~(a["ratio"] - 1)
    i["f"] = _bits_f32(
        global_load(
            a["freqs"] + fx.Int64((rope_pos * ROPE + fx.max(e - NOPE, 0)) * 4), 1
        )
    )
    i["fk"] = _bits_f32(
        global_load(
            a["freqs"]
            + fx.Int64((first * KEY_ROPE + fx.max(ek - (KEY - KEY_ROPE), 0)) * 4),
            1,
        )
    )
    if two:
        ring = a["ring"]
        wofs = pos % ring
        rofs = (wofs == 0).select(ring - 1, wofs - 1)
        i["own"] = a["state"] + fx.Int64(rid * ring + wofs) * (2 * DIM * 4)
        prev = a["state"] + fx.Int64(rid * ring + rofs) * (2 * DIM * 4)
        i["kvr"] = global_load(prev + fx.Int64(e * 4), 1)
        i["scr"] = global_load(prev + fx.Int64((DIM + e) * 4), 1)
    else:
        i["kvr"] = i["scr"] = fx.Int32(0)
        i["own"] = fx.Int64(0)
    return tuple(i[n] for n in ROW_INPUTS)


@flyc.jit
def run_row(c, a, row, inputs, ws):
    """Row ``row``, its ``row_inputs`` and ``wk_loads`` issued: its projection (and at ratio
    2 a verify block's row before), then compression, the record, the index key."""
    tid = c["tid"]
    e = tid
    two = c["r"] == 2
    i = dict(zip(ROW_INPUTS, inputs))
    pos, loc = i["pos"], i["loc"]
    if const_expr(two):
        kv = _bits_f32(kv_word(a, row * 2 * DIM + e))
        sc = _bits_f32(kv_word(a, row * 2 * DIM + DIM + e))
        # a verify block's later rows pair with the row before, the first with the ring's
        inblock = (a["per_req"] > 1) & ((row % a["per_req"]) != 0)
        before = fx.max(row - 1, 0) * 2 * DIM
        kvo = _bits_f32(inblock.select(kv_word(a, before + e), i["kvr"]))
        sco = _bits_f32(inblock.select(kv_word(a, before + DIM + e), i["scr"]))
    else:
        w = kv_word(a, (row * DIM + (e & ~1)) // 2)
        kv = ((e & 1) == 0).select(
            (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)
        )
    # the latent before the norm, rounded to bf16
    if const_expr(two):
        even, odd = (pos & 1) == 0, (pos & 1) == 1
        own = i["own"]
        if (loc != 0) & even:
            fx.generic_store(global_ptr(own + fx.Int64(e * 4)), _f32_bits(kv))
            fx.generic_store(global_ptr(own + fx.Int64((DIM + e) * 4)), _f32_bits(sc))
        delta = sco - sc
        s = fmath.exp(-abs(delta))
        s0 = (delta > 0.0).select(fx.Float32(1.0), s)
        s1 = (delta > 0.0).select(s, fx.Float32(1.0))
        x = bf16_round((kvo * s0 + kv * s1) / (1.0 + s))
        live = (loc != 0) & odd
        slot = loc >> 1
    else:
        x = kv
        live = loc > 0
        slot = loc
    sq = _block_sum(c, x * x, 0)
    nf = hw_rsq(sq * (1.0 / DIM) + _bits_f32(a["eps"]))
    wb = i["wb"]
    wv = ((e & 1) == 0).select(
        (wb << 16).bitcast(fx.Float32), (wb & fx.Int32(-65536)).bitcast(fx.Float32)
    )
    lat = bf16_round(x * nf * wv)
    fx.ptr_store(lat.bitcast(fx.Int32), c["lds"] + LAT_AT + e)
    r = _rope_pair(lat, i["f"], e)
    v = (e >= NOPE).select(r, lat)
    if live:
        stage_record(c, a, v, e, slot)
    gpu.barrier()
    if live:
        stage_key(c, a, i["islot"], ws, i["wbk"], i["fk"])


def build_kc(tokens: int, ratio: int):
    """KC alone for a step of ``tokens`` rows of an index KV source of compress ratio
    ``ratio``, from a projection in memory."""
    assert ratio in (1, 2), ratio
    name = kernel_symbol("dsv41_mono_kc", s=tokens, r=ratio)
    keyed = (("tokens", tokens), ("ratio", ratio), ("sources", SOURCES))

    @fx.struct
    class KcWords:
        words: fx.Array[fx.Int32, LDS_WORDS, 16]

    @fx.union
    class KcLds:
        kc: KcWords

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kc(
        kv: Int64, per_req: Int32, state: Int64, ring: Int32, norm: Int64, eps: Int32,
        freqs: Int64, pos: Int64, req: Int64, raw_loc: Int64, iloc: Int64, cache: Int64,
        page_bits: Int32, page_bytes: Int32, wk: Int64, knorm: Int64, keps: Int32,
        payload: Int64, pscale: Int64, wide: Int32, live: Int32,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(KcLds).kc.peek().words.ptr
        a = {
            "kv": kv, "per_req": per_req, "state": state, "ring": ring, "norm": norm,
            "eps": eps, "freqs": freqs, "pos": pos, "req": req, "raw_loc": raw_loc,
            "iloc": iloc, "cache": cache, "page_bits": page_bits, "page_bytes": page_bytes,
            "wk": wk, "knorm": knorm, "keps": keps, "payload": payload, "pscale": pscale,
            "ratio": fx.Int32(ratio), "wide": wide, "cm": 0,
        }  # fmt: skip
        c = {"tid": tid, "lane": tid % 64, "wave": tid // 64, "lds": lds, "r": ratio}
        if fx.block_idx.x < live:
            row = fx.block_idx.x
            run_row(c, a, row, row_inputs(c, a, row), wk_loads(a, tid))

    @flyc.jit
    def launch(
        kv: Int64, per_req: Int32, state: Int64, ring: Int32, norm: Int64, eps: Int32,
        freqs: Int64, pos: Int64, req: Int64, raw_loc: Int64, iloc: Int64, cache: Int64,
        page_bits: Int32, page_bytes: Int32, wk: Int64, knorm: Int64, keps: Int32,
        payload: Int64, pscale: Int64, wide: Int32, live: Int32,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        kc(
            kv, per_req, state, ring, norm, eps, freqs, pos, req, raw_loc, iloc, cache,
            page_bits, page_bytes, wk, knorm, keps, payload, pscale, wide, live,
        ).launch(grid=(tokens,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
