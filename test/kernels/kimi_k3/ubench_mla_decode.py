# SPDX-License-Identifier: Apache-2.0
"""gfx950 facts the Kimi-K3 MLA decode is sized by, on one GPU:

* ``mfma``: e4m3 v_mfma_scale_f32_32x32x64_f8f6f4 throughput (independent
  chains a wave, 4 or 8 waves a CU), and the shader clock under that load;
* ``lds``: LDS read bandwidth a CU (ds_read_b128, conflict-free);
* ``layout``: the 32x32x64 operand and result layout (A, B from lane bytes,
  D lane l element i) against candidate mappings;
* ``tr8``: ds_read_b64_tr_b8's lane / byte mapping.
"""

import argparse

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from sgl_amd_mono.common.ops import rsrc

F8 = torch.float8_e4m3fn


def _memtime():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memtime", [], [], []))


def _realtime():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def mfma32(a, b, c):
    one = fx.Int32(127)
    return fx.Vector(rocdl.mfma_scale_f32_32x32x64_f8f6f4(T.vec(16, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))


def _ld8(ptr, i):
    lo = fx.Vector(bo.buffer_load(rsrc(ptr), i, vec_width=4, dtype=T.i32))
    hi = fx.Vector(bo.buffer_load(rsrc(ptr), i + 4, vec_width=4, dtype=T.i32))
    return fx.Vector.from_elements([lo[e] for e in range(4)] + [hi[e] for e in range(4)], fx.Int32)


_K = {}


def mfma_kernel(threads, chains):
    key = ("mfma", threads, chains)
    if key in _K:
        return _K[key]

    @flyc.kernel(name=f"ub_mfma_{threads}_{chains}", known_block_size=[threads, 1, 1])
    def ub(src: Int64, out: Int64, n: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        a = _ld8(src, (tid % 64) * 8)
        b = _ld8(src, 512 + (tid % 64) * 8)
        t0 = _memtime()
        r0 = _realtime()
        init = [fx.Float32(0.0)] * (16 * chains)
        for it, acc in range(fx.Int32(0), n, fx.Int32(1), init=init):
            vs = [fx.Vector.from_elements([fx.Float32(x) for x in acc[16 * c : 16 * c + 16]], fx.Float32) for c in range(chains)]
            vs = [mfma32(a, b, v) for v in vs]
            acc = yield [vs[c][e] for c in range(chains) for e in range(16)]
        acc = [fx.Float32(x) for x in acc]
        t1 = _memtime()
        r1 = _realtime()
        tot = acc[0]
        for e in range_constexpr(1, 16 * chains):
            tot = tot + acc[e]
        if tid % 64 == 0:
            w = (bid * (threads // 64) + tid // 64) * 4
            bo.buffer_store(fx.Int32(t1 - t0), rsrc(out), w)
            bo.buffer_store(fx.Int32(r1 - r0), rsrc(out), w + 1)
            bo.buffer_store(tot, rsrc(out), w + 2)

    @flyc.jit
    def launch(src: Int64, out: Int64, n: Int32, blocks: Int32, stream: fx.Stream = fx.Stream(None)):
        ub(src, out, n).launch(grid=(blocks,), block=(threads,), stream=stream)

    _K[key] = launch
    return launch


LDS_WORDS = 16384  # 64 KB


def lds_kernel(threads, reads):
    key = ("lds", threads, reads)
    if key in _K:
        return _K[key]

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int32, LDS_WORDS, 16]

    @flyc.kernel(name=f"ub_lds_{threads}_{reads}", known_block_size=[threads, 1, 1])
    def ub(out: Int64, n: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        p = lds.buf.ptr
        for j in range_constexpr(LDS_WORDS // threads // 4):
            fx.ptr_store(fx.Vector.from_elements([tid + j, tid, j, bid], fx.Int32), p + 4 * (tid + threads * j))
        rocdl.s_barrier()
        t0 = _memtime()
        init = [fx.Int32(0)] * 4
        for it, acc in range(fx.Int32(0), n, fx.Int32(1), init=init):
            x = [fx.Int32(v) for v in acc]
            for r in range_constexpr(reads):
                off = (4 * (tid + threads * r) + 4 * (it % 2) * 64) % LDS_WORDS
                v = fx.Vector(llvm.load(T.vec(4, T.i32), (p + off).llvm_ptr, alignment=16))
                x = [x[e] ^ fx.Int32(v[e]) for e in range(4)]
            acc = yield x
        acc = [fx.Int32(v) for v in acc]
        t1 = _memtime()
        if tid % 64 == 0:
            w = (bid * (threads // 64) + tid // 64) * 4
            bo.buffer_store(fx.Int32(t1 - t0), rsrc(out), w)
            bo.buffer_store(acc[0] ^ acc[1] ^ acc[2] ^ acc[3], rsrc(out), w + 1)

    @flyc.jit
    def launch(out: Int64, n: Int32, blocks: Int32, stream: fx.Stream = fx.Stream(None)):
        ub(out, n).launch(grid=(blocks,), block=(threads,), stream=stream)

    _K[key] = launch
    return launch


def layout_kernel():
    if "layout" in _K:
        return _K["layout"]

    @flyc.kernel(name="ub_layout", known_block_size=[64, 1, 1])
    def ub(a: Int64, b: Int64, out: Int64):
        lane = fx.thread_idx.x
        d = mfma32(_ld8(a, lane * 8), _ld8(b, lane * 8), fx.Vector.filled(16, 0.0, fx.Float32))
        for i in range_constexpr(16):
            bo.buffer_store(fx.Float32(d[i]), rsrc(out), lane * 16 + i)

    @flyc.jit
    def launch(a: Int64, b: Int64, out: Int64, stream: fx.Stream = fx.Stream(None)):
        ub(a, b, out).launch(grid=(1,), block=(64,), stream=stream)

    _K["layout"] = launch
    return launch


def mfma16(a, b, c):
    one = fx.Int32(127)
    return fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))


def layout16_kernel():
    if "layout16" in _K:
        return _K["layout16"]

    @flyc.kernel(name="ub_layout16", known_block_size=[64, 1, 1])
    def ub(a: Int64, b: Int64, out: Int64):
        lane = fx.thread_idx.x
        d = mfma16(_ld8(a, lane * 8), _ld8(b, lane * 8), fx.Vector.filled(4, 0.0, fx.Float32))
        for i in range_constexpr(4):
            bo.buffer_store(fx.Float32(d[i]), rsrc(out), lane * 4 + i)

    @flyc.jit
    def launch(a: Int64, b: Int64, out: Int64, stream: fx.Stream = fx.Stream(None)):
        ub(a, b, out).launch(grid=(1,), block=(64,), stream=stream)

    _K["layout16"] = launch
    return launch


def tr8_kernel():
    if "tr8" in _K:
        return _K["tr8"]

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int32, 1024, 16]

    @flyc.kernel(name="ub_tr8", known_block_size=[64, 1, 1])
    def ub(src: Int64, addr: Int64, out: Int64):
        lane = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        p = lds.buf.ptr
        for j in range_constexpr(16):
            fx.ptr_store(fx.Int32(bo.buffer_load(rsrc(src), lane + 64 * j, vec_width=1, dtype=T.i32)), p + (lane + 64 * j))
        rocdl.s_barrier()
        w = fx.Int32(bo.buffer_load(rsrc(addr), lane, vec_width=1, dtype=T.i32))  # byte address
        r = fx.Vector(rocdl.ds_read_tr8_b64(fx.Vector.make_type(2, fx.Int32), (p + w // 4).llvm_ptr).result)
        bo.buffer_store(fx.Int32(r[0]), rsrc(out), 2 * lane)
        bo.buffer_store(fx.Int32(r[1]), rsrc(out), 2 * lane + 1)

    @flyc.jit
    def launch(src: Int64, addr: Int64, out: Int64, stream: fx.Stream = fx.Stream(None)):
        ub(src, addr, out).launch(grid=(1,), block=(64,), stream=stream)

    _K["tr8"] = launch
    return launch


def bench_mfma(dev):
    src = torch.randint(0, 0x30, (1024,), dtype=torch.int32, device=dev)  # small positive e4m3 bytes
    src = (src * 0x01010101).contiguous()
    for threads, chains in ((256, 4), (256, 8), (512, 4), (512, 8)):
        out = torch.zeros(256 * (threads // 64) * 4, dtype=torch.int32, device=dev)
        n = 4000
        k = mfma_kernel(threads, chains)
        k(src.data_ptr(), out.data_ptr(), n, 256)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        k(src.data_ptr(), out.data_ptr(), n, 256)
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        o = out.view(-1, 4).cpu()
        clk = o[:, 0].double().mean().item()
        rt = o[:, 1].double().mean().item()  # 100 MHz
        waves = 256 * threads // 64
        flop = waves * chains * n * 32 * 32 * 64 * 2
        mhz = clk / (rt / 100.0)
        per_simd = clk / (n * chains * (threads // 64) / 4)
        print(f"mfma threads {threads} chains {chains}: {ms:.3f} ms {flop / ms / 1e9:.0f} TFLOP/s; "
              f"shader clock {mhz:.0f} MHz; {per_simd:.1f} clk an MFMA a SIMD", flush=True)


def bench_lds(dev):
    for threads, reads in ((256, 8), (512, 8), (256, 16)):
        out = torch.zeros(256 * (threads // 64) * 4, dtype=torch.int32, device=dev)
        n = 2000
        k = lds_kernel(threads, reads)
        k(out.data_ptr(), n, 256)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        k(out.data_ptr(), n, 256)
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        clk = out.view(-1, 4)[:, 0].double().mean().item()
        byts = threads * n * reads * 16
        print(f"lds threads {threads} reads {reads}: {ms:.3f} ms, {byts / clk:.1f} B/clk a CU, "
              f"{256 * byts / ms / 1e9:.1f} TB/s", flush=True)


def check_layout(dev):
    g = torch.Generator().manual_seed(0)
    A = torch.randint(-3, 4, (32, 64), generator=g).float()
    B = torch.randint(-3, 4, (64, 32), generator=g).float()
    D = A @ B
    cands = {
        "contig": lambda h, j: 32 * h + j,
        "split16": lambda h, j: (16 * h + j) if j < 16 else (32 + 16 * h + j - 16),
        "inter8": lambda h, j: 8 * (j // 8) * 2 + 8 * h + j % 8,
    }
    for name, kmap in cands.items():
        ab = torch.zeros(64, 32)
        bb = torch.zeros(64, 32)
        for l in range(64):
            for j in range(32):
                ab[l, j] = A[l % 32, kmap(l // 32, j)]
                bb[l, j] = B[kmap(l // 32, j), l % 32]
        a8 = ab.to(F8).view(torch.uint8).contiguous().to(dev)
        b8 = bb.to(F8).view(torch.uint8).contiguous().to(dev)
        out = torch.zeros(64, 16, device=dev)
        layout_kernel()(a8.data_ptr(), b8.data_ptr(), out.data_ptr())
        torch.cuda.synchronize()
        o = out.cpu()
        exp = torch.zeros(64, 16)
        for l in range(64):
            for i in range(16):
                exp[l, i] = D[8 * (i // 4) + 4 * (l // 32) + i % 4, l % 32]
        print(f"layout k map {name}: max |D - ref| {(o - exp).abs().max().item():.3g}", flush=True)
    # symmetric check: the same lane bytes on A and B give the dot product whatever the hardware's k order
    ab = torch.randint(-3, 4, (64, 32), generator=g).float()
    bb = torch.randint(-3, 4, (64, 32), generator=g).float()
    a8 = ab.to(F8).view(torch.uint8).contiguous().to(dev)
    b8 = bb.to(F8).view(torch.uint8).contiguous().to(dev)
    out = torch.zeros(64, 16, device=dev)
    layout_kernel()(a8.data_ptr(), b8.data_ptr(), out.data_ptr())
    torch.cuda.synchronize()
    o = out.cpu()
    exp = torch.zeros(64, 16)
    for l in range(64):
        n = l % 32
        for i in range(16):
            m = 8 * (i // 4) + 4 * (l // 32) + i % 4
            exp[l, i] = sum((ab[m + 32 * h] * bb[n + 32 * h]).sum() for h in range(2))
    print(f"layout same-bytes pairing: max |D - ref| {(o - exp).abs().max().item():.3g}", flush=True)


def check_layout16(dev):
    """16x16x128: D lane l element i = (row 4 (l / 16) + i, col l % 16), A and B paired by (lane group, byte)."""
    g = torch.Generator().manual_seed(1)
    ab = torch.randint(-3, 4, (64, 32), generator=g).float()
    bb = torch.randint(-3, 4, (64, 32), generator=g).float()
    a8 = ab.to(F8).view(torch.uint8).contiguous().to(dev)
    b8 = bb.to(F8).view(torch.uint8).contiguous().to(dev)
    out = torch.zeros(64, 4, device=dev)
    layout16_kernel()(a8.data_ptr(), b8.data_ptr(), out.data_ptr())
    torch.cuda.synchronize()
    o = out.cpu()
    exp = torch.zeros(64, 4)
    for l in range(64):
        n = l % 16
        for i in range(4):
            m = 4 * (l // 16) + i
            exp[l, i] = sum((ab[16 * gg + m] * bb[16 * gg + n]).sum() for gg in range(4))
    print(f"layout16 same-bytes pairing: max |D - ref| {(o - exp).abs().max().item():.3g}", flush=True)


def check_tr8(dev):
    src = torch.arange(4096, dtype=torch.int32) % 256
    src = src.to(torch.uint8).view(torch.int32).to(dev)  # byte b holds b % 256
    # lane l reads row l of 8 bytes, rows 8 bytes apart within a 16-lane group, groups 512 bytes apart
    for name, addr in (
        ("rows8", [(l // 16) * 512 + (l % 16) * 8 for l in range(64)]),
        ("rows64", [(l // 16) * 1024 + (l % 16) * 64 for l in range(64)]),
    ):
        a = torch.tensor(addr, dtype=torch.int32, device=dev)
        out = torch.zeros(128, dtype=torch.int32, device=dev)
        tr8_kernel()(src.data_ptr(), a.data_ptr(), out.data_ptr())
        torch.cuda.synchronize()
        by = out.cpu().view(torch.uint8).view(64, 8).int()
        print(f"tr8 {name}: lane -> 8 source bytes (byte address mod 256)")
        for l in range(0, 64, 1):
            print(f"  lane {l:2d} addr {addr[l]:5d}: {by[l].tolist()}")


def compile_all():
    """Every build once with placeholder arguments (FLYDSL_COMPILE_ONLY=1, no GPU)."""
    for threads, chains in ((256, 4), (256, 8), (512, 4), (512, 8)):
        mfma_kernel(threads, chains)(0, 0, 1, 1, stream=0)
    for threads, reads in ((256, 8), (512, 8), (256, 16)):
        lds_kernel(threads, reads)(0, 1, 1, stream=0)
    layout_kernel()(0, 0, 0, stream=0)
    tr8_kernel()(0, 0, 0, stream=0)
    layout16_kernel()(0, 0, 0, stream=0)
    print("COMPILE_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=["layout", "tr8", "mfma", "lds"])
    args = ap.parse_args()
    if args.what == ["compile"]:
        return compile_all()
    dev = torch.device("cuda")
    for w in args.what:
        {"mfma": bench_mfma, "lds": bench_lds, "layout": check_layout, "layout16": check_layout16, "tr8": check_tr8}[w](dev)


if __name__ == "__main__":
    main()
