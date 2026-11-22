# SPDX-License-Identifier: Apache-2.0
"""``mla_decode_c64`` with an in-launch writer, on one GPU: each request's new row (its rank-local last key)
holds an e4m3 NaN pattern in the pool when the launch starts; one CTA, after a delay, stores the real record at
device scope, waits for the stores, and raises the request's flag, which ``kv_ready`` polls (as the MLA front's
``stage_kv`` and KVRDY). The CTAs that read the row run on other XCDs and have read its cache lines'
neighbours (the previous key, the next request's first key) before the flag. The W_UV output and the LSE must
equal, bitwise, a launch whose pool held the real rows from the start."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import check_mla_decode as cd  # noqa: E402
import check_mla_tail as ct  # noqa: E402

F8, BF = torch.float8_e4m3fn, torch.bfloat16
NHA, KV, DQK, G, WO, DCP = 96, 512, 576, 6, 128, 8
WORDS = DQK // 4


def build_race(md, key):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import gpu, rocdl
    from flydsl.expr.typing import Int32, Int64, T

    from sgl_amd_mono.common.ops import CM_DEV, rsrc

    keyed = md.key_tuple(key, md.SOURCES)

    @flyc.jit
    def writer(kv, kv_idx, kv_ptr, fresh, flag, nreq, delay, tid):
        t0 = md._realtime()
        now = md._realtime()
        while now - t0 < delay:
            rocdl.s_sleep(8)
            now = md._realtime()
        for r in range(nreq):
            if tid < WORDS:
                row = md._ld(kv_idx, md._ld(kv_ptr, r + 1) - 1)
                w = fx.Int32(bo.buffer_load(rsrc(fresh), r * WORDS + tid, vec_width=1, dtype=T.i32))
                bo.buffer_store(w, rsrc(kv), row * WORDS + tid, cache_modifier=CM_DEV)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid < nreq:
            bo.buffer_store(fx.Int32(1), rsrc(flag), tid, cache_modifier=CM_DEV)

    @flyc.jit
    def poll_flag(flag, rq, tid):
        if tid % 64 == 0:
            v = fx.Int32(bo.buffer_load(rsrc(flag), rq, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
            while v == 0:
                rocdl.s_sleep(1)
                v = fx.Int32(bo.buffer_load(rsrc(flag), rq, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))

    @flyc.kernel(name="k3_mla_decode_c64_race", known_block_size=[md.THREADS, 1, 1])
    def race(
        q8: Int64, kv: Int64, kv_idx: Int64, kv_ptr: Int64, g_ptr: Int64, rank: Int32,
        o: Int64, lse: Int64, part: Int64, plse: Int64, wq: Int64, cnt: Int64,
        wuv: Int64, half: Int64, wout: Int64, fresh: Int64, flag: Int64, nreq: Int32, delay: Int64, wcta: Int32,
    ):  # fmt: skip
        _ = keyed
        bid, tid = fx.block_idx.x, fx.thread_idx.x
        if bid == wcta:
            writer(kv, kv_idx, kv_ptr, fresh, flag, nreq, delay, tid)
        a = dict(kv=kv, kv_idx=kv_idx, kv_ptr=kv_ptr, g_ptr=g_ptr, rank=rank, o=o, lse=lse, part=part, plse=plse, wq=wq,
                 cnt=cnt, layer=fx.Int32(0), wuv=wuv, half=half, tl=fx.Int64(0))  # fmt: skip
        q_load, ready, _, emit_values, emit_lse = md.default_hooks(q8, wout, lse)

        def kv_ready(rq):
            poll_flag(flag, rq, tid)

        md.decode_stage(key, fx.SharedAllocator().allocate(md.Smem).peek(), bid, tid, a, q_load, ready, kv_ready,
                        emit_values, emit_lse)  # fmt: skip

    @flyc.jit
    def launch(
        q8: Int64, kv: Int64, kv_idx: Int64, kv_ptr: Int64, g_ptr: Int64, rank: Int32,
        o: Int64, lse: Int64, part: Int64, plse: Int64, wq: Int64, cnt: Int64,
        wuv: Int64, half: Int64, wout: Int64, fresh: Int64, flag: Int64, nreq: Int32, delay: Int64, wcta: Int32,
        stream: fx.Stream = md._STREAM,
    ):  # fmt: skip
        _ = keyed
        race(q8, kv, kv_idx, kv_ptr, g_ptr, rank, o, lse, part, plse, wq, cnt, wuv, half, wout, fresh, flag, nreq, delay,
             wcta).launch(grid=(md.BLOCKS,), block=(md.THREADS,), stream=stream)  # fmt: skip

    md._agpr_hint_lowering()
    return flyc.compile[{"agpr_alloc": 0, "llvm_options": {"slp-threshold": 100000}}](launch)


def case(name, ps, r, args, dev, bufs, wuv, md, race):
    reqs = [ct.Req(p, args.seed * 1000 + 7 * i, dev) for i, p in enumerate(ps)]
    assert all((r - p) % DCP < G for p in ps), f"rank {r} owns no new row of some request in {ps}"
    s = G * len(ps)
    g = torch.Generator(device=dev).manual_seed(args.seed + 5)
    q8 = (torch.randn(s, NHA, DQK, generator=g, device=dev) * args.qstd).clamp(-448, 448).to(F8).contiguous()
    b = ct.RankBatch(reqs, r, dev)
    key = md.MlaDecodeBuild(scale=cd.SCALE)
    o = torch.zeros(s, NHA, KV, dtype=BF, device=dev)
    lse = torch.zeros(s, NHA, dtype=torch.float32, device=dev)
    wout = torch.zeros(s * NHA * 2 * WO, dtype=torch.float32, device=dev)

    def plan():
        md.mla_decode_plan_launch(kv_ptr=b.kv_ptr, g_ptr=b.g_kv_ptr, nreq=len(ps), bufs=bufs)

    plan()
    md.mla_decode_launch(key, q8=q8, kv=b.pool, kv_idx=b.kv_idx, kv_ptr=b.kv_ptr, g_ptr=b.g_kv_ptr, rank=b.r, o=o,
                         lse=lse, bufs=bufs, wuv=wuv, wout=wout)  # fmt: skip
    torch.cuda.synchronize()
    ref_w, ref_l = wout.clone(), lse.clone()
    rows = b.kv_idx[(b.kv_ptr[1:] - 1).long()].long()
    fresh = b.pool[rows].contiguous()
    pool = b.pool.clone()
    flag = torch.zeros(64, dtype=torch.int32, device=dev)
    line = f"{name:10s} rank {r} rows {rows.tolist()}:"
    for d in args.delays:
        bad = nan = 0
        for _ in range(args.trials):
            pool.copy_(b.pool)
            pool.view(torch.uint8)[rows] = 0x7F  # e4m3 NaN
            flag.zero_()
            wout.zero_()
            lse.zero_()
            o.zero_()
            plan()
            race(q8.data_ptr(), pool.data_ptr(), b.kv_idx.data_ptr(), b.kv_ptr.data_ptr(), b.g_kv_ptr.data_ptr(), b.r,
                 o.data_ptr(), lse.data_ptr(), bufs.part.data_ptr(), bufs.plse.data_ptr(), bufs.wq.data_ptr(),
                 bufs.cnt.data_ptr(), wuv.data_ptr(), bufs.half.data_ptr(), wout.data_ptr(), fresh.data_ptr(),
                 flag.data_ptr(), len(ps), d, md.BLOCKS - 1, stream=torch.cuda.current_stream())  # fmt: skip
            torch.cuda.synchronize()
            same = torch.equal(wout, ref_w) and torch.equal(lse, ref_l)
            bad += int(not same)
            nan += int(bool(torch.isnan(wout).any() or torch.isnan(lse).any()))
        line += f" | delay {d}: {bad}/{args.trials} differ, {nan} with NaN"
    print(line, flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="a JSON of {name: {lens: [...]}}")
    ap.add_argument("--uniform", type=int, nargs="*", default=[1, 2, 3, 4, 5], help="requests at --ctx each")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ranks", type=int, nargs="+", default=[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qstd", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--delays", type=int, nargs="+", default=[0, 2000], help="the writer's delay (100 MHz ticks)")
    args = ap.parse_args()
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode_c64 as md

    dev = torch.device("cuda")
    bufs = md.DecodeBuffers(dev)
    gw = torch.Generator(device=dev).manual_seed(11)
    wuv = (torch.randn(NHA * KV * WO, generator=gw, device=dev) * 0.03).to(BF)
    race = build_race(md, md.MlaDecodeBuild(scale=cd.SCALE))
    todo = []
    if args.cases:
        todo += [(k, v["lens"]) for k, v in json.load(open(args.cases)).items()]
    todo += [(f"{args.ctx // 1024}Kx{n}", [args.ctx] * n) for n in args.uniform]
    tot = 0
    for name, ps in todo:
        for r in args.ranks:
            tot += case(name, ps, r, args, dev, bufs, wuv, md, race)
    print(f"RACE_DONE differing launches {tot}", flush=True)


if __name__ == "__main__":
    main()
