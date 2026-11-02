"""Replay a served attention case dumped by check mode (SGLANG_AMD_MONO_ATTN_DUMP): the
request's K / V rows put back in fresh pages of a cache, the dumped q, the standalone
K-attn launch, its output against the dump's exact attention and the served output.
``python repro_muse_spark_attn_dump.py <dump.pt> [iters]``"""

import struct
import sys

import torch


def main():
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx950.layer import KAttnBuild, build_kattn
    from sgl_amd_mono.muse_spark.gfx950.layout import (
        kattn_scratch_layout,
        scratch_bytes,
    )

    d = torch.load(sys.argv[1], weights_only=False)
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    dev = torch.device("cuda")
    rows, ql, extra = d["rows"], d["q_len"], d["extra"]
    lens = d["lens"].to(dev).to(torch.int32)
    keys = int(lens[0]) + extra
    n_pages = (keys + 15) // 16
    k_rows, v_rows = d["k"].to(dev), d["v"].to(dev)  # [keys, KV heads, 64] fp8
    PAGES = n_pages + 64
    MAXB = 65536
    k_cache = torch.zeros(PAGES * 16, *k_rows.shape[1:], dtype=k_rows.dtype, device=dev)
    v_cache = torch.zeros_like(k_cache)
    which = (
        sys.argv[3]
        if len(sys.argv) > 3 and not sys.argv[3].startswith("--")
        else "fresh"
    )
    if which == "served":
        pages = (
            d["pages"].to(dev).to(torch.int32)
        )  # the served page ids (the cache sized to hold them)
        k_cache = torch.zeros(
            int(pages.max()) * 16 + 16,
            *k_rows.shape[1:],
            dtype=k_rows.dtype,
            device=dev,
        )
        v_cache = torch.zeros_like(k_cache)
    else:
        pages = torch.arange(1, n_pages + 1, dtype=torch.int32, device=dev)
    slots = (
        pages.long()[:, None] * 16 + torch.arange(16, device=dev)[None, :]
    ).reshape(-1)[:keys]
    k_cache[slots] = k_rows
    v_cache[slots] = v_rows
    table = torch.zeros(1, MAXB, dtype=torch.int32, device=dev)
    table[0, :n_pages] = pages
    q = d["q"].to(dev).to(torch.bfloat16)
    ref = d["ref"].to(dev).float()
    served = d["o"].to(dev).float()
    o = torch.full_like(q, float("nan"))
    scratch = torch.zeros(
        scratch_bytes(kattn_scratch_layout(rows, ql)), dtype=torch.uint8, device=dev
    )
    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(30.0)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    launch = build_kattn(KAttnBuild(tokens=rows, q_len=ql))
    scale_bits = struct.unpack("i", struct.pack("f", d["scale"]))[0]
    cold = "--cold" in sys.argv
    flush = torch.empty(2 * 2**30, dtype=torch.uint8, device=dev) if cold else None
    for it in range(iters):
        if cold:
            flush.fill_(it & 0xFF)  # evict the caches: the K / V reads come from HBM
            torch.cuda.synchronize()
        o.fill_(float("nan"))
        launch(
            q.data_ptr(), o.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), table.data_ptr(), MAXB, table.numel() * 4,
            lens.data_ptr(), 0, extra, 0, 0, scale_bits, scratch.data_ptr(), e.data_ptr(), d["layer"],
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        e[0] += 1
        torch.cuda.synchronize()
        got = o.float()
        c = torch.nn.functional.cosine_similarity(
            got.reshape(rows, -1, 64), ref.reshape(rows, -1, 64), dim=-1
        )
        cs = torch.nn.functional.cosine_similarity(
            served.reshape(rows, -1, 64), ref.reshape(rows, -1, 64), dim=-1
        )
        cg = torch.nn.functional.cosine_similarity(
            got.reshape(rows, -1, 64), served.reshape(rows, -1, 64), dim=-1
        )
        print(
            f"iter {it} ({which} pages): replay vs exact min {c.min().item():.5f}; served vs exact min {cs.min().item():.5f}; replay vs served min {cg.min().item():.5f}; low {(c < 0.99).nonzero().tolist()[:12]}"
        )


if __name__ == "__main__":
    main()
