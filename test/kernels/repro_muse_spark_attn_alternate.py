"""Alternate dumped served attention cases (check mode's SGLANG_AMD_MONO_ATTN_DUMP)
through one K-attn scratch: a launch whose scratch still holds another case's
partials shows whether a merge can read a stale partial record. Each case keeps
its own cache, pages and q. ``python repro_muse_spark_attn_alternate.py iters a.pt b.pt ...``"""

import struct
import sys

import torch


def _case(path, dev, MAXB):
    d = torch.load(path, weights_only=False)
    rows, ql, extra = d["rows"], d["q_len"], d["extra"]
    lens = d["lens"].to(dev).to(torch.int32)
    keys = int(lens[0]) + extra
    n_pages = (keys + 15) // 16
    pages = d["pages"].to(dev).to(torch.int32)
    k_rows, v_rows = d["k"].to(dev), d["v"].to(dev)
    k_cache = torch.zeros(
        int(pages.max()) * 16 + 16, *k_rows.shape[1:], dtype=k_rows.dtype, device=dev
    )
    v_cache = torch.zeros_like(k_cache)
    slots = (
        pages.long()[:, None] * 16 + torch.arange(16, device=dev)[None, :]
    ).reshape(-1)[:keys]
    k_cache[slots] = k_rows
    v_cache[slots] = v_rows
    table = torch.zeros(1, MAXB, dtype=torch.int32, device=dev)
    table[0, :n_pages] = pages
    return {
        "name": path.rsplit("/", 1)[-1], "rows": rows, "ql": ql, "extra": extra, "lens": lens, "k": k_cache,
        "v": v_cache, "table": table, "q": d["q"].to(dev).to(torch.bfloat16), "ref": d["ref"].to(dev).float(),
        "layer": d["layer"], "scale": struct.unpack("i", struct.pack("f", d["scale"]))[0],
    }  # fmt: skip


def main():
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx950.layer import KAttnBuild, build_kattn
    from sgl_amd_mono.muse_spark.gfx950.layout import (
        kattn_scratch_layout,
        scratch_bytes,
    )

    iters = int(sys.argv[1])
    dev = torch.device("cuda")
    MAXB = 65536
    cases = [_case(p, dev, MAXB) for p in sys.argv[2:]]
    rows, ql = cases[0]["rows"], cases[0]["ql"]
    assert all(c["rows"] == rows and c["ql"] == ql for c in cases)
    scratch = torch.zeros(
        scratch_bytes(kattn_scratch_layout(rows, ql)), dtype=torch.uint8, device=dev
    )
    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(30.0)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    launch = build_kattn(KAttnBuild(tokens=rows, q_len=ql))
    o = torch.full(
        (rows, cases[0]["q"].shape[1]), float("nan"), device=dev, dtype=torch.bfloat16
    )
    bad = 0
    for it in range(iters):
        c = cases[it % len(cases)]
        o.fill_(float("nan"))
        launch(
            c["q"].data_ptr(), o.data_ptr(), c["k"].data_ptr(), c["v"].data_ptr(), c["table"].data_ptr(), MAXB,
            c["table"].numel() * 4, c["lens"].data_ptr(), 0, c["extra"], 0, 0, c["scale"], scratch.data_ptr(),
            e.data_ptr(), c["layer"], stream=torch.cuda.current_stream(),
        )  # fmt: skip
        e[0] += 1
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            o.float().reshape(rows, -1, 64), c["ref"].reshape(rows, -1, 64), dim=-1
        )
        if cos.min().item() < 0.999 or o.isnan().any():
            bad += 1
            if bad <= 8:
                print(
                    f"iter {it} {c['name']}: min {cos.min().item():.5f} low {(cos < 0.999).nonzero().tolist()[:10]}",
                    flush=True,
                )
    errs = e[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
    print(f"{iters} launches over {len(cases)} cases: {bad} bad; guard errors {errs}")


if __name__ == "__main__":
    main()
