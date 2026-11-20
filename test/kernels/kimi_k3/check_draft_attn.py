# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 drafter attention (``kimi_k3.gfx950.attention.draft``) against
SGLang's grouped-head verify kernel (``verify_shared_kv_fwd``, the stock draft
path on gfx950) and an fp32 reference, on one GPU.

    python3 test/kernels/kimi_k3/check_draft_attn.py --lens 250000,10000,10000,10000

Inputs have the served shapes and layouts: a rank's 8 query and 2 KV heads of 64
as views of the fused qkv rows, ``--rows`` block rows a request, int64 slot ids
from ``--slot-base`` (each request's slots contiguous; ``--scattered``: a random
permutation of the pool), the draft KV in FP8 e4m3 at ``--k-scale`` /
``--v-scale``. Logits have a few planted peaks a request, so the softmax is not
flat. Prints rel-L2 and max-abs of each kernel against the reference and of the
new kernel against the stock one; a check passes when the new kernel is finite,
repeats bitwise and is no further from the reference than the stock kernel.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from sgl_amd_mono.kimi_k3.gfx950.attention import draft  # noqa: E402

FP8 = torch.float8_e4m3fn


def make_inputs(
    lens,
    rows=5,
    seed=0,
    scattered=False,
    k_scale=1.0,
    v_scale=1.0,
    slot_base=1024,
    device="cuda",
):
    g = torch.Generator(device=device).manual_seed(seed)
    bs, total = len(lens), sum(lens)
    pool = total + slot_base
    if scattered:
        slots = torch.randperm(pool, generator=g, device=device)[:total]
    else:
        slots = torch.arange(slot_base, slot_base + total, device=device)
    kv_indices = slots.to(torch.int64)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(torch.tensor(lens, device=device), 0)
    qo_indptr = torch.arange(bs + 1, dtype=torch.int64, device=device) * rows

    def rn(*shape):
        return torch.randn(*shape, generator=g, device=device)

    # q, k and v are views of the fused qkv rows (8 + 2 + 2 heads of 64), as served
    qkv = rn(bs * rows, draft.H_Q + 2 * draft.H_KV, draft.HEAD_DIM).to(torch.bfloat16)
    q = (2.0 * qkv[:, : draft.H_Q]).to(torch.bfloat16)
    qkv[:, : draft.H_Q] = q
    q = qkv[:, : draft.H_Q]
    k = rn(pool, draft.H_KV, draft.HEAD_DIM)
    v = rn(pool, draft.H_KV, draft.HEAD_DIM)
    # planted peaks: 16 context keys a request along its mean query direction
    qm = q.float().view(bs, rows, draft.H_KV, draft.GROUP, draft.HEAD_DIM).mean((1, 3))
    for b in range(bs):
        lo = int(kv_indptr[b])
        pos = torch.randint(0, lens[b], (16,), generator=g, device=device)
        k[slots[lo + pos].long()] = (
            1.5 * qm[b] / qm[b].norm(dim=-1, keepdim=True) * draft.HEAD_DIM**0.5
        )
    k_buf = (k / k_scale).to(FP8)
    v_buf = (v / v_scale).to(FP8)
    k_ext = qkv[:, draft.H_Q : draft.H_Q + draft.H_KV]
    v_ext = qkv[:, draft.H_Q + draft.H_KV :]
    return dict(
        q=q,
        k_ext=k_ext,
        v_ext=v_ext,
        k_buf=k_buf,
        v_buf=v_buf,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        sm_scale=draft.HEAD_DIM**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
        rows=rows,
        lens=list(lens),
    )


def reference(x):
    """fp32: every block row over the dequantized context and the whole block."""
    rows, out = x["rows"], []
    for b, n in enumerate(x["lens"]):
        lo = int(x["kv_indptr"][b])
        idx = x["kv_indices"][lo : lo + n].long()
        kc = torch.cat(
            [
                x["k_buf"][idx].float() * x["k_scale"],
                x["k_ext"][b * rows : (b + 1) * rows].float(),
            ]
        )
        vc = torch.cat(
            [
                x["v_buf"][idx].float() * x["v_scale"],
                x["v_ext"][b * rows : (b + 1) * rows].float(),
            ]
        )
        q = x["q"][b * rows : (b + 1) * rows].float()  # [rows, H_Q, D]
        kh = kc.repeat_interleave(draft.GROUP, dim=1)  # [n + rows, H_Q, D]
        vh = vc.repeat_interleave(draft.GROUP, dim=1)
        s = torch.einsum("lhd,thd->hlt", q, kh) * x["sm_scale"]
        out.append(torch.einsum("hlt,thd->lhd", torch.softmax(s, -1), vh))
    return torch.cat(out)


def run_stock(x, o):
    from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd

    ran = verify_shared_kv_fwd(
        x["q"],
        x["k_ext"],
        x["v_ext"],
        o,
        x["k_buf"],
        x["v_buf"],
        x["qo_indptr"],
        x["kv_indptr"],
        x["kv_indices"],
        None,
        False,
        None,
        x["rows"],
        x["k_scale"],
        x["v_scale"],
        x["sm_scale"],
        max_bs=len(x["lens"]),
    )
    assert ran, "verify_shared_kv_fwd declined the draft shape"
    return o


def new_runner(x):
    impl = draft.DraftAttention(len(x["lens"]), x["rows"], x["q"].device)

    def run(o):
        return impl(
            x["q"],
            x["k_ext"],
            x["v_ext"],
            o,
            x["k_buf"],
            x["v_buf"],
            x["qo_indptr"],
            x["kv_indptr"],
            x["kv_indices"],
            x["sm_scale"],
            x["k_scale"],
            x["v_scale"],
        )

    return run


def err(a, b):
    a, b = a.float(), b.float()
    return {
        "rel_l2": ((a - b).norm() / b.norm()).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def check(
    lens, rows=5, seed=0, scattered=False, k_scale=1.0, v_scale=1.0, slot_base=1024
):
    x = make_inputs(lens, rows, seed, scattered, k_scale, v_scale, slot_base)
    ref = reference(x)
    shape = (len(lens) * rows, draft.H_Q, draft.HEAD_DIM)
    o_stock = run_stock(x, torch.empty(shape, dtype=torch.bfloat16, device="cuda"))
    o_new = new_runner(x)(torch.empty(shape, dtype=torch.bfloat16, device="cuda"))
    o_new2 = new_runner(x)(torch.empty(shape, dtype=torch.bfloat16, device="cuda"))
    torch.cuda.synchronize()
    res = {
        "lens": lens
        if len(lens) <= 8
        else f"{len(lens)} reqs, sum {sum(lens)}, max {max(lens)}",
        "scattered": scattered,
        "slot_base": slot_base,
        "stock_vs_ref": err(o_stock, ref),
        "new_vs_ref": err(o_new, ref),
        "new_vs_stock": err(o_new, o_stock),
        "new_repeat_bitwise": torch.equal(o_new, o_new2),
        "finite": bool(torch.isfinite(o_new.float()).all()),
    }
    res["ok"] = (
        res["finite"]
        and res["new_repeat_bitwise"]
        and res["new_vs_ref"]["rel_l2"] <= res["stock_vs_ref"]["rel_l2"]
        and res["new_vs_ref"]["max_abs"] <= res["stock_vs_ref"]["max_abs"]
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lens",
        action="append",
        required=True,
        help="comma-separated contexts (repeatable)",
    )
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scattered", action="store_true")
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--v-scale", type=float, default=1.0)
    ap.add_argument(
        "--slot-base",
        type=int,
        default=1024,
        help="first slot id (above 2**24 the K / V offsets pass 2**31)",
    )
    a = ap.parse_args()
    bad = 0
    for spec in a.lens:
        lens = [int(t) for t in spec.split(",")]
        r = check(lens, a.rows, a.seed, a.scattered, a.k_scale, a.v_scale, a.slot_base)
        bad += not r["ok"]
        print("CHECK", r, flush=True)
    print("CHECK_DONE", "FAIL" if bad else "PASS", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
