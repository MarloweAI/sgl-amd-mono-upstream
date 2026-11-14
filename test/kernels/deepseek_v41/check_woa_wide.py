"""The wide-row wo_a (gfx950/stages/woa_wide) against SGLang's gfx950 route (the Triton batched GEMM
with wo_b's MXFP8 quant in its epilogue, one walk over K), one GPU: the e4m3 codes and ue8m0 scales must
match bit for bit at every row count and group count; synthetic checkpoint-like weights (e4m3 with
power-of-two 32 x 32 block scales), or a real layer's with --weights.

    python3 check_woa_wide.py [--weights MODEL_DIR --layer 5] [--rows 65,96,97,150,192,250,366,384]

Up to 64 rows SGLang's default route splits K eight ways (its deterministic-inference route does not);
those rows are compared with the single walk, and the split route's difference is printed alongside.
"""

import argparse
import json
import os

import torch

from sgl_amd_mono.deepseek_v41.gfx950.stages import woa_wide

R, D = 1024, 4096


def synthetic(g, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn(g * R, D, device="cuda", generator=gen) * 64).clamp(-448, 448)
    w8 = w8.to(torch.float8_e4m3fn)
    codes = torch.randint(
        112, 124, (g * R // 32, D // 32), device="cuda", generator=gen
    )
    return w8, codes.to(torch.uint8)


def real(model, layer, g):
    from safetensors import safe_open

    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))[
        "weight_map"
    ]
    out = []
    for name in ("weight", "scale"):
        key = f"layers.{layer}.attn.wo_a.{name}"
        with safe_open(
            os.path.join(model, index[key]), framework="pt", device="cuda"
        ) as f:
            out.append(f.get_tensor(key))
    w8 = out[0][: g * R]
    bits = out[1][: g * R // 32].float().contiguous().view(torch.int32)
    return w8.contiguous(), (bits >> 23).to(torch.uint8)


def dequant(w8, codes):
    n, k = w8.shape
    s = torch.exp2(codes.float() - 127)[:, None, :, None]
    return (w8.float().view(n // 32, 32, k // 32, 32) * s).view(n, k).to(torch.bfloat16)


def main():
    from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import (
        batched_gemm_bf16_fp8_grid,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--rows", default="49,54,64,65,96,97,150,192,250,366,384")
    a = ap.parse_args()
    bad = 0
    for g in (4, 2):
        w8, codes = real(a.weights, a.layer, g) if a.weights else synthetic(g, 1 + g)
        wb = dequant(w8, codes).view(g, R, D)
        wp = woa_wide.pack_weight(w8)
        for m in map(int, a.rows.split(",")):
            x = torch.randn(m, g, D, device="cuda", dtype=torch.bfloat16)
            q = torch.empty(m, g * R, device="cuda", dtype=torch.float8_e4m3fn)
            s = torch.empty(m, g * R // 32, device="cuda", dtype=torch.uint8)
            tl = woa_wide.TILES[(g, woa_wide.width_class(m))]
            pw, fw = woa_wide.workspace(m, g, tl)
            part = torch.empty(pw, device="cuda", dtype=torch.float32)
            flags = torch.zeros(fw, device="cuda", dtype=torch.int32)
            woa_wide.standalone(m, g)(
                x.data_ptr(), wp.data_ptr(), codes.data_ptr(), q.data_ptr(), s.data_ptr(),
                part.data_ptr(), flags.data_ptr(), stream=torch.cuda.current_stream(),
            )  # fmt: skip
            rq, rs = batched_gemm_bf16_fp8_grid(
                x, wb, fp8_grid=False, split_k=False, emit_fp8=True
            )
            ok = torch.equal(q.view(torch.uint8), rq.view(torch.uint8)) and torch.equal(
                s, rs
            )
            bad += not ok
            line = f"g={g} m={m:3d}: {'bit for bit' if ok else 'DIFFERS'}"
            if m <= 64:
                dq, ds = batched_gemm_bf16_fp8_grid(
                    x, wb, fp8_grid=False, emit_fp8=True
                )
                frac = (
                    (dq.view(torch.uint8) != q.view(torch.uint8)).float().mean().item()
                )
                line += f"; the default split-K route differs at {frac:.2e} of codes"
            print(line, flush=True)
    print("PASS" if bad == 0 else f"FAIL ({bad} cases)")


if __name__ == "__main__":
    main()
