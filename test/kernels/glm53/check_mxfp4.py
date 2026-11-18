"""The MXFP4 MLPs of the GLM-5.3 layer launch (an MXFP4 checkpoint's) against
the AITER ops SGLang runs for them, under torchrun (TP 4 or 8):

    torchrun --nproc-per-node 8 test/kernels/glm53/check_mxfp4.py --kind shared

At each of ``--rows`` and ``--rowsets`` (``check_layer.py``'s), one launch of
``--kind`` on random FP8 attention and a random MXFP4 MLP as SGLang holds it: the
experts in AITER's layouts (``shuffle_weight`` (16, 16), ``e8m0_shuffle``), or a
dense MLP's row-major linears, which the launch reads as expert slices. On the
launch's own FFN input (``xqd``) and routing (``sel``, ``prob``), every rank's part
summed: SGLang's op (``fused_moe`` per_1x32, or the Triton ``dynamic_mxfp4_quant``
and ``gemm_afp4wfp4`` linears) and an f32 MLP on the same MXFP4 quantization
(``quant``, bitwise AITER's: ``mxfp4_moe_quant``'s, ``dynamic_mxfp4_quant``'s). The
launch's MLP output is its output minus its residual ``a``; ``a`` plus the op's,
rounded to bf16, is the floor a bf16 output shows. Exits 1 past ``--rel`` (the op is
~4e-3 from the f32 MLP), or when the launch's MXFP4 FFN input or mid differs from
``quant`` of its own f32 values."""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_layer as cl  # noqa: E402

from sgl_amd_mono.glm53.check import (  # noqa: E402
    E2M1,
    _pairs_f32,
    _pairs_raw_f32,
    _rel,
    ffn_input,
    mx_f32,
)
from sgl_amd_mono.glm53.gfx950.config import HIDDEN, MOE_SLOTS  # noqa: E402


def quant(x: torch.Tensor, even: bool = False) -> torch.Tensor:
    """AITER's MXFP4 quant of f32 ``x`` [..., k], dequantized: a 32's E8M0 from its
    amax (the MoE's: amax / 6 rounded up to a power of two, at most 254; ``even``, the
    Triton linears': amax's exponent once 2^21 is added to its bits, over 4), then
    E2M1 round-to-nearest-even, saturating."""
    x = x.float()
    shp = x.shape
    b = x.reshape(*shp[:-1], -1, 32)
    amax = b.abs().amax(-1, keepdim=True)
    if even:
        bits = (amax.view(torch.int32) + 0x200000) & 0x7F800000
        code = ((bits >> 23) - 2).clamp(0, 254)
    else:
        bits = (amax * torch.tensor(1.0 / 6.0)).view(torch.int32)
        code = ((bits + 0x7FFFFF) >> 23 & 0xFF).clamp(max=254)
    scale = torch.exp2(code.float() - 127)
    q = (b / scale).clamp(-6.0, 6.0)
    table = torch.tensor(E2M1, device=x.device)
    d = (q.abs().unsqueeze(-1) - table).abs()
    best = d.min(-1, keepdim=True).values
    # ties go to the even code (the E2M1 index with a zero mantissa bit)
    tie_even = (d == best) & (torch.arange(8, device=x.device) % 2 == 0)
    pick = torch.where(tie_even.any(-1), tie_even.float().argmax(-1), d.argmin(-1))
    return (table[pick] * q.sign() * scale).reshape(shp)


def f32_moe(x, ids, weights, raw) -> torch.Tensor:
    """This rank's part of the MoE in f32 on ``quant``'s activations and mid."""
    w_ug, s_ug, w_dn, s_dn = raw
    inter = w_ug.shape[1] // 2
    xq = quant(x)
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for s in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[s, j])
            gu = mx_f32(w_ug[e], s_ug[e]) @ xq[s]
            mid = torch.nn.functional.silu(gu[:inter]) * gu[inter:]
            out[s] += weights[s, j] * (mx_f32(w_dn[e], s_dn[e]) @ quant(mid))
    return out


def f32_dense(x, raw) -> torch.Tensor:
    """This rank's part of the dense MLP in f32 on ``quant(even=True)``'s activations,
    gate | up and the mid rounded to bf16 as SGLang's linears and SiluAndMul leave them."""
    gu, s_gu, dn, s_dn = raw
    inter = gu.shape[0] // 2
    y = (quant(x, even=True) @ mx_f32(gu, s_gu).T).bfloat16().float()
    mid = torch.nn.functional.silu(y[:, :inter]) * y[:, inter:]
    return quant(mid.bfloat16(), even=True) @ mx_f32(dn, s_dn).T


def aiter_dense(x, raw) -> torch.Tensor:
    """This rank's part of the dense MLP as SGLang's MXFP4 linears run it (no
    output buffer from a DeepseekV2MLP): AITER's Triton ``dynamic_mxfp4_quant`` of the
    bf16 input, then ``gemm_afp4wfp4`` to bf16; SiluAndMul in bf16."""
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    def linear(a, w, s):
        a_q, a_s = dynamic_mxfp4_quant(a.bfloat16())
        y = torch.empty(a.shape[0], w.shape[0], dtype=torch.bfloat16, device=a.device)
        gemm_afp4wfp4(a_q, w, a_s, s, torch.bfloat16, y)
        return y.float()

    gu, s_gu, dn, s_dn = raw
    inter = gu.shape[0] // 2
    y = linear(x, gu, s_gu)
    mid = torch.nn.functional.silu(y[:, :inter]) * y[:, inter:]
    return linear(mid, dn, s_dn)


def aiter_moe(x, ids, weights, w) -> torch.Tensor:
    """This rank's part of the MoE as SGLang's MXFP4 MoE calls AITER."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    w1 = w.w_ug.view(fp4) if fp4 is not None else w.w_ug
    w2 = w.w_dn.view(fp4) if fp4 is not None else w.w_dn
    w1.is_shuffled = w2.is_shuffled = True  # as SGLang marks its preshuffled weights
    return fused_moe(
        x.bfloat16(),
        w1,
        w2,
        weights.float(),
        ids.int(),
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w.s_ug,
        w2_scale=w.s_dn,
        gate_mode=GateMode.SEPARATED.value,
    ).float()


def quant_exact(rt, x: torch.Tensor, S: int, dense: bool) -> bool:
    """The launch's MXFP4 FFN input (``xq``, ``xqs``) and mid (``midq``, ``midqs``)
    equal ``quant`` of its own f32 input (``xqd``) and mid, value for value."""

    def words(name: str, n: int) -> torch.Tensor:
        return _pairs_f32(rt, name, n).view(torch.int32).contiguous()

    inter = rt.key.inter
    n_mid = S * MOE_SLOTS * inter
    xq = words("xq", S * HIDDEN // 8).view(torch.uint8).view(S, HIDDEN // 2)
    xqs = words("xqs", S * HIDDEN // 32).view(S, -1)
    mid = _pairs_f32(rt, "mid", n_mid).view(1, -1)
    midq = words("midq", n_mid // 8).view(torch.uint8).view(1, -1)
    midqs = words("midqs", n_mid // 32).view(1, -1)
    return torch.equal(mx_f32(xq, xqs), quant(x, even=dense)) and torch.equal(
        mx_f32(midq, midqs), quant(mid, even=dense)
    )


def check(args, S: int, rowset: str, rank: int, tp: int, group, device) -> bool:
    """One launch at width S against the op and the f32 MLP; rank 0 prints them."""
    positions, requests = cl.rows_of(S, rowset)
    model = cl.Model(
        [args.kind], positions, requests, rank, tp, group, device, args, stop=True
    )
    out = model.step()
    torch.cuda.synchronize()
    rt = model.runtimes[args.kind]
    a = ffn_input(rt, S).float()
    x = _pairs_raw_f32(rt, "xqd", S * HIDDEN).view(S, HIDDEN).clone()
    ids = _pairs_f32(rt, "sel", S * MOE_SLOTS).view(torch.int32).view(S, -1).clone()
    weights = _pairs_f32(rt, "prob", S * MOE_SLOTS).view(S, MOE_SLOTS).clone()
    raw = model.weights[0].keep[-4:]
    dense = args.kind == "dense"
    exact = quant_exact(rt, x, S, dense)
    if args.kind == "dense":
        ref, f32 = aiter_dense(x, raw), f32_dense(x, raw)
    else:
        ref = aiter_moe(x, ids, weights, model.weights[0])
        f32 = f32_moe(x, ids, weights, raw)
    parts = torch.stack([ref, f32]).cpu()
    dist.all_reduce(parts, group=group)
    ref, f32 = parts[0], parts[1]
    ok = model.errors() == 0 and exact
    if rank == 0:
        mine = out.float().cpu() - a.cpu()
        floor = (a.cpu() + ref).bfloat16().float() - a.cpu()
        rel = _rel(mine, ref)
        used = "-" if args.kind == "dense" else ids.unique().numel()
        print(
            f"MXFP4 kind={args.kind} rows={S}{rowset}: launch vs op rel {rel:.3e}, vs f32 "
            f"{_rel(mine, f32):.3e}; op vs f32 {_rel(ref, f32):.3e}; bf16 floor "
            f"{_rel(floor, ref):.3e}; max |launch - op| "
            f"{(mine - ref).abs().max().item():.3e} (|op| max "
            f"{ref.abs().max().item():.3e}); experts used {used}; quant exact "
            f"{exact}; clean={model.errors() == 0}",
            flush=True,
        )
        ok = ok and rel <= args.rel
    model.close()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument("--rowsets", default="p0,v4", help="comma-separated: p0, p1, v4")
    ap.add_argument("--kind", choices=("shared", "index", "dense"), default="shared")
    ap.add_argument("--kv", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--deadline", type=float, default=60.0)
    ap.add_argument("--rel", type=float, default=0.01, help="largest rel to the op")
    args = ap.parse_args()
    args.mxfp4, args.similar = True, 0.0
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    ok = all(
        [
            check(args, S, rowset, rank, tp, group, device)
            for S in map(int, args.rows.split(","))
            for rowset in args.rowsets.split(",")
        ]
    )
    verdict = [ok]
    dist.broadcast_object_list(verdict, src=0, group=group)
    dist.destroy_process_group()
    return 0 if verdict[0] else 1


if __name__ == "__main__":
    sys.exit(main())
