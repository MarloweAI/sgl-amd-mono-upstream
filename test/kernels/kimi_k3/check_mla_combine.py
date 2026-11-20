# SPDX-License-Identifier: Apache-2.0
"""The MLA front's in-launch LSE combine and output gate (``mla_combine``)
against SGLang's ``_dcp_lse_combine_kernel`` and ``mla_output_gate_kernel``,
bit for bit, on one GPU:

* ``gate``: bf16(x * bf16(sigmoid(g))) on every bf16 g, x = 1 and x random;
* ``combine``: the combine and gate of 8 sources' records on synthetic
  partials and LSEs (wide LSE spreads, ties, SGLang's empty-row LSE -1e30,
  NaN, +inf and -inf LSEs, values of every scale), from ``combine4``, the
  function the launch's combine tasks run;
* ``dcp8`` (``--ctx``): ``check_mla_front.py --ranks 8`` covers the launch end
  to end; this runs it for every P mod 8 at small and long contexts.
"""

import argparse
import os
import subprocess
import sys

import torch

BF = torch.bfloat16
DCP, NHL, VD = 8, 12, 128


def _kernels():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import Int32, Int64, T

    from sgl_amd_mono.common.ops import bf16_pair, bf_lo, rsrc
    from sgl_amd_mono.common.plan import kernel_sources
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_combine as mc

    # the kernels' cache key: the code they trace in mla_combine
    keyed = kernel_sources(mc.__name__)

    @flyc.kernel(name="k3_check_gate", known_block_size=[256, 1, 1])
    def gate_kernel(x: Int64, g: Int64, out: Int64):
        _ = keyed
        i = fx.block_idx.x * 256 + fx.thread_idx.x
        xw = fx.Int32(bo.buffer_load(rsrc(x), i, vec_width=1, dtype=T.i32))
        gw = fx.Int32(bo.buffer_load(rsrc(g), i, vec_width=1, dtype=T.i32))
        lo = bf_lo(xw) * mc._sigmoid_bf16(bf_lo(gw))
        hi = (xw & fx.Int32(-65536)).bitcast(fx.Float32) * mc._sigmoid_bf16((gw & fx.Int32(-65536)).bitcast(fx.Float32))
        bo.buffer_store(bf16_pair(lo, hi).bitcast(fx.Int32), rsrc(out), i)

    @flyc.jit
    def gate_launch(x: Int64, g: Int64, out: Int64, n: Int32, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        gate_kernel(x, g, out).launch(grid=(n // 256,), block=(256,), stream=stream)

    # item i = (row b, local head h, cell u): its 8 sources' cells from recv
    # [8][B][12][130] bf16 (the LSE as fp32 word 64), its g from [B][1536]
    @flyc.kernel(name="k3_check_combine", known_block_size=[256, 1, 1])
    def comb_kernel(recv: Int64, g: Int64, out: Int64, nb: Int32):
        _ = keyed
        i = fx.block_idx.x * 256 + fx.thread_idx.x
        u = i % mc.VAL_CELLS
        bh = i // mc.VAL_CELLS
        rec_words = (VD + 2) // 2
        cells, lses = [], []
        for src in range_constexpr(DCP):
            base = (src * nb * NHL + bh) * rec_words
            v = fx.Vector(bo.buffer_load(rsrc(recv), base + 2 * u, vec_width=2, dtype=T.i32))
            cells.append((v[0], v[1]))
            lses.append(fx.Int32(bo.buffer_load(rsrc(recv), base + VD // 2, vec_width=1, dtype=T.i32)))
        b = bh // NHL
        h = bh % NHL
        gw = fx.Vector(bo.buffer_load(rsrc(g), (b * mc.GATE + h * VD + 4 * u) // 2, vec_width=2, dtype=T.i32))
        ow = mc.combine4(cells, lses, gw)
        bo.buffer_store(ow, rsrc(out), (b * mc.GATE + h * VD + 4 * u) // 2)

    @flyc.jit
    def comb_launch(recv: Int64, g: Int64, out: Int64, nb: Int32, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        comb_kernel(recv, g, out, nb).launch(grid=(nb * NHL * mc.VAL_CELLS // 256,), block=(256,), stream=stream)

    return gate_launch, comb_launch


def check_gate(device, gate_launch):
    from sglang.kernels.ops.elementwise import mla_output_gate

    bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
    g = bits.view(BF).contiguous()
    gen = torch.Generator(device=device).manual_seed(5)
    out = {}
    for name, x in (
        ("x=1", torch.ones_like(g)),
        ("x random", (torch.randn(65536, generator=gen, device=device) * 4).to(BF)),
    ):
        mine = torch.empty_like(g)
        gate_launch(x.data_ptr(), g.data_ptr(), mine.data_ptr(), 65536 // 2, stream=torch.cuda.current_stream())
        ref = mla_output_gate.kimi_k3_mla_output_gate(x.view(-1, 8), g.view(-1, 8)).view(-1)
        torch.cuda.synchronize()
        nan = torch.isnan(ref.float()) & torch.isnan(mine.float())
        bad = (mine.view(torch.int16) != ref.view(torch.int16)) & ~nan
        out[name] = (int(bad.sum()), g[bad][:6].float().tolist(), mine[bad][:6].float().tolist(), ref[bad][:6].float().tolist())
    return out


def synth_records(nb, seed, device):
    """recv [8, B, 12, 130] bf16 and g [B, 1536]: rows of every kind."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, device=device)  # noqa: E731
    vals = rn(DCP, nb, NHL, VD) * torch.exp2(torch.randint(-20, 12, (DCP, nb, NHL, 1), generator=gen, device=device).float())
    lse = rn(DCP, nb, NHL) * 4 + rn(1, nb, NHL) * 30
    kind = torch.randint(0, 16, (DCP, nb, NHL), generator=gen, device=device)
    lse = torch.where(kind == 0, torch.full_like(lse, -1e30), lse)  # an empty row on a rank
    lse = torch.where(kind == 1, lse[:1].expand_as(lse), lse)  # ties with source 0
    lse = torch.where(kind == 2, lse + 120, lse)  # far apart: tiny weights for the others
    lse = torch.where(kind == 3, lse - 90, lse)
    odd = torch.randint(0, 200, (DCP, nb, NHL), generator=gen, device=device)
    lse = torch.where(odd == 0, torch.full_like(lse, float("nan")), lse)
    lse = torch.where(odd == 1, torch.full_like(lse, float("inf")), lse)
    lse = torch.where(odd == 2, torch.full_like(lse, float("-inf")), lse)
    # some rows empty on every rank
    allempty = torch.randint(0, 64, (1, nb, NHL), generator=gen, device=device) == 0
    lse = torch.where(allempty, torch.full_like(lse, -1e30), lse)
    vals = torch.where(kind.unsqueeze(-1) == 0, torch.zeros_like(vals), vals)
    recv = torch.empty(DCP, nb, NHL, VD + 2, dtype=BF, device=device)
    recv[..., :VD] = vals.to(BF)
    recv.view(torch.float32)[..., VD // 2] = lse
    g = (rn(nb, NHL * VD) * torch.exp2(torch.randint(-6, 5, (nb, 1), generator=gen, device=device).float())).to(BF)
    return recv, g


def check_combine(device, comb_launch, nb, seed):
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    from sglang.kernels.ops.elementwise import mla_output_gate

    recv, g = synth_records(nb, seed, device)
    mine = torch.empty(nb, NHL * VD, dtype=BF, device=device)
    comb_launch(recv.data_ptr(), g.data_ptr(), mine.data_ptr(), nb, stream=torch.cuda.current_stream())
    comb, _ = dcp_lse_combine_triton(recv[..., :VD], recv.view(torch.float32)[..., VD // 2], is_lse_base_on_e=True)
    ref = mla_output_gate.kimi_k3_mla_output_gate(comb.reshape(nb, -1).contiguous(), g)
    torch.cuda.synchronize()
    nan = torch.isnan(ref.float()) & torch.isnan(mine.float())
    bad = (mine.view(torch.int16) != ref.view(torch.int16)) & ~nan
    return int(bad.sum()), mine.numel(), mine[bad][:6].float().tolist(), ref[bad][:6].float().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096, help="synthetic combine rows")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--ctx", type=int, nargs="*", default=[], help="contexts P: each P .. P + 7 through check_mla_front --ranks 8")
    ap.add_argument("--front-args", default="", help="more check_mla_front.py arguments")
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    fails = []
    gate_launch, comb_launch = _kernels()
    for name, (n, gs, got, want) in check_gate(device, gate_launch).items():
        print(f"gate {name}: {65536 - n}/65536 bf16 g bit for bit" + (f" BAD g {gs} got {got} want {want}" if n else ""), flush=True)
        if n:
            fails.append(f"gate {name}")
    for seed in range(args.seeds):
        n, tot, got, want = check_combine(device, comb_launch, args.rows, seed)
        print(f"combine seed {seed}: {tot - n}/{tot} gated core values bit for bit" + (f" BAD got {got} want {want}" if n else ""), flush=True)
        if n:
            fails.append(f"combine seed {seed}")
    here = os.path.dirname(os.path.abspath(__file__))
    for p0 in args.ctx:
        ps = [str(p0 + k) for k in range(DCP)]
        cmd = [sys.executable, os.path.join(here, "check_mla_front.py"), "--ranks", "8", "--skip-units", "--repeat", "1", "--ctx", *ps]
        cmd += args.front_args.split()
        print("RUN", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        if rc:
            fails.append(f"dcp8 P={p0}..{p0 + 7}")
    print("ALL PASS" if not fails else f"FAILED: {fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
