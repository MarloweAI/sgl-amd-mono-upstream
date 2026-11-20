# SPDX-License-Identifier: Apache-2.0
"""The DSpark aux capture from the KDA seam (``KdaWideBuild.aux_slot``): at the
layer after each captured target layer (7 / 23 / 51 / 67 / 83), the front's
pre-norm AttnRes mixture into ``aux_hidden_states_buffer[row, j * 7168 ..]``,
against SGLang's capture (``aggregate_stream_hip``: ``attn_res_hip`` with no
output norm) on the same layer's real weights: within one bf16 step at each
row's scale (the two reduce in different orders; the drafter then reads the
mixture the target's own x is normed from), the differing elements counted; the
other slots untouched; x bit for bit unchanged by the flag.

    python3 check_kda_aux.py --copy ~/pe-k3/copy/<key>
"""

import argparse
import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kda_wide_case as kc  # noqa: E402
from check_kda_layer import Front, seam_weights  # noqa: E402

LAYERS = (8, 24, 52, 68, 84)  # slot j's layer
BLOCK = 12  # attn_res_block_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", required=True)
    ap.add_argument("--reqs", default="1,2,4,8")
    args = ap.parse_args()
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    gen = torch.Generator(device=device).manual_seed(0)
    bad = 0
    for j, layer in enumerate(LAYERS):
        nb = -(-layer // BLOCK)
        w = kc.load_layer(args.copy, layer, device)
        sw = seam_weights(args.copy, layer, device, gen)
        cw = (sw["ares_nw"].float() * sw["ares_qk"].float()).contiguous()
        for R in map(int, args.reqs.split(",")):
            S = R * kc.L
            g = torch.Generator(device=device).manual_seed(900 + 10 * layer + R)
            stream = (torch.randn(S, kc.H, generator=g, device=device) * 0.3).to(kc.BF)
            blocks = (torch.randn(S, nb + 1, kc.H, generator=g, device=device) * 0.3).to(kc.BF)
            ref = torch.empty_like(stream)
            attn_res_hip(stream, blocks, cw, None, ref, nb, kc.EPS, 0.0)
            outs = {}
            for slot in (-1, j):
                front = Front(w, sw, S, nb, -1, device, layer=layer)
                front.key = dataclasses.replace(front.key, aux_slot=slot)
                aux = torch.full((S, 5 * kc.H), 7.0, dtype=kc.BF, device=device) if slot >= 0 else None
                pools = kc.Pools(R, g, device)
                core = torch.empty(S, kc.PROJ, dtype=kc.BF, device=device)
                kw.kda_wide(
                    front.key, x=front.x, w_in=front.w_in, w_fb=w["w_fb"], conv_w=w["conv_w"],
                    conv_state=pools.conv.mT, a_log=w["a_log"], dt_bias=w["dt_bias"],
                    on_w=w["on_w"], rstate=pools.temporal, st_idx=pools.slots,
                    conv_win=pools.window, win_row=pools.rows, ring=pools.ring, core_out=core,
                    scratch=front.scratch, layer=layer, epoch=torch.ones(1, dtype=torch.int32, device=device),
                    prefix=stream, blocks=blocks, aux=aux, **sw,
                )  # fmt: skip
                torch.cuda.synchronize()
                outs[slot] = (front.x_rows(S).clone(), aux)
            aux = outs[j][1]
            got = aux[:, j * kc.H : (j + 1) * kc.H]
            others = torch.cat([aux[:, : j * kc.H], aux[:, (j + 1) * kc.H :]], dim=1)
            r = {
                "layer": layer, "slot": j, "R": R, "nvb": nb,
                "vs_sglang": kc.cmp(got, ref),
                "others_untouched": bool((others == 7.0).all()),
                "x_unchanged": bool(torch.equal(outs[j][0], outs[-1][0])),
            }  # fmt: skip
            # bf16 steps apart at the element (cancellations near 0 are a few)
            # and at its row's scale (a step of the row's largest magnitude)
            steps = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
            row_step = ref.float().abs().amax(dim=1, keepdim=True) * 2.0**-7
            d = (got.float() - ref.float()).abs()
            r["max_bf16_steps"] = int(steps.max())
            r["max_row_steps"] = float((d / row_step).max())
            r["ok"] = r["max_row_steps"] <= 1 and r["others_untouched"] and r["x_unchanged"]
            bad += not r["ok"]
            print("KDA_AUX", json.dumps(r), flush=True)
    print("KDA_AUX_DONE", "fail" if bad else "pass", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
