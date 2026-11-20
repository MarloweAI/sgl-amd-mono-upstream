# SPDX-License-Identifier: Apache-2.0
"""One KDA layer's attention half at R requests x 6 rows, run two ways on the
same inputs: SGLang's own ReplaySSM verify path (in_proj GEMM, the tuned f_b
GEMM, ``kda_verify_gfx950`` on the ring, the gated RMSNorm) and the wide launch
(``kimi_k3/gfx950/attention/kda_wide.py``).

Weights: one rank's real layer from a tier-0 weight copy (``--copy DIR --layer
N``: ``L<N>.pt``, SGLang's post-load tensors by storage) or seeded random ones.
State: seeded random, in pools larger than the batch, with the requests'
slots and window rows shuffled so that no index equals another.
"""

import json
import os
from types import SimpleNamespace

import torch

BF, F32 = torch.bfloat16, torch.float32
H, NPROJ, NH, HD, PROJ = 7168, 6288, 12, 128, 1536
QKV = 3 * PROJ
L = 6
RING_LEN = 8  # SGLang's KDA spec ring: draft tokens rounded up to a power of two
EPS = 1e-5
LB = -5.0
SPLITS = [QKV, PROJ, HD, NH, NPROJ - QKV - PROJ - HD - NH]


def load_layer(copy, layer, device):
    """The KDA weights of ``layer`` from a tier-0 copy directory."""
    pre = f"language_model.model.layers.{layer}.self_attn."
    with open(os.path.join(copy, f"L{layer}.pt.json")) as f:
        meta = json.load(f)
    data = torch.load(os.path.join(copy, f"L{layer}.pt"), map_location="cpu", weights_only=True)

    def view(storage, name, shape, dtype):
        for vname, off, vshape, _stride, vdt in meta[storage]["views"]:
            if vname == name:
                n = 1
                for x in shape:
                    n *= x
                assert off + n * dtype.itemsize <= meta[storage]["nbytes"], name
                b = data[storage][off:]
                return b[: n * dtype.itemsize].view(dtype).view(*shape).to(device)
        raise KeyError(name)

    w = {
        "w_in": view(pre + "fused_qkvg_proj.weight", pre + "fused_qkvg_proj.weight", (NPROJ, H), BF),
        "w_fb": view(pre + "_bfa_f_b_w", pre + "f_b_proj.weight", (PROJ, HD), BF),
        "conv_w": view(pre + "qkv_conv1d.weight", pre + "attn.conv_weights", (QKV, 4), F32),
        "a_log": view(pre + "A_log", pre + "A_log", (NH,), F32),
        "dt_bias": view(pre + "dt_bias", pre + "dt_bias", (PROJ,), F32),
        "on_w": view(pre + "o_norm.weight", pre + "o_norm.weight", (HD,), BF),
    }
    del data
    return w


def random_layer(gen, device):
    def r(*shape, scale, dtype):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(dtype)

    return {
        "w_in": r(NPROJ, H, scale=0.02, dtype=BF),
        "w_fb": r(PROJ, HD, scale=0.1, dtype=BF),
        "conv_w": r(QKV, 4, scale=0.3, dtype=F32),
        "a_log": r(NH, scale=0.5, dtype=F32),
        "dt_bias": r(PROJ, scale=0.5, dtype=F32),
        "on_w": (1.0 + r(HD, scale=0.1, dtype=F32)).to(BF),
    }


def clone_layer(w):
    return {k: v.clone() for k, v in w.items()}


class Pools:
    """One layer's KDA pools: committed state and conv, verify windows, ring."""

    def __init__(self, R, gen, device, extra=3, state_scale=0.05):
        self.R = R
        n = R + extra
        self.temporal = torch.randn(n, NH, HD, HD, generator=gen, device=device) * state_scale
        self.conv = (torch.randn(n, 3, QKV, generator=gen, device=device) * 0.5).to(BF)
        self.window = torch.zeros(n, L, 3, QKV, dtype=BF, device=device)
        self.ring = (
            torch.zeros(n, NH, RING_LEN, HD, dtype=BF, device=device),
            torch.zeros(n, NH, RING_LEN, HD, dtype=BF, device=device),
            torch.zeros(n, NH, RING_LEN, HD, dtype=F32, device=device),
            torch.zeros(n, NH, RING_LEN, dtype=F32, device=device),
        )
        p = torch.randperm(n, generator=torch.Generator().manual_seed(R))
        q = torch.randperm(n, generator=torch.Generator().manual_seed(R + 100))
        self.slots = p[:R].to(torch.int32).to(device)
        self.rows = q[:R].to(torch.int32).to(device)

    def clone(self):
        o = Pools.__new__(Pools)
        o.R = self.R
        o.temporal, o.conv, o.window = self.temporal.clone(), self.conv.clone(), self.window.clone()
        o.ring = tuple(t.clone() for t in self.ring)
        o.slots, o.rows = self.slots, self.rows
        return o

    def outputs(self):
        s = self.slots.long()
        r = self.rows.long()
        return {
            "ring_k": self.ring[0][s, :, :L],
            "ring_v": self.ring[1][s, :, :L],
            "ring_g": self.ring[2][s, :, :L],
            "ring_b": self.ring[3][s, :, :L],
            "window": self.window[r],
        }


class Stock:
    """SGLang's ReplaySSM verify path of one layer, as ``KimiK3`` runs it on gfx950."""

    def __init__(self, w):
        from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.kimi_k3_amd.kimi_k3_hip import kda_f_b_gemm_hip

        self.w = w
        self.inproj = UnquantizedLinearMethod()
        self.merged = SimpleNamespace(weight=w["w_in"])
        self.f_b = kda_f_b_gemm_hip
        self.o_norm = FusedRMSNormGated(HD, eps=EPS, activation="sigmoid").to(w["w_in"].device)
        self.o_norm.weight.data = w["on_w"].clone()

    def proj(self, x):
        return self.inproj.apply(self.merged, x, None)

    def attn(self, fused, pools, keep=None):
        from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import kda_verify_gfx950

        qkv, g, f_a, beta, _ = torch.split(fused, SPLITS, dim=-1)
        fg = self.f_b(f_a, self.w["w_fb"])
        o = kda_verify_gfx950(
            mixed_qkv=qkv,
            conv_weight=self.w["conv_w"],
            conv_state=pools.conv.transpose(-1, -2),
            intermediate_conv_window=pools.window.transpose(-1, -2),
            intermediate_state_indices=pools.rows,
            a=fg,
            b=beta,
            A_log=self.w["a_log"],
            dt_bias=self.w["dt_bias"],
            ssm_states=pools.temporal,
            cache_indices=pools.slots,
            intermediate_states_buffer=None,
            scale=HD**-0.5,
            T=L,
            num_q_heads=NH,
            num_v_heads=NH,
            head_k_dim=HD,
            head_v_dim=HD,
            lower_bound=LB,
            cache_ring=True,
            replayssm_rawv=pools.ring[1],
            replayssm_rawk=pools.ring[0],
            replayssm_g=pools.ring[2],
            replayssm_beta=pools.ring[3],
        )
        if keep is not None:
            keep["o"] = o.squeeze(0).flatten(-2).clone()
        core = self.o_norm(o, g.unflatten(-1, (-1, HD)))
        return core.squeeze(0).flatten(-2)

    def __call__(self, x, pools, keep=None):
        fused = self.proj(x)
        if keep is not None:
            keep["proj"] = fused
        return self.attn(fused, pools, keep)


class Wide:
    """The wide launch on the same weights and pools."""

    def __init__(self, w, S, device, debug=False, depth=0, layer=1, timeline=False):
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

        self.kw = kw
        self.w = w
        self.w_in = kw.pack_w_in(w["w_in"])
        self.key = kw.KdaWideBuild(
            tokens=S, qlen=L, onorm_eps=EPS, lower_bound=LB, debug=debug, depth=depth, timeline=timeline
        )
        self.tl = (
            torch.zeros(kw.BLOCKS, kw.TL_POINTS, dtype=torch.int64, device=device) if timeline else None
        )
        # uncached, as the layer runner's scratch
        self._scratch = PeerBuffer(kw.scratch_bytes(self.key), None, 0, 1, device)
        self.scratch = self._scratch.bytes
        self.layer = layer
        self.S = S
        self.dbg = (
            (torch.zeros(S, NPROJ, dtype=BF, device=device), torch.zeros(S, PROJ, dtype=BF, device=device))
            if debug
            else (None, None)
        )

    def packed_x(self, x):
        """``pack_x`` of x, made once per x (as the producer writes it)."""
        k = (x.data_ptr(), x._version)
        if getattr(self, "_xk", None) != k:
            self._xk, self._xp = k, self.kw.pack_x(x, self.kw.tiles(self.key))
        return self._xp

    def __call__(self, x, pools, epoch, core):
        self.kw.kda_wide(
            self.key,
            x=self.packed_x(x),
            w_in=self.w_in,
            w_fb=self.w["w_fb"],
            conv_w=self.w["conv_w"],
            conv_state=pools.conv.mT,
            a_log=self.w["a_log"],
            dt_bias=self.w["dt_bias"],
            on_w=self.w["on_w"],
            rstate=pools.temporal,
            st_idx=pools.slots,
            conv_win=pools.window,
            win_row=pools.rows,
            ring=pools.ring,
            core_out=core,
            scratch=self.scratch,
            layer=self.layer,
            epoch=epoch,
            proj_dbg=self.dbg[0],
            o_dbg=self.dbg[1],
            tl=self.tl,
        )


def floor_bytes(R):
    """Bytes the region must move at R requests (weights, x, state, conv, ring,
    windows, core)."""
    S = R * L
    w = NPROJ * H * 2 + PROJ * HD * 2 + QKV * 16 + PROJ * 4 + NH * 4 + HD * 2
    per_req = NH * HD * HD * 4 + 3 * QKV * 2 + NH * L * (HD * 2 * 2 + HD * 4 + 4) + L * 3 * QKV * 2
    return w + S * H * 2 + R * per_req + S * PROJ * 2


def cmp(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    d = (a - b).abs()
    bad = (a != b).nonzero().flatten()[:4]
    return {
        "examples": [[float(a[i]), float(b[i])] for i in bad.tolist()],
        "equal": bool(torch.equal(a, b)),
        "neq": int((a != b).sum()),
        "n": a.numel(),
        "max_abs": float(d.max()),
        "rel_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
    }
