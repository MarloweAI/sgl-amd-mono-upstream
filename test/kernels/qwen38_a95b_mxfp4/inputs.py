"""Random weights and inputs for the Qwen3.8 MoE gfx950 kernel checks: one TP8
rank's shards in SGLang's layouts, fp4 bytes any byte and e8m0 scales near 2^-8.
The launches run under the guard, so each check makes its epoch buffer with a
deadline (``epoch``)."""

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L

BF = torch.bfloat16
DEADLINE_S = 30.0
HIDDEN, CORE, QKVZ, BA, CONV = L.HIDDEN, L.CORE, L.QKVZ, L.BA, L.CONV
NV, HD, E, RI, SI, MAXT = L.NV, L.HD, L.E, L.RI, L.SI, L.MAX_TOKENS


def gen(dev, seed):
    return torch.Generator(device=dev).manual_seed(seed)


def rnd(g, *shape, scale=1.0, dtype=BF):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def codes(g, *shape, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, device=g.device, dtype=torch.uint8)


def epoch(dev, seconds=DEADLINE_S):
    """A launch's guard block: the epoch, the deadline, zero error words."""
    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(seconds)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def errors(e):
    return {w: int(e[w]) for w in guard.ERROR_WORDS}


def k1_weights(g):
    """``gdn_pre``'s weights."""
    w = {
        "ln_w": rnd(g, HIDDEN, scale=0.1),
        "w_ba": rnd(g, BA, HIDDEN, scale=0.05),
        "conv_w": rnd(g, CONV, 1, 4, scale=0.4),
        "a_log": rnd(g, NV, scale=0.5, dtype=torch.float32),
        "dt_bias": rnd(g, NV, scale=0.5),
        "norm_w": (1 + rnd(g, HD, scale=0.1, dtype=torch.float32)).to(BF),
    }
    w["w_qkvz"] = rnd(g, QKVZ, HIDDEN, scale=0.02)
    return w


def k1_state(g, slots, bf16=False):
    """The conv state (slots, dim, 3) and the SSM state (slots, HV, V, K), as
    SGLang's mamba pool holds them."""
    return rnd(g, slots, CONV, 3), rnd(
        g, slots, NV, HD, HD, scale=0.05, dtype=BF if bf16 else torch.float32
    )


def k2_weights(dev, rank, seed, fse=False):
    """``layer_post``'s weights: this rank's shards, then the replicated ones."""
    ne = E + int(fse)
    g = gen(dev, seed + rank)
    w = {
        "w13": codes(g, ne, 2 * RI, HIDDEN // 2),
        "w13s": codes(g, ne * 2 * RI * HIDDEN // 32, lo=117, hi=121),
        "w2": codes(g, ne, HIDDEN, RI // 2),
        "w2s": codes(g, ne * HIDDEN * RI // 32, lo=117, hi=121),
    }
    w["w_o"] = rnd(g, HIDDEN, CORE, scale=0.02)
    if not fse:
        w["w_sgu"] = rnd(g, 2 * SI, HIDDEN, scale=0.02)
        w["w_sd"] = rnd(g, HIDDEN, SI, scale=0.02)
    g = gen(dev, seed)
    w["ln_w"] = rnd(g, HIDDEN, scale=0.1)
    w["w_gate"] = rnd(g, E, HIDDEN, scale=0.02)
    w["w_sg"] = rnd(g, 1, HIDDEN, scale=0.02)
    return w
