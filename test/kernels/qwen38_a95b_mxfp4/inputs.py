"""Random weights and inputs for the Qwen3.8 MoE gfx950 kernel checks: one TP8
rank's shards in SGLang's layouts, fp4 bytes any byte and e8m0 scales near 2^-8.
The launches run under the guard, so each check makes its epoch buffer with a
deadline (``epoch``)."""

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L

BF = torch.bfloat16
DEADLINE_S = 30.0
HIDDEN, CORE = L.HIDDEN, L.CORE
E, RI, SI, MAXT = L.E, L.RI, L.SI, L.MAX_TOKENS


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


def k2_weights(dev, rank, seed):
    """``layer_post``'s weights: this rank's shards, then the replicated ones."""
    g = gen(dev, seed + rank)
    w = {
        "w13": codes(g, E, 2 * RI, HIDDEN // 2),
        "w13s": codes(g, E * 2 * RI * HIDDEN // 32, lo=117, hi=121),
        "w2": codes(g, E, HIDDEN, RI // 2),
        "w2s": codes(g, E * HIDDEN * RI // 32, lo=117, hi=121),
    }
    w["w_o"] = rnd(g, HIDDEN, CORE, scale=0.02)
    w["w_sgu"] = rnd(g, 2 * SI, HIDDEN, scale=0.02)
    w["w_sd"] = rnd(g, HIDDEN, SI, scale=0.02)
    g = gen(dev, seed)
    w["ln_w"] = rnd(g, HIDDEN, scale=0.1)
    w["w_gate"] = rnd(g, E, HIDDEN, scale=0.02)
    w["w_sg"] = rnd(g, 1, HIDDEN, scale=0.02)
    return w
