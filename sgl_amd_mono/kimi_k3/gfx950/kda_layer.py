# SPDX-License-Identifier: Apache-2.0
"""``kda_layer``: one Kimi-K3 KDA layer of 1 to 8 requests as one launch. The
KDA attention (``attention/kda_wide``'s body, with its seam front, or reading
the x the previous layer's K2 tail left) runs first, then K2 (``layer.k2_body``: o_proj, its all-reduce, the MLP seam and the
MoE), whose o_proj reads the core the attention's tasks wrote, chunk by chunk
behind each (request, local head) box (``K2Build.core_box``). A CTA runs its
attention role, then its K2 role: the CTAs with no KDA task start K2, and so its
o_proj weight stream, while the tasks still run.

Scratch, the runner's (``layer.scratch_regions``): K2's layout from 0, as its
own launch has it (its CBOX among it, its tagged x rows at ``tag_offset``); the
attention's in its own region from ``attn_offset``, then the core [S, 1536]
bf16 (device-scope stores and loads).

LDS: one union of the attention's (``kda_wide.lds_types``) and K2's
(``layer.lds_types``).

The host side is ``layer.k2_launch(attn=Attn(...))``: K1's wide form hands its
launch values to the layer's K2 launch, which runs both.
"""

from dataclasses import dataclass
from typing import NamedTuple

import flydsl.compiler as flyc
import msgspec
import flydsl.expr as fx
import torch
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common.ops import KernArgs
from sgl_amd_mono.common.plan import BLOCKS, THREADS, kernel_sources, key_tuple
from sgl_amd_mono.common.sync import sreg
from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw
from sgl_amd_mono.kimi_k3.gfx950.stages import oproj
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import PROJ
from sgl_amd_mono.kimi_k3.gfx950.tag import step_tag

SOURCES = kernel_sources(__name__)


@dataclass(frozen=True)
class KdaLayerBuild:
    attn: kw.KdaWideBuild  # the wide form: its seam front, or (front off) x given
    k2: k2.K2Build  # the layer's K2 key, core_box on


class Attn(NamedTuple):
    """The attention part K1 hands the layer's K2 launch: its build key and
    ``kda_wide.launch_values`` (core into ``core_view``, scratch from
    ``attn_offset``)."""

    key: kw.KdaWideBuild
    vals: tuple


def attn_offset() -> int:
    """Where the attention's scratch starts: its region (``layer.scratch_regions``)."""
    return k2.scratch_regions()["attn"][0]


def _rel_layout(akey: kw.KdaWideBuild) -> dict:
    n = kw.scratch_bytes(akey)
    return {"attn": (0, n), "core": (-(-n // 256) * 256, akey.tokens * PROJ * 2)}


def region_bytes(akey: kw.KdaWideBuild) -> int:
    """The attention's region a launch of ``akey`` takes: its scratch, then its core."""
    return max(o + n for o, n in _rel_layout(akey).values())


def scratch_layout(akey: kw.KdaWideBuild) -> dict:
    """name -> (byte offset, bytes) in the runner's scratch: ``attn``, ``core``."""
    o = attn_offset()
    return {name: (o + r, n) for name, (r, n) in _rel_layout(akey).items()}


def core_view(akey: kw.KdaWideBuild, scratch):
    """The launch's core [S, 1536] bf16 in the runner's ``scratch``."""
    off, n = scratch_layout(akey)["core"]
    return scratch[off : off + n].view(torch.bfloat16).view(akey.tokens, PROJ)


def attn_scratch(akey: kw.KdaWideBuild, scratch):
    """The attention's scratch in the runner's ``scratch``."""
    off, n = scratch_layout(akey)["attn"]
    return scratch[off : off + n]


# the KDA task CTAs (from TASK0, the last to start K2) take one o_proj task each
O_SHIFT = BLOCKS - kw.TASK0

_BUILDS: dict = {}
_STREAM = fx.Stream(None)

# K2's kernel parameters, by K2_ARGS name, as this kernel names them
_K2_PARAMS = {n: "k_" + n for n in k2.K2_ARGS}


def build(key: KdaLayerBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    akey, kkey = key.attn, key.k2
    s = akey.tokens
    assert not kw.is_gemm(akey) and kkey.core_box and kkey.tokens == s
    # a block-write front stores its bank row plain; K2 reads the bank before any poll
    assert akey.write_idx < 0
    # a core box is a request's verify rows of a local head
    assert akey.qlen == oproj.CORE_REQ_ROWS and s // akey.qlen <= kw.WIDE_REQS
    cbox_off, cbox_n = k2.scratch_layout(kkey)["cbox"]
    at, kt = kw.lds_types(akey), k2.lds_types(kkey)

    @fx.struct
    class AttnPart:
        smem: at.smem
        role: at.role

    @fx.struct
    class K2Part:
        smem: kt.smem
        route: kt.route
        stage: kt.stage

    @fx.union
    class LayerLds:
        attn: AttnPart
        k2: K2Part

    keyed = key_tuple(key, SOURCES)
    name = (
        f"k3_mono_kda_layer_s{s}_" + (f"f{akey.nblocks}" if akey.front else "x")
        + (f"a{akey.aux_slot}" if akey.aux_slot >= 0 else "")
        + f"_k{kkey.nblocks}"
        f"_t{int(akey.timeline)}{int(kkey.timeline)}_o{kkey.opt}"
        + (f"_tl{kkey.tail.tiles}" if kkey.tail is not None else "")
    )

    def layer_body(
        x: Int64, w_in: Int64, w_fb: Int64, conv_w: Int64, conv_st: Int64,
        a_log: Int64, dt_bias: Int64, on_w: Int64, rstate: Int64, st_idx: Int64,
        win_row: Int64, conv_win: Int64, ring_k: Int64, ring_v: Int64,
        ring_g: Int64, ring_b: Int64, core_out: Int64, proj_dbg: Int64,
        o_dbg: Int64, cs_seq: Int32, cs_dim: Int32, cs_tok: Int32, rs_seq: Int32,
        cw_seq: Int32, cw_tok: Int32, cw_q: Int32, rk_seq: Int32, rv_seq: Int32,
        rg_seq: Int32, rb_seq: Int32, ring_len: Int32, scratch: Int64,
        layer: Int32, epoch: Int64, tl: Int64, prefix: Int64, blocks: Int64,
        blk_sm: Int32, blk_sr: Int32, ares_nw: Int64, ares_qk: Int64,
        in_nw: Int64, reqs: Int32, aux: Int64,
        k_core: Int64, k_core_ld: Int32, k_w_o: Int64, k_prefix: Int64,
        k_prefix_in: Int64, k_blocks: Int64, k_blk_sm: Int32, k_blk_sr: Int32,
        k_ares_nw: Int64, k_ares_qk: Int64, k_in_nw: Int64, k_w_gate: Int64,
        k_bias: Int64, k_w_ld: Int64, k_w_sgu: Int64, k_w_sd: Int64, k_w13: Int64,
        k_w13s: Int64, k_w2: Int64, k_w2s: Int64, k_ln_w: Int64, k_w_up: Int64,
        k_out: Int64, k_scratch: Int64, k_peers: Int64, k_rank: Int32,
        k_epoch: Int64, k_layer: Int32, k_tl: Int64, k_tail: Int64,
    ):  # fmt: skip
        _ = keyed
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(LayerLds)
        params = (
            x, w_in, w_fb, conv_w, conv_st, a_log, dt_bias, on_w, rstate, st_idx,
            win_row, conv_win, ring_k, ring_v, ring_g, ring_b, core_out, proj_dbg,
            o_dbg, cs_seq, cs_dim, cs_tok, rs_seq, cw_seq, cw_tok, cw_q, rk_seq,
            rv_seq, rg_seq, rb_seq, ring_len, layer, epoch, tl, prefix, blocks,
            blk_sm, blk_sr, ares_nw, ares_qk, in_nw, reqs, aux,
        )  # fmt: skip
        args = dict(zip(kw.KDA_ARGS, params))
        # each task's box holds K2's tag once its core rows are stored
        args["core_box"] = (sreg(k_scratch, cbox_off, "cbox"), step_tag(k_epoch, k_layer))
        kw.kda_body(akey, lds.attn.smem.peek(), lds.attn.role, scratch, args)
        # the CTA's attention LDS is done with before its K2 role reuses it
        gpu.barrier()
        # K2's arguments re-read here (scalar loads of the kernarg segment), so
        # none is held in SGPRs across the attention
        a = ka(*_K2_PARAMS.values(), "k_scratch")
        k2.k2_body(
            kkey,
            k2.K2Lds(smem=lds.k2.smem.peek(), route=lds.k2.route.peek(), stage=lds.k2.stage),
            a.k_scratch,
            {n: getattr(a, p) for n, p in _K2_PARAMS.items()},
        )

    ka = KernArgs(
        [(n, 8 if t is Int64 else 4) for n, t in layer_body.__annotations__.items() if n != "return"]
    )
    kda_layer = flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])(layer_body)

    @flyc.jit
    def run(
        x: Int64, w_in: Int64, w_fb: Int64, conv_w: Int64, conv_st: Int64,
        a_log: Int64, dt_bias: Int64, on_w: Int64, rstate: Int64, st_idx: Int64,
        win_row: Int64, conv_win: Int64, ring_k: Int64, ring_v: Int64,
        ring_g: Int64, ring_b: Int64, core_out: Int64, proj_dbg: Int64,
        o_dbg: Int64, cs_seq: Int32, cs_dim: Int32, cs_tok: Int32, rs_seq: Int32,
        cw_seq: Int32, cw_tok: Int32, cw_q: Int32, rk_seq: Int32, rv_seq: Int32,
        rg_seq: Int32, rb_seq: Int32, ring_len: Int32, scratch: Int64,
        layer: Int32, epoch: Int64, tl: Int64, prefix: Int64, blocks: Int64,
        blk_sm: Int32, blk_sr: Int32, ares_nw: Int64, ares_qk: Int64,
        in_nw: Int64, reqs: Int32, aux: Int64,
        k_core: Int64, k_core_ld: Int32, k_w_o: Int64, k_prefix: Int64,
        k_prefix_in: Int64, k_blocks: Int64, k_blk_sm: Int32, k_blk_sr: Int32,
        k_ares_nw: Int64, k_ares_qk: Int64, k_in_nw: Int64, k_w_gate: Int64,
        k_bias: Int64, k_w_ld: Int64, k_w_sgu: Int64, k_w_sd: Int64, k_w13: Int64,
        k_w13s: Int64, k_w2: Int64, k_w2s: Int64, k_ln_w: Int64, k_w_up: Int64,
        k_out: Int64, k_scratch: Int64, k_peers: Int64, k_rank: Int32,
        k_epoch: Int64, k_layer: Int32, k_tl: Int64, k_tail: Int64,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        kda_layer(
            x, w_in, w_fb, conv_w, conv_st, a_log, dt_bias, on_w, rstate, st_idx,
            win_row, conv_win, ring_k, ring_v, ring_g, ring_b, core_out, proj_dbg,
            o_dbg, cs_seq, cs_dim, cs_tok, rs_seq, cw_seq, cw_tok, cw_q, rk_seq,
            rv_seq, rg_seq, rb_seq, ring_len, scratch, layer, epoch, tl, prefix,
            blocks, blk_sm, blk_sr, ares_nw, ares_qk, in_nw, reqs, aux,
            k_core, k_core_ld, k_w_o, k_prefix, k_prefix_in, k_blocks, k_blk_sm,
            k_blk_sr, k_ares_nw, k_ares_qk, k_in_nw, k_w_gate, k_bias, k_w_ld,
            k_w_sgu, k_w_sd, k_w13, k_w13s, k_w2, k_w2s, k_ln_w, k_w_up, k_out,
            k_scratch, k_peers, k_rank, k_epoch, k_layer, k_tl, k_tail,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    _BUILDS[key] = run
    return run


def run_layer(attn: Attn, kkey: k2.K2Build, k2_vals: tuple) -> None:
    """The layer's one launch: ``attn`` from K1, ``k2_vals`` K2's kernel
    arguments as ``layer.k2_launch`` passes them (its core ``core_view``)."""
    assert not kkey.core_box, "the layer's K2 key is its own launch's"
    key = KdaLayerBuild(
        attn=attn.key, k2=msgspec.structs.replace(kkey, core_box=True, o_shift=O_SHIFT)
    )
    build(key)(*attn.vals, *k2_vals, stream=torch.cuda.current_stream())
