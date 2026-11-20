# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 DSpark drafter's layers on gfx950: three launches a layer around the
draft attention, the TP all-reduces inside them (``gfx950/drafter.py``).

``install(model, group)`` gives the drafter (RadixArk/Kimi-K3-DSpark at TP 8) a
forward that runs, a layer: D-pre (the input norm, qkv, the q / k norms and RoPE),
the draft attention through the layer's own ``RadixAttention`` (the backend's verify
hook; the block's keys and values come in as arguments and are not stored: the
cache rows are the target's, written after verify), D-post (o_proj, its all-reduce
and the post norm's residual add) and D-MLP (the post norm, gate / up, SiLU, down,
its all-reduce and the next norm's residual add). The embedding and the final norm
(on the residual stream, the add done) stay SGLang's. A step of more than
``MAX_TOKENS`` rows runs the model's own forward.
"""

from __future__ import annotations

import logging
import types

import torch

from sgl_amd_mono.kimi_k3.gfx950 import drafter as D

logger = logging.getLogger(__name__)

# RadixArk/Kimi-K3-DSpark
DRAFT_CONFIG = dict(
    hidden_size=D.H,
    intermediate_size=D.I * D.TP,
    num_hidden_layers=5,
    num_attention_heads=D.Q_HEADS * D.TP,
    num_key_value_heads=D.KV_HEADS * D.TP,
    head_dim=D.HEAD_DIM,
    hidden_act="silu",
    rms_norm_eps=D.EPS,
    attention_bias=False,
)

SGLANG_READS = {
    "srt/models/dflash.py": (
        "layers", "norm", "embed_tokens", "forward_embed", "input_layernorm", "post_attention_layernorm",
        "self_attn", "qkv_proj", "q_norm", "k_norm", "rotary_emb", "cos_sin_cache", "o_proj", "attn", "mlp",
        "gate_up_proj", "down_proj", "attention_conv", "mlp_conv", "v_scale", "attention_sink_bias",
        "use_table_qk_norm_rope",
    ),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "next_token_logits", "hidden_states"),
    "srt/layers/radix_attention.py": ("forward(save_kv_cache)",),
}  # fmt: skip


def _views(layer) -> dict:
    a, m = layer.self_attn, layer.mlp
    return dict(
        w_in=layer.input_layernorm.weight,
        w_qkv=a.qkv_proj.weight,
        q_norm=a.q_norm.weight,
        k_norm=a.k_norm.weight,
        cos_sin=a.rotary_emb.cos_sin_cache,
        w_o=a.o_proj.weight,
        w_post=layer.post_attention_layernorm.weight,
        w_gu=m.gate_up_proj.weight,
        w_down=m.down_proj.weight,
    )


_SHAPES = {
    "w_in": (D.H,), "w_qkv": (D.QKV_ROWS, D.H), "q_norm": (D.HEAD_DIM,), "k_norm": (D.HEAD_DIM,),
    "w_o": (D.H, D.Q_K), "w_post": (D.H,), "w_gu": (D.GU_ROWS, D.H), "w_down": (D.H, D.I),
}  # fmt: skip


def disable_reasons(model, tp_size: int) -> list[str]:
    """Why the layer launches cannot take ``model`` (empty: they can)."""
    cfg = getattr(model, "config", None)
    reasons = []
    for k, want in DRAFT_CONFIG.items():
        got = getattr(cfg, k, None)
        if got != want:
            reasons.append(f"draft {k} {got}, the launches take {want}")
    if tp_size != D.TP:
        reasons.append(f"draft TP {tp_size}, the launches take {D.TP}")
    if reasons:
        return reasons
    for i, layer in enumerate(model.layers):
        a = layer.self_attn
        if layer.attention_conv is not None or layer.mlp_conv is not None:
            reasons.append(f"draft layer {i}: grouped convolutions")
        if a.v_scale is not None or a.attention_sink_bias is not None or not a.use_table_qk_norm_rope:
            reasons.append(f"draft layer {i}: a V scale, attention sinks or no table RoPE")
        for name, t in _views(layer).items():
            if name == "cos_sin":
                ok = t.dtype == torch.float32 and t.is_contiguous() and t.dim() == 2 and t.shape[1] == D.HEAD_DIM
                want = f"contiguous fp32 [*, {D.HEAD_DIM}]"
            else:
                ok = t.dtype == torch.bfloat16 and t.is_contiguous() and tuple(t.shape) == _SHAPES[name]
                want = f"contiguous bf16 {_SHAPES[name]}"
            if not ok:
                reasons.append(f"draft layer {i} {name}: {t.dtype} {tuple(t.shape)}, the launches take {want}")
    return reasons


def _p(t) -> int:
    return t.data_ptr()


class DrafterLayers:
    """One rank's drafter layer launches: the weights (views), the scratch, the two
    peer buffers (D-post's and D-MLP's), the step epoch and the row buffers."""

    def __init__(self, model, group, rank: int, device, deadline_s: float = 20.0):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer

        self.model, self.rank = model, rank
        self.views = [_views(layer) for layer in model.layers]
        dev = torch.device(device)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        ticks = guard.deadline_ticks(deadline_s)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self.scratch_pre = torch.zeros(D.scratch_bytes(D.dp_scratch_layout()), dtype=torch.uint8, device=dev)
        self.scratch_mlp = torch.zeros(D.scratch_bytes(D.dm_scratch_layout()), dtype=torch.uint8, device=dev)
        # collective: every rank builds them in this order
        self.sym_post = PeerBuffer(D.sym_bytes() + guard.FLAG_BYTES, group, rank, D.TP, dev)
        self.sym_mlp = PeerBuffer(D.sym_bytes() + guard.FLAG_BYTES, group, rank, D.TP, dev)
        n = D.MAX_TOKENS

        def rows(width):
            return torch.zeros(n, width, dtype=torch.bfloat16, device=dev)

        # the residual stream: D-post reads one and writes the other, D-MLP back
        self.res = (rows(D.H), rows(D.H))
        self.qkv = rows(D.QKV_ROWS)
        # each token's sums of squares by all-reduce CTA: D-post's for D-MLP, D-MLP's for the next D-pre
        self.ss = tuple(torch.zeros(n, D.SS_PARTS, dtype=torch.float32, device=dev) for _ in range(2))
        self._launch: dict = {}

    def kernels(self, s: int):
        if s not in self._launch:
            self._launch[s] = (
                D.build(D.DrafterBuild(kind="pre", tokens=s, first=True)),
                D.build(D.DrafterBuild(kind="pre", tokens=s)),
                D.build(D.DrafterBuild(kind="post", tokens=s)),
                D.build(D.DrafterBuild(kind="mlp", tokens=s)),
            )
        return self._launch[s]

    def layers(self, hidden, positions, forward_batch):
        """The draft layers on ``hidden`` (the embedded block rows, bf16 [S, H]): the
        residual stream after the last layer, the final norm's input."""
        s = hidden.shape[0]
        pre0, pre, post, mlp = self.kernels(s)
        stream = torch.cuda.current_stream()
        emb = hidden.contiguous()
        pos = positions.contiguous()
        pos64 = int(pos.dtype == torch.int64)
        sym_p, sym_m = self.sym_post, self.sym_mlp
        ra, rb = self.res[0][:s], self.res[1][:s]
        ss_mid, ss_next = self.ss
        qkv = self.qkv[:s]
        for i, layer in enumerate(self.model.layers):
            v = self.views[i]
            (pre0 if i == 0 else pre)(
                _p(emb if i == 0 else ra), _p(ss_next), _p(ra), _p(v["w_in"]), _p(v["w_qkv"]), _p(v["q_norm"]),
                _p(v["k_norm"]), _p(v["cos_sin"]), _p(pos), pos64, _p(qkv), _p(self.scratch_pre), _p(self.epoch), i,
                0, stream=stream,
            )  # fmt: skip
            q, k, vv = qkv.split([D.Q_K, D.KV_HEADS * D.HEAD_DIM, D.KV_HEADS * D.HEAD_DIM], dim=-1)
            o = layer.self_attn.attn(q, k, vv, forward_batch, save_kv_cache=False)
            o = o.reshape(s, D.Q_K)
            assert o.is_contiguous()
            post(
                _p(o), _p(v["w_o"]), _p(ra), _p(rb), _p(ss_mid), sym_p.local, _p(sym_p.addresses), self.rank,
                _p(self.epoch), i, 0, stream=stream,
            )  # fmt: skip
            mlp(
                _p(rb), _p(ss_mid), _p(v["w_post"]), _p(v["w_gu"]), _p(v["w_down"]), _p(rb), _p(ra), _p(ss_next),
                _p(self.scratch_mlp), sym_m.local, _p(sym_m.addresses), self.rank, _p(self.epoch), i, 0,
                stream=stream,
            )  # fmt: skip
        self.epoch[0].add_(1)
        return ra


def install(model, group, deadline_s: float = 20.0) -> bool:
    """Give the Kimi-K3 DSpark drafter ``model`` the layer launches (``group``: the TP
    group, whose CPU group the peer handshake takes). Returns whether it did; a
    model they do not take keeps its forward (the reasons logged)."""
    reasons = disable_reasons(model, group.world_size)
    if reasons:
        logger.info("kimi_k3 drafter layers: not installed: %s", "; ".join(reasons))
        return False
    runner = DrafterLayers(model, group.cpu_group, group.rank_in_group, model.norm.weight.device, deadline_s)
    original = model.forward

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, *args, **kwargs):
        s = positions.shape[0]
        if s == 0 or s > D.MAX_TOKENS:
            return original(input_ids, positions, forward_batch, input_embeds, *args, **kwargs)
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids) if self.embed_tokens is not None else self.forward_embed(input_ids)
        hidden = self.norm(runner.layers(input_embeds, positions, forward_batch))
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    model.forward = types.MethodType(forward, model)
    model.__dict__["_k3_drafter_layers"] = runner
    logger.info("kimi_k3 drafter layers: installed (up to %d rows)", D.MAX_TOKENS)
    return True
