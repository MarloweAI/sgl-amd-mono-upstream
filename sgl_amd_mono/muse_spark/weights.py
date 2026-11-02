# SPDX-License-Identifier: Apache-2.0
"""What the kernels read of one Muse Spark decoder layer, checked at startup.

The kernels read tensors through raw pointers: a differently-laid-out weight
would not fault, it would produce plausible garbage. So every module name, dtype,
shape and constant the kernels assume is checked once (``layer_reasons``, a refusal
at startup) on SGLang's ROCm Muse Spark modules, and only then are the views taken
(``LayerViews.build``).

Everything is read in place except the copies logged at startup: a norm's
effective weight (``weight + gain_center``) where the model has not cached it, a
residual gate's alpha / beta where not precomputed, and the router's bf16 hi + lo
split where the model did not pack it next to the pooler (``SGLANG_MUSE_JOINT_ROUTER_POOLER``).
"""

from __future__ import annotations

import math
import os

import msgspec
import torch

# SGLANG_AMD_MONO_MUSE_QUANT_BF16=1: the bf16-expert layers (0 and 61 of the Quark
# checkpoint, as of the NVFP4 one) run the kernels on an online MXFP4 quantization of
# their experts (aiter's rule, see mxfp4_quant.py) instead of SGLang's modules. Off:
# the checkpoints keep these layers bf16 on purpose; on was ~5% faster at C1.
QUANT_BF16 = os.environ.get("SGLANG_AMD_MONO_MUSE_QUANT_BF16", "0") == "1"
# SGLANG_AMD_MONO_MUSE_FP8_DENSE=0: the pooler and unpooler stay bf16 (else the kernels
# stream FP8 E4M3 copies with a scale a row, 36 % of a layer's bytes halved; see fp8_quant.py)
# off: per-row E4M3 pooler / unpooler copies were -0.2 ms a C1 step but cut AIME26 pass@1 0.958 -> 0.908 (native
# 0.958 / 0.967 over two seeds; the same build with them off 0.950): long reasoning cannot afford them
FP8_DENSE = os.environ.get("SGLANG_AMD_MONO_MUSE_FP8_DENSE", "0") == "1"

from sgl_amd_mono.muse_spark import TP
from sgl_amd_mono.muse_spark.gfx950.config import (
    EXPERTS,
    GATE_TEMPERATURE,
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_EXPERTS,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MOE_HIDDEN,
    MXFP4_GROUP,
    POST_EPS,
    Q_K,
    QK_SCALE,
    QKVG_ROWS,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    SOFTMAX_SCALE,
    TOP_K,
    WINDOW,
)

# The SGLang names these checks read, by file under the sglang package ("x": a
# string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/rocm_muse_spark/model.py": (
        "is_moe_layer",
        "self_attn",
        "mlp",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "use_qk_norm",
        "use_o_gate",
        "use_o_norm",
        "scaling",
        "scale_query_by",
        "norm_eps",
        "qkv_proj",
        "o_proj",
        "rotary_emb",
        "attn",
        "pre_attn_norm",
        "post_attn_norm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_attention_residual_gate",
        "post_feedforward_residual_gate",
        "tp_size",
        "ep_size",
        "num_local_experts",
        "top_k",
        "route_scale",
        "route_eps",
        "router_multiplier",
        "expert_hidden_size",
        "intermediate_size",
        "router",
        '"expert_bias"',
        "pre_expert_proj",
        "post_expert_proj",
        "pre_expert_norm",
        "post_expert_norm",
        "experts",
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "MuseSpark1xMxfp4Experts",
        "MuseSpark1xNativeExperts",
    ),
    "srt/rocm_muse_spark/layers.py": (
        "MuseSpark1xRMSNorm",
        "MuseSpark1xRMSNormWithInputScale",
        "eps",
        "has_weight",
        "weight",
        "gate",
        "temperature",
        "_get_effective_weight",
        "_effective_weight_cache",
        "_coefficients",
        "_alpha",
        "_beta",
    ),
    "srt/rocm_muse_spark/router_pooler.py": ('"router_pooler_packed_weight"',),
    "srt/layers/rotary_embedding/*.py": (
        "cos_sin_cache",
        "is_neox_style",
        "rotary_dim",
    ),
    "srt/layers/radix_attention.py": ("sliding_window_size", "k_scale", "v_scale"),
}

NORM_TYPES = ("MuseSpark1xRMSNorm", "MuseSpark1xRMSNormWithInputScale")
EXPERT_TYPES = ("MuseSpark1xMxfp4Experts", "MuseSpark1xNativeExperts")


def _close(a, b: float) -> bool:
    return math.isclose(float(a), b, rel_tol=1e-9, abs_tol=0.0)


def _tensor_reasons(t, want_dtype, want_shape, what: str) -> list[str]:
    if t is None:
        return [f"{what}: missing"]
    if not (
        t.dtype == want_dtype
        and tuple(t.shape) == tuple(want_shape)
        and t.is_contiguous()
    ):
        return [
            f"{what}: {t.dtype} {tuple(t.shape)}, want contiguous {want_dtype} {tuple(want_shape)}"
        ]
    return []


def _norm_reasons(
    module, n: int, what: str, *, eps: float, weighted: bool
) -> list[str]:
    """A MuseSpark1xRMSNorm of width ``n`` and ``eps``, with (``weighted``) a bf16
    gain the kernels read as ``weight + gain_center``."""
    if type(module).__name__ not in NORM_TYPES:
        return [
            f"{what}: {type(module).__name__}, the kernels apply MuseSpark1xRMSNorm"
        ]
    why = []
    if not _close(module.eps, eps):
        why.append(f"{what}: eps {module.eps}, built for {eps}")
    if bool(module.has_weight) != weighted:
        why.append(f"{what}: {'weighted' if module.has_weight else 'weightless'}, built for {'weighted' if weighted else 'weightless'}")  # fmt: skip
    elif weighted:
        why += _tensor_reasons(module.weight, torch.bfloat16, (n,), f"{what} weight")
    return why


def _gate_reasons(module, what: str) -> list[str]:
    """A channel-wise norm-preserving residual gate at the built temperature."""
    g = module.gate
    if g is None or g.numel() != HIDDEN:
        return [
            f"{what}: gate {None if g is None else tuple(g.shape)}, want channel-wise [{HIDDEN}]"
        ]
    if not _close(module.temperature, GATE_TEMPERATURE):
        return [
            f"{what}: temperature {module.temperature}, built for {GATE_TEMPERATURE}"
        ]
    return []


def _attention_reasons(attn, i: int) -> list[str]:
    why = []
    shape = (attn.num_heads, attn.num_kv_heads, attn.head_dim)
    if shape != (LOCAL_Q_HEADS, LOCAL_KV_HEADS, HEAD_DIM):
        why.append(f"layer {i}: heads / dims {shape} a rank, built for {(LOCAL_Q_HEADS, LOCAL_KV_HEADS, HEAD_DIM)}")  # fmt: skip
    if not (attn.use_qk_norm and attn.use_o_gate and attn.use_o_norm):
        why.append(
            f"layer {i}: the kernels apply the q / k norms, the o norm and the o gate"
        )
    if not _close(attn.scaling, SOFTMAX_SCALE):
        why.append(
            f"layer {i}: softmax scale {attn.scaling}, built for {SOFTMAX_SCALE}"
        )
    if not _close(attn.scale_query_by, QK_SCALE):
        why.append(f"layer {i}: q scale {attn.scale_query_by}, built for {QK_SCALE}")
    if not _close(attn.norm_eps, RMS_EPS):
        why.append(
            f"layer {i}: q / k / o norm eps {attn.norm_eps}, built for {RMS_EPS}"
        )
    why += _tensor_reasons(
        attn.qkv_proj.weight, torch.bfloat16, (QKVG_ROWS, HIDDEN), f"layer {i} qkv_proj"
    )
    why += _tensor_reasons(
        attn.o_proj.weight, torch.bfloat16, (HIDDEN, Q_K), f"layer {i} o_proj"
    )
    rope = attn.rotary_emb
    if rope is not None:
        table = rope.cos_sin_cache
        if not (
            table.dim() == 2
            and table.shape[1] == HEAD_DIM
            and table.dtype == torch.float32
        ):
            why.append(f"layer {i}: cos / sin cache {table.dtype} {tuple(table.shape)}, want fp32 [*, {HEAD_DIM}]")  # fmt: skip
        if rope.is_neox_style or rope.rotary_dim != HEAD_DIM:
            why.append(f"layer {i}: rotary neox {rope.is_neox_style} dim {rope.rotary_dim}, built for interleaved {HEAD_DIM}")  # fmt: skip
    window = attn.attn.sliding_window_size
    if window not in (WINDOW, -1):
        why.append(
            f"layer {i}: sliding window {window}, built for {WINDOW} or full attention"
        )
    for scale, what in ((attn.attn.k_scale, "k_scale"), (attn.attn.v_scale, "v_scale")):
        if isinstance(scale, torch.Tensor) and scale.numel() != 1:
            why.append(f"layer {i} {what}: {tuple(scale.shape)}, want one scale")
    why += _norm_reasons(
        attn.pre_attn_norm,
        HIDDEN,
        f"layer {i} pre_attn_norm",
        eps=RMS_EPS,
        weighted=True,
    )
    return why


def _experts_reasons(experts, i: int) -> list[str]:
    kind = type(experts).__name__
    if kind not in EXPERT_TYPES:
        return [
            f"layer {i}: experts {kind}, the kernels read Quark MXFP4 or bf16 experts"
        ]
    E, what = LOCAL_EXPERTS, f"layer {i} experts"
    if kind == "MuseSpark1xNativeExperts":
        return _tensor_reasons(
            experts.w13_weight,
            torch.bfloat16,
            (E, 2 * INTER, MOE_HIDDEN),
            f"{what} w13",
        ) + _tensor_reasons(
            experts.w2_weight, torch.bfloat16, (E, MOE_HIDDEN, INTER), f"{what} w2"
        )
    why = _tensor_reasons(
        experts.w13_weight, torch.uint8, (E, 2 * INTER, MOE_HIDDEN // 2), f"{what} w13"
    )
    why += _tensor_reasons(
        experts.w2_weight, torch.uint8, (E, MOE_HIDDEN, INTER // 2), f"{what} w2"
    )
    why += _tensor_reasons(experts.w13_weight_scale, torch.uint8, (E, 2 * INTER, MOE_HIDDEN // MXFP4_GROUP), f"{what} w13 scale")  # fmt: skip
    why += _tensor_reasons(experts.w2_weight_scale, torch.uint8, (E, MOE_HIDDEN, INTER // MXFP4_GROUP), f"{what} w2 scale")  # fmt: skip
    return why


def _moe_reasons(moe, i: int) -> list[str]:
    why = []
    if (moe.tp_size, moe.ep_size) != (TP, TP):
        why.append(
            f"layer {i}: MoE TP {moe.tp_size} / EP {moe.ep_size}, the kernels shard experts over EP {TP}"
        )
    shape = (
        moe.num_local_experts,
        moe.top_k,
        moe.expert_hidden_size,
        moe.intermediate_size,
    )
    if shape != (LOCAL_EXPERTS, TOP_K, MOE_HIDDEN, INTER):
        why.append(f"layer {i}: experts / top-k / pooled / inter {shape}, built for {(LOCAL_EXPERTS, TOP_K, MOE_HIDDEN, INTER)}")  # fmt: skip
    for got, want, what in ((moe.route_scale, ROUTE_SCALE, "route scale"), (moe.route_eps, ROUTE_EPS, "route eps"),
                            (moe.router_multiplier, ROUTER_MULT, "router multiplier")):  # fmt: skip
        if not _close(got, want):
            why.append(f"layer {i}: {what} {got}, built for {want}")
    why += _tensor_reasons(
        moe.router.weight, torch.float32, (HIDDEN, EXPERTS), f"layer {i} router"
    )
    why += _tensor_reasons(
        moe.expert_bias, torch.float32, (EXPERTS,), f"layer {i} expert_bias"
    )
    for proj, shape, what in ((moe.pre_expert_proj, (MOE_HIDDEN, HIDDEN), "pooler"), (moe.post_expert_proj, (HIDDEN, MOE_HIDDEN), "unpooler")):  # fmt: skip
        if proj is None:
            why.append(
                f"layer {i}: no {what}; the kernels pool {HIDDEN} to {MOE_HIDDEN}"
            )
        else:
            why += _tensor_reasons(
                proj.weight, torch.bfloat16, shape, f"layer {i} {what}"
            )
    why += _norm_reasons(
        moe.pre_expert_norm,
        MOE_HIDDEN,
        f"layer {i} pre_expert_norm",
        eps=RMS_EPS,
        weighted=True,
    )
    why += _norm_reasons(
        moe.post_expert_norm,
        MOE_HIDDEN,
        f"layer {i} post_expert_norm",
        eps=POST_EPS,
        weighted=True,
    )
    return why + _experts_reasons(moe.experts, i)


def layer_reasons(layer, i: int) -> list[str]:
    """Every reason the kernels cannot read decoder layer ``i`` (empty: they can)."""
    if not layer.is_moe_layer:
        return [f"layer {i}: a dense MLP layer; the kernels run routed experts"]
    why = _attention_reasons(layer.self_attn, i)
    why += _norm_reasons(
        layer.post_attn_norm,
        HIDDEN,
        f"layer {i} post_attn_norm",
        eps=POST_EPS,
        weighted=False,
    )
    why += _norm_reasons(layer.pre_feedforward_layernorm, HIDDEN, f"layer {i} pre_feedforward_layernorm", eps=RMS_EPS, weighted=True)  # fmt: skip
    why += _norm_reasons(layer.post_feedforward_layernorm, HIDDEN, f"layer {i} post_feedforward_layernorm", eps=POST_EPS, weighted=True)  # fmt: skip
    why += _gate_reasons(
        layer.post_attention_residual_gate, f"layer {i} post_attention_residual_gate"
    )
    why += _gate_reasons(
        layer.post_feedforward_residual_gate,
        f"layer {i} post_feedforward_residual_gate",
    )
    return why + _moe_reasons(layer.mlp, i)


def expert_kind(layer) -> str:
    """``"mxfp4"`` or ``"bf16"``: which experts a layer ``layer_reasons`` accepted holds."""
    return (
        "bf16"
        if type(layer.mlp.experts).__name__ == "MuseSpark1xNativeExperts"
        else "mxfp4"
    )


def _effective_norm_weight(module, i: int, what: str, copies: dict) -> torch.Tensor:
    """A norm's ``weight + gain_center`` as the kernels read it: the model's cache
    where it made one, else a copy kept in ``copies``."""
    cached = getattr(module, "_effective_weight_cache", None)
    if cached is not None:
        return cached.data
    key = ("effective norm weight", i, what)
    if key not in copies:
        with torch.no_grad():
            copies[key] = (
                module._get_effective_weight().detach().to(torch.bfloat16).contiguous()
            )
    return copies[key]


def _gate_coefficients(
    gate, i: int, what: str, copies: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """A residual gate's (alpha, beta) fp32 [HIDDEN]: the precomputed ones, else a copy."""
    if gate._alpha is not None and gate._beta is not None:
        return gate._alpha.reshape(-1).float().contiguous(), gate._beta.reshape(
            -1
        ).float().contiguous()
    keys = (("residual gate", i, what, "alpha"), ("residual gate", i, what, "beta"))
    if keys[0] not in copies:
        with torch.no_grad():
            alpha, beta = gate._coefficients(
                gate.gate.view(1, -1).float() / gate.temperature
            )
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.full_like(beta, float(alpha))
            copies[keys[0]] = alpha.reshape(-1).float().contiguous()
            copies[keys[1]] = beta.reshape(-1).float().contiguous()
    return copies[keys[0]], copies[keys[1]]


def _quantized_experts(experts, i: int, copies: dict) -> tuple:
    """A bf16-expert layer's experts as Quark-layout MXFP4 copies (w13, s13, w2, s2),
    quantized once at the build by aiter's rule and kept in ``copies``."""
    from sgl_amd_mono.muse_spark.mxfp4_quant import quantize_mxfp4

    keys = tuple(
        ("online MXFP4 experts", i, name) for name in ("w13", "s13", "w2", "s2")
    )
    if keys[0] not in copies:
        with torch.no_grad():
            w13, s13 = quantize_mxfp4(experts.w13_weight.data)
            w2, s2 = quantize_mxfp4(experts.w2_weight.data)
            for k, t in zip(keys, (w13, s13, w2, s2)):
                copies[k] = t
    return tuple(copies[k] for k in keys)


def _fp8_dense(weight: torch.Tensor, i: int, what: str, copies: dict) -> torch.Tensor:
    """A dense bf16 weight's packed FP8 copy (``pack_fp8_rows``), kept in ``copies``."""
    from sgl_amd_mono.muse_spark.fp8_quant import pack_fp8_rows, pack_fp8_rows_pooler

    key = ("fp8 dense", i, what)
    if key not in copies:
        with torch.no_grad():
            pack = (
                pack_fp8_rows_pooler if what == "pooler" else pack_fp8_rows
            )  # the pooler's lane-swizzled layout
            copies[key] = pack(weight.data)
    return copies[key]


def _router_hi_lo(moe, i: int, copies: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The fp32 router as bf16 hi + lo [EXPERTS, HIDDEN]: the model's packed weight
    (rows 4096 .. 4352 and 4352 .. 4608 next to the pooler) when it prepared one,
    else copies."""
    packed = getattr(moe, "router_pooler_packed_weight", None)
    if packed is not None and tuple(packed.shape) == (MOE_HIDDEN + 2 * EXPERTS, HIDDEN):
        return packed[MOE_HIDDEN : MOE_HIDDEN + EXPERTS], packed[MOE_HIDDEN + EXPERTS :]
    keys = (("router hi/lo", i, "hi"), ("router hi/lo", i, "lo"))
    if keys[0] not in copies:
        with torch.no_grad():
            hi = moe.router.weight.data.T.to(torch.bfloat16).contiguous()
            lo = (moe.router.weight.data.T - hi.float()).to(torch.bfloat16).contiguous()
            copies[keys[0]], copies[keys[1]] = hi, lo
    return copies[keys[0]], copies[keys[1]]


class LayerViews(msgspec.Struct, frozen=True):
    """Everything the launches read for one decoder layer ``layer_reasons`` accepted."""

    layer_id: int
    kind: str  # "mxfp4" or "bf16" experts
    is_swa: bool
    use_rope: bool
    cos_sin: object  # fp32 [max_pos, HEAD_DIM] (cos | sin) or None on NoPE layers
    w_qkvg: torch.Tensor  # bf16 [QKVG_ROWS, HIDDEN]
    w_o: torch.Tensor  # bf16 [HIDDEN, Q_K]
    alpha: torch.Tensor  # fp32 [HIDDEN]: the post-attention residual gate
    beta: torch.Tensor
    pre_w: torch.Tensor  # bf16 [HIDDEN]: pre_feedforward_layernorm (+1)
    gate_hi: torch.Tensor  # bf16 [EXPERTS, HIDDEN]
    gate_lo: torch.Tensor
    bias: torch.Tensor  # fp32 [EXPERTS]
    w_pool: torch.Tensor  # bf16 [MOE_HIDDEN, HIDDEN]
    xn_w: torch.Tensor  # bf16 [MOE_HIDDEN]: pre_expert_norm (+1)
    w13: object  # MXFP4 u8 [LOCAL_EXPERTS, 2 INTER, MOE_HIDDEN / 2] or None (bf16 experts)
    s13: object
    w2: object
    s2: object
    pen_w: torch.Tensor  # bf16 [MOE_HIDDEN]: post_expert_norm (+1)
    w_unpool: torch.Tensor  # bf16 [HIDDEN, MOE_HIDDEN]
    pfw: torch.Tensor  # bf16 [HIDDEN]: post_feedforward_layernorm (+1)
    alpha2: torch.Tensor  # fp32 [HIDDEN]: the post-FFN residual gate
    beta2: torch.Tensor
    npw: (
        torch.Tensor
    )  # bf16 [HIDDEN]: the next layer's pre_attn_norm (+1), or the final norm's
    quantized: bool = False  # bf16 experts quantized online (kind is then "mxfp4")
    w_pool_fp8: object = None  # packed FP8 copy of w_pool (``pack_fp8_rows``), or None
    w_unpool_fp8: object = None  # packed FP8 copy of w_unpool, or None

    @staticmethod
    def build(layer, i: int, next_pre_norm, copies: dict) -> LayerViews:
        attn, moe, experts = layer.self_attn, layer.mlp, layer.mlp.experts
        kind = expert_kind(layer)
        quantized = kind == "bf16" and QUANT_BF16
        if quantized:
            w13, s13, w2, s2 = _quantized_experts(experts, i, copies)
            kind = "mxfp4"
        elif kind == "mxfp4":
            w13, s13, w2, s2 = experts.w13_weight.data, experts.w13_weight_scale.data, experts.w2_weight.data, experts.w2_weight_scale.data  # fmt: skip
        else:
            w13 = s13 = w2 = s2 = None
        alpha, beta = _gate_coefficients(
            layer.post_attention_residual_gate, i, "post_attention", copies
        )
        alpha2, beta2 = _gate_coefficients(
            layer.post_feedforward_residual_gate, i, "post_feedforward", copies
        )
        hi, lo = _router_hi_lo(moe, i, copies)
        return LayerViews(
            layer_id=i,
            kind=kind,
            is_swa=attn.attn.sliding_window_size == WINDOW,
            use_rope=attn.rotary_emb is not None,
            cos_sin=None
            if attn.rotary_emb is None
            else attn.rotary_emb.cos_sin_cache.data,
            w_qkvg=attn.qkv_proj.weight.data,
            w_o=attn.o_proj.weight.data,
            alpha=alpha,
            beta=beta,
            pre_w=_effective_norm_weight(
                layer.pre_feedforward_layernorm, i, "pre_feedforward_layernorm", copies
            ),
            gate_hi=hi,
            gate_lo=lo,
            bias=moe.expert_bias.data,
            w_pool=moe.pre_expert_proj.weight.data,
            xn_w=_effective_norm_weight(
                moe.pre_expert_norm, i, "pre_expert_norm", copies
            ),
            w13=w13,
            s13=s13,
            w2=w2,
            s2=s2,
            pen_w=_effective_norm_weight(
                moe.post_expert_norm, i, "post_expert_norm", copies
            ),
            w_unpool=moe.post_expert_proj.weight.data,
            pfw=_effective_norm_weight(
                layer.post_feedforward_layernorm,
                i,
                "post_feedforward_layernorm",
                copies,
            ),
            alpha2=alpha2,
            beta2=beta2,
            npw=_effective_norm_weight(next_pre_norm, i, "next pre_attn_norm", copies),
            quantized=quantized,
            w_pool_fp8=_fp8_dense(moe.pre_expert_proj.weight, i, "pooler", copies)
            if FP8_DENSE
            else None,
            w_unpool_fp8=_fp8_dense(moe.post_expert_proj.weight, i, "unpooler", copies)
            if FP8_DENSE
            else None,
        )


def tp_expert_copies(views, modules, rank: int, npes: int, group) -> list:
    """TP experts (``K3Build.tpx``, ATOM's TP-sharded MoE): for every kernel layer, this
    rank's slices of every expert -- the gate and up rows of its INTER / TP
    intermediate columns and the down rows of its MOE_HIDDEN / TP output columns,
    with their scales -- as (w13, s13, w2, s2) [EXPERTS, rows, bytes] in global
    expert order; None for a layer the kernels do not run. EP holds whole experts
    a rank, so the slices come from every rank: one all-to-all a tensor over the TP
    device group (``group``). A copy: EXPERTS / TP x (INTER and MOE_HIDDEN slices)
    is LOCAL_EXPERTS whole experts' bytes, ~0.8 GB a layer."""
    import torch.distributed as dist

    RS, CS = INTER // npes, MOE_HIDDEN // npes
    out = []

    def a2a(parts):
        inp = torch.stack(parts).contiguous()
        if npes == 1:
            return inp
        got = torch.empty_like(inp)
        dist.all_to_all_single(got, inp, group=group)
        return got

    with torch.no_grad():
        for i, L in enumerate(views):
            if L.kind != "mxfp4" or L.w13 is None:
                out.append(None)
                continue
            E = L.w13.shape[0]
            start = int(modules[i].mlp.local_expert_start)
            if E * npes != EXPERTS or start != rank * E:
                raise ValueError(f"layer {i}: experts {E} a rank from {start} on rank {rank}: TP experts take EP's contiguous {EXPERTS // npes} a rank")  # fmt: skip

            def rows13(t, d):
                return torch.cat(
                    [
                        t[:, d * RS : (d + 1) * RS],
                        t[:, INTER + d * RS : INTER + (d + 1) * RS],
                    ],
                    dim=1,
                )

            w13 = a2a([rows13(L.w13, d) for d in range(npes)]).reshape(
                EXPERTS, 2 * RS, -1
            )
            s13 = a2a([rows13(L.s13, d) for d in range(npes)]).reshape(
                EXPERTS, 2 * RS, -1
            )
            w2 = a2a([L.w2[:, d * CS : (d + 1) * CS] for d in range(npes)]).reshape(
                EXPERTS, CS, -1
            )
            s2 = a2a([L.s2[:, d * CS : (d + 1) * CS] for d in range(npes)]).reshape(
                EXPERTS, CS, -1
            )
            out.append((w13, s13, w2, s2))
    return out
