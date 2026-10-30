# SPDX-License-Identifier: Apache-2.0
"""The layer views the Qwen3.8-Flash-Next launches read, from SGLang's params.

The MXFP4 experts (FusedMoE w13 / w2, AITER's (16, 16) shuffle, the shared expert
fused at index 512) and the FP8 projections AITER pre-shuffles are read in place.
The small bf16 tensors (hyper-connection norms and mixers, GDN norms, q / k norms,
router, PLE) are SGLang's params too: the views pack the ones the kernel reads in
its MFMA tile layout, and every pack is a copy logged in ``copies``. Each pointer
the kernel reads is checked first (dtype, shape, contiguity, shuffle): a weight in
another layout would not fault, it would produce plausible garbage.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import layout as LY
from . import packing
from .config import (
    GDN_CONV_DIM,
    GDN_HV,
    HC_DIM,
    HC_LOWRANK,
    HIDDEN,
    N_EXPERTS,
)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/quantization/fp8_utils.py": ("use_aiter_bpreshuffle_gemm",),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "num_fused_shared_experts",
        "moe_runner_config",
    ),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "is_shuffled",
    ),
    "srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py": (
        "weight",
        "weight_scale",
    ),
    "srt/layers/moe/moe_runner/base.py": ("activation",),
    "srt/layers/hyperconnection.py": (
        "hc_norm",
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
    ),
    "srt/models/qwen3_5.py": (
        "layers",
        "mlp",
        "linear_attn",
        "conv1d",
        "A_log",
        "dt_bias",
        "norm",
        "in_proj_qkvz",
        "in_proj_ba",
        "out_proj",
        "qkv_proj",
        "o_proj",
        "q_norm",
        "k_norm",
    ),
    "srt/layers/attention/qsa/qsa_indexer.py": (
        "index_qk_proj",
        "q_layernorm",
        "k_layernorm",
    ),
    "srt/models/qwen2_moe.py": ("gate", "shared_expert_gate", "experts"),
    "srt/models/qwen4_exp_mtp.py": (
        "model",
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "fc_embedding",
        "fc_hidden",
    ),
    "srt/models/qwen4_exp.py": (
        "attn_hyper_connection",
        "mlp_hyper_connection",
        "hyper_connection_mixer",
        "indexer",
        "ple",
        "ple_embedding",
        "ngram_embedding",
        '"layer_multipliers"',
        '"ngram_heads_vocab_sizes"',
        '"ngram_heads_offsets"',
        "key_proj",
        "value_proj",
        "norm_key",
        "norm_query",
        "norm_conv",
    ),
}

# kernel name -> (SGLang param of the decoder layer, kernel rows, K)
GDN_PROJ = {
    "ip": ("linear_attn.in_proj_qkvz", 16384, 2560),
    "out": ("linear_attn.out_proj", 2560, 6144),
    "ba": ("linear_attn.in_proj_ba", LY.BA_ROWS, 2560),  # 96 real rows
}
# the QSA decoder layer holds qkv_proj / o_proj itself and the QSA indexer as .indexer
QSA_PROJ = {
    "qp": ("qkv_proj", LY.QP_RG_QKV * 16, 2560),
    "qp2": ("indexer.index_qk_proj", LY.QP_ROWS - LY.QP_RG_QKV * 16, 2560),
    "out": ("o_proj", 2560, 6144),
}

_BF16, _F32 = torch.bfloat16, torch.float32
_HC_NAMES = ("attn_hyper_connection", "mlp_hyper_connection")


def _hc_tensors(prefix: str, inject: bool = True) -> dict:
    out = {
        f"{prefix}.hc_norm.weight": ((HC_DIM,), (_BF16,)),
        f"{prefix}.input_mix_weight_down.weight": ((HC_LOWRANK, HC_DIM), (_BF16,)),
        f"{prefix}.input_mix_weight_up.weight": ((HC_DIM, HC_LOWRANK), (_BF16,)),
    }
    if inject:
        out[f"{prefix}.block_inject_weight.weight"] = ((4, HC_DIM), (_BF16,))
    return out


# The small tensors of a layer the packers read, by the checkpoint's name in the
# layer (what ``packing`` takes) -> (SGLang's param path in the decoder layer,
# shape, accepted dtypes). SGLang's loader drops the checkpoint's ``self_attn.``.
def _small(kind: str, ple: bool) -> dict:
    t = {}
    for hc in _HC_NAMES:
        t.update({k: (k, *v) for k, v in _hc_tensors(hc).items()})
    t["mlp.gate.weight"] = ("mlp.gate.weight", (N_EXPERTS, HIDDEN), (_BF16,))
    t["mlp.shared_expert_gate.weight"] = (
        "mlp.shared_expert_gate.weight",
        (1, HIDDEN),
        (_BF16,),
    )
    if kind == "gdn":
        # A_log, dt_bias and the gated norm are read as f32 (the packers convert)
        t["linear_attn.A_log"] = ("linear_attn.A_log", (GDN_HV,), (_BF16, _F32))
        t["linear_attn.dt_bias"] = ("linear_attn.dt_bias", (GDN_HV,), (_BF16, _F32))
        t["linear_attn.norm.weight"] = ("linear_attn.norm.weight", (128,), (_BF16, _F32))  # fmt: skip
        t["linear_attn.conv1d.weight"] = (
            "linear_attn.conv1d.weight",
            (GDN_CONV_DIM, 1, 4),
            (_BF16,),
        )
    else:
        for sg, n in (
            ("q_norm.weight", 256),
            ("k_norm.weight", 256),
            ("indexer.q_layernorm.weight", 128),
            ("indexer.k_layernorm.weight", 128),
        ):
            t[f"self_attn.{sg}"] = (sg, (n,), (_BF16,))
    if ple:
        for name, shape in (
            ("key_proj.weight", (HC_DIM, HIDDEN)),
            ("value_proj.weight", (HIDDEN, HIDDEN)),
            ("norm_key.weight", (HC_DIM,)),
            ("norm_query.weight", (HC_DIM,)),
            ("norm_conv.weight", (HC_DIM,)),
            ("conv1d.weight", (HC_DIM, 1, 4)),
        ):
            t[f"ple.{name}"] = (f"ple.{name}", shape, (_BF16,))
    return t


def _attr(module, path: str):
    """The object at dotted ``path`` under ``module``, or None."""
    obj = module
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _get(module, path: str):
    """The tensor at dotted ``path`` under ``module`` (a param's data), or None."""
    obj = _attr(module, path)
    return obj.data if isinstance(obj, torch.nn.Parameter) else obj


def _tensor_reason(t, shape, dtypes, what: str) -> Optional[str]:
    if not isinstance(t, torch.Tensor):
        return f"no {what}"
    if tuple(t.shape) != tuple(shape) or t.dtype not in dtypes:
        want = " or ".join(str(d).replace("torch.", "") for d in dtypes)
        return (
            f"{what} is {str(t.dtype).replace('torch.', '')} {tuple(t.shape)}, the"
            f" kernel reads {want} {tuple(shape)}"
        )
    if not t.is_contiguous():
        return f"{what} is not contiguous"
    return None


def moe_reasons(experts) -> list[str]:
    """What the kernel assumes of SGLang's MXFP4 FusedMoE params."""
    reasons = []
    if getattr(experts, "num_fused_shared_experts", 0) != 1:
        reasons.append(
            "needs the shared expert fused into the FusedMoE as expert 512 (AITER "
            "shared-expert fusion)"
        )
    activation = getattr(getattr(experts, "moe_runner_config", None), "activation", None)  # fmt: skip
    if activation not in (None, "silu"):
        # SiTU reorders w13 into AITER's gate / up interleaved layout
        reasons.append(f"expert activation {activation!r}, the kernel runs silu")
    E, IP = N_EXPERTS + 1, LY.MOE_INTER_PAD
    for name, shape in (
        ("w13_weight", (E, 2 * IP, HIDDEN // 2)),
        ("w2_weight", (E, HIDDEN, IP // 2)),
        ("w13_weight_scale", (E, 2 * IP, LY.GU_SGROUPS)),
        ("w2_weight_scale", (E, HIDDEN, LY.DN_SGROUPS)),
    ):
        t = _get(experts, name)
        # the scales are E8M0 bytes, whichever 1-byte dtype holds them
        scale = name.endswith("_scale") and isinstance(t, torch.Tensor)
        ok = (t.dtype,) if scale and t.element_size() == 1 else (torch.uint8,)
        why = _tensor_reason(t, shape, ok, f"experts.{name}")
        if why is not None:
            reasons.append(why)
        elif name.endswith("_weight") and not getattr(
            _attr(experts, name), "is_shuffled", False
        ):
            reasons.append(
                f"experts.{name} is not in AITER's (16, 16) shuffle (SGLANG_USE_AITER "
                "on gfx950 shuffles MXFP4 experts at load)"
            )
    return reasons


def proj_reasons(layer, table: dict) -> list[str]:
    """What the kernel assumes of SGLang's Quark W8A8 FP8 per-channel linears."""
    reasons = []
    for _, (name, rows, k) in table.items():
        w = _get(layer, f"{name}.weight")
        sc = _get(layer, f"{name}.weight_scale")
        if not isinstance(w, torch.Tensor):
            reasons.append(f"no {name}.weight")
            continue
        if w.dtype != torch.float8_e4m3fn:
            reasons.append(
                f"{name}.weight is {str(w.dtype).replace('torch.', '')}, the kernel "
                "reads FP8 E4M3 (Quark W8A8)"
            )
            continue
        if not isinstance(sc, torch.Tensor):
            # SGLang serves a linear whose N is not a multiple of 16 dequantized on HIP
            reasons.append(f"{name} has no per-channel weight scale")
            continue
        n = sc.numel()
        if sc.dtype != torch.float32 or not sc.is_contiguous() or n > rows:
            reasons.append(
                f"{name}.weight_scale is {str(sc.dtype).replace('torch.', '')} [{n}],"
                f" the kernel reads float32 [<= {rows}]"
            )
            continue
        wt = w.t() if tuple(w.shape) == (k, n) else w
        if tuple(wt.shape) != (n, k) or not wt.is_contiguous():
            reasons.append(
                f"{name}.weight is {tuple(w.shape)} with strides {w.stride()}, the kernel "
                f"reads a contiguous [{n}, {k}]"
            )
    return reasons


def small_reasons(layer, kind: str, ple: bool) -> list[str]:
    return [
        why
        for _, (path, shape, dtypes) in _small(kind, ple).items()
        if (why := _tensor_reason(_get(layer, path), shape, dtypes, path)) is not None
    ]


def layer_reasons(layer, i: int, kind: str) -> list[str]:
    """Every assumption the kernel makes of decoder layer ``i`` (a ``kind`` layer),
    as "layer i: ..." reasons; none raises."""
    mlp = getattr(layer, "mlp", None)
    reasons = moe_reasons(getattr(mlp, "experts", None))
    reasons += proj_reasons(layer, GDN_PROJ if kind == "gdn" else QSA_PROJ)
    reasons += small_reasons(layer, kind, ple=getattr(layer, "ple", None) is not None)
    return [f"layer {i}: {why}" for why in reasons]


def mixer_reasons(mixer) -> list[str]:
    """The model-level hyper-connection mixer (no combine)."""
    return [
        f"hyper_connection_mixer: {why}"
        for path, (shape, dtypes) in _hc_tensors("m", inject=False).items()
        if (why := _tensor_reason(_get(mixer, path[2:]), shape, dtypes, path[2:]))
        is not None
    ]


# ---------------------------------------------------------------- views
def moe_params(experts) -> dict:
    """SGLang's FusedMoE params (AITER MXFP4 layout), read in place."""
    m = {
        "gu": experts.w13_weight.data,
        "gu_s": experts.w13_weight_scale.data.view(torch.uint8),
        "dn": experts.w2_weight.data,
        "dn_s": experts.w2_weight_scale.data.view(torch.uint8),
    }
    packing.check_moe_params(m)
    return m


def proj_params(layer, table: dict) -> dict:
    """SGLang's Quark W8A8 FP8 per-channel params of a decoder layer -> {kernel name:
    (weight [rows, K] in AITER's shuffle_weight (16, 16) layout, f32 scale [rows])}.
    The ones SGLang keeps AITER-shuffled are read in place; the narrow ones it keeps
    row-major (N not a multiple of 64: use_aiter_bpreshuffle_gemm) are a zero-padded,
    shuffled copy with unit scales on the pad rows."""
    from aiter.ops.shuffle import shuffle_weight
    from sglang.srt.layers.quantization.fp8_utils import use_aiter_bpreshuffle_gemm

    params = dict(layer.named_parameters())
    out = {}
    for key, (name, rows, k) in table.items():
        w, sc = params[name + ".weight"].data, params[name + ".weight_scale"].data
        n = sc.numel()
        # SGLang keeps the [N, K] weight behind a transposed view
        wt = w.t() if tuple(w.shape) == (k, n) else w
        if n == rows and use_aiter_bpreshuffle_gemm(n):
            kw, ks = wt, sc.reshape(-1)
        else:
            if use_aiter_bpreshuffle_gemm(n):
                raise ValueError(
                    f"Qwen3.8-Flash-Next monokernel: {name} has {n} rows, past the "
                    f"kernel's {rows}"
                )
            kw = torch.zeros(rows, k, dtype=wt.dtype, device=wt.device)
            kw[:n] = wt
            kw = shuffle_weight(kw, (16, 16))
            ks = torch.ones(rows, dtype=torch.float32, device=sc.device)
            ks[:n] = sc.reshape(-1)
        out[key] = (kw, ks)
    return out


def checkpoint_dict(layer, kind: str, ple: bool) -> dict:
    """The layer's small tensors by their checkpoint names in the layer, as the
    packers take them: SGLang's params, read in place."""
    return {ck: _get(layer, path) for ck, (path, _, _) in _small(kind, ple).items()}


def _log_copies(copies: dict, kind: str, ids: tuple, packed, params: set) -> None:
    """Record every packed tensor that is not one of SGLang's params."""
    if isinstance(packed, torch.Tensor):
        if packed.data_ptr() not in params:
            copies[(kind, *ids)] = packed
    elif isinstance(packed, dict):
        for name, v in packed.items():
            _log_copies(copies, kind, (*ids, name), v, params)
    elif isinstance(packed, tuple):
        for j, v in enumerate(packed):
            _log_copies(copies, kind, (*ids, j), v, params)


def param_ptrs(model) -> set:
    return {p.data_ptr() for p in model.parameters()}


def layer_views(layer, i: int, kind: str, copies: dict, params: set) -> dict:
    """Decoder layer ``i``'s packed views (``packing.pack_gdn_layer`` /
    ``pack_qsa_layer``), with its PLE views under "ple" when it carries PLE."""
    ple = getattr(layer, "ple", None) is not None
    lw = checkpoint_dict(layer, kind, ple)
    moe = moe_params(layer.mlp.experts)
    proj = proj_params(layer, GDN_PROJ if kind == "gdn" else QSA_PROJ)
    if kind == "gdn":
        views = packing.pack_gdn_layer(lw, moe=moe, proj=proj)
    else:
        views = packing.pack_qsa_layer(lw, moe=moe, proj=proj)
    _log_copies(copies, f"{kind} layer packed", (i,), views, params)
    if ple:
        views["ple"] = packing.pack_ple(lw)
        _log_copies(copies, "ple packed", (i,), views["ple"], params)
    return views


def mixer_views(mixer, copies: dict, params: set) -> dict:
    """The model-level hyper-connection mixer, packed (no combine: its gate tile is
    zero)."""
    views = {
        "norm": mixer.hc_norm.weight.data.contiguous(),
        "down": packing.pack_hc_down(mixer.input_mix_weight_down.weight.data),
        "up": packing.pack_bf16(mixer.input_mix_weight_up.weight.data.contiguous()),
    }
    _log_copies(copies, "final mixer packed", (), views, params)
    return views


def ngram_views(ple_layer, copies: dict, params: set) -> dict:
    """Layer 1's PLE n-gram embedding: the FP8 table and its scale in place, and the
    int64 metadata (layer multipliers, per-head vocab sizes and offsets) as one
    table."""
    emb = ple_layer.ple_embedding
    meta = (
        torch.cat(
            [
                emb.layer_multipliers,
                emb.ngram_heads_vocab_sizes,
                emb.ngram_heads_offsets,
            ]
        )
        .to(dtype=torch.int64)
        .contiguous()
    )
    _log_copies(copies, "ngram metadata", (1,), meta, params)
    return {
        "meta": meta,
        "table": emb.ngram_embedding.weight,
        "scale": emb.ngram_embedding.weight_scale,
    }


def ngram_reasons(ple_layer) -> list[str]:
    emb = getattr(ple_layer, "ple_embedding", None)
    table = getattr(getattr(emb, "ngram_embedding", None), "weight", None)
    reasons = []
    if not isinstance(table, torch.Tensor) or table.dtype != torch.float8_e4m3fn:
        reasons.append(
            "layer 1: the PLE n-gram table must be FP8 E4M3 (the PLEFP8 checkpoint)"
        )
    elif table.shape[1] != LY.NGRAM_DIM or table.device.type == "cpu":
        reasons.append(
            f"layer 1: the PLE n-gram table is {tuple(table.shape)} on "
            f"{table.device}, the kernel reads it as a device pointer [n, "
            f"{LY.NGRAM_DIM}] (no --ple-offload-embedding)"
        )
    return reasons


# ---------------------------------------------------------------- the MTP draft
_MTP_FUSION = {
    "pre_fc_norm_embedding.weight": (HIDDEN,),
    "pre_fc_norm_hidden.weight": (HC_DIM,),
    "fc_embedding.weight": (HIDDEN, HIDDEN),
    "fc_hidden.weight": (HIDDEN, HIDDEN),
}


def mtp_reasons(lm) -> list[str]:
    """What the draft launch assumes of SGLang's ``Qwen4ExpForCausalLMMTP``: its QSA
    layer, its input fusion (separate embedding / hidden projections) and its mixer."""
    layers = getattr(getattr(lm, "model", None), "layers", None)
    if layers is None or len(layers) != 1:
        return ["the MTP draft must hold one decoder layer"]
    reasons = layer_reasons(layers[0], 0, "qsa")
    reasons += [
        f"MTP input fusion: {why}"
        for name, shape in _MTP_FUSION.items()
        if (why := _tensor_reason(_get(lm, name), shape, (_BF16,), name)) is not None
    ]
    reasons += mixer_reasons(getattr(lm.model, "hyper_connection_mixer", None))
    return reasons


def mtp_views(lm, copies: dict, params: set) -> dict:
    """The MTP draft's packed views (``packing.pack_mtp``)."""
    layer = lm.model.layers[0]
    lw = {f"layers.0.{k}": v for k, v in checkpoint_dict(layer, "qsa", False).items()}
    lw.update({name: _get(lm, name) for name in _MTP_FUSION})
    mixer = lm.model.hyper_connection_mixer
    for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up"):
        lw[f"hyper_connection_mixer.{name}.weight"] = _get(mixer, f"{name}.weight")
    views = packing.pack_mtp(
        lw, moe=moe_params(layer.mlp.experts), proj=proj_params(layer, QSA_PROJ)
    )
    _log_copies(copies, "mtp packed", (), views, params)
    return views
