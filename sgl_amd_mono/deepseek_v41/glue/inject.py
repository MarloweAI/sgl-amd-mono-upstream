# SPDX-License-Identifier: Apache-2.0
"""A verify's committed KV into every DSpark draft stage's SWA cache: after main_proj, the
main norm, the MXFP8 quant, the stacked wkv GEMM and every stage's store in one launch
(``gfx950/inject/kv``) where aiter runs the stacked wkv as 5 fused splits of K 1024 (5 to 8
rows); otherwise SGLang's projection, then the stages' stores in one launch (``gfx950/inject``).
The slots are computed in the launch. Bitwise SGLang's chain.
``SGLANG_AMD_MONO_DSV41_INJECT=0`` keeps SGLang's."""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

from sgl_amd_mono.deepseek_v41.glue import chose, flag, route

ENV = "SGLANG_AMD_MONO_DSV41_INJECT"
DSPARK = "sglang.srt.models.deepseek_v4_dspark"
DRAFT_MODEL = "sglang.kernels.ops.speculative.dspark.dspark_draft_model"
INJECTOR = "sglang.srt.speculative.dspark_components.dspark_kv_inject"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4_dspark.py": (
        "DeepseekV4ForCausalLMDSpark",
        "main_proj",
        "main_norm",
        "write_target_hidden_kv(main_hidden)",
        "write_target_hidden_kv(swa_loc)",
        "write_target_hidden_kv(positions)",
        "write_target_hidden_kv(pool)",
        "project_target_hidden",
        "CommitKvProj",
        "is_unified_kv_triton",
        "stages",
    ),
    "srt/models/deepseek_v4.py": (
        "self_attn",
        "wkv",
        "kv_norm",
        "layer_id",
        "eps",
        "freqs_cis",
    ),
    "kernels/ops/speculative/dspark/dspark_draft_model.py": (
        "execute(main_x)",
        "execute(wkv_linears)",
        "execute(allow_strided_output)",
        "_stacked_wkv_weight",
        "mxfp8_scale",
        '"mxfp8_dense_backend"',
        '"mxfp8_aiter_preshuffled"',
    ),
    "srt/layers/quantization/fp8_utils.py": ("is_gfx95_mxfp8_aiter",),
    "srt/layers/quantization/fp8_hip.py": ("mxfp8_aiter_preshuffled",),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "kernels/ops/speculative/dspark/commit_swa.py": ("committed_swa_locations",),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "uniform_fp8",
        "kv_layout",
        "swa_page_size",
        "get_swa_raw_buffer",
        "swa_req_ring_size",
        "full_to_swa_index_mapping",
    ),
    "srt/speculative/dspark_components/dspark_kv_inject.py": (
        "TargetHiddenKvInjector",
        "_inject_mla",
        "draft_model",
    ),
}


class _Committed(NamedTuple):
    """A verify's committed rows, ``committed_swa_locations``'s inputs."""

    cache_loc: torch.Tensor
    mapping: torch.Tensor
    commit: torch.Tensor
    width: int

    def slots(self):
        from sglang.kernels.ops.speculative.dspark.commit_swa import (
            committed_swa_locations,
        )

        return committed_swa_locations(*self)


class _Stages:
    def __init__(self, model, pool):
        attns = [s.self_attn for s in model.stages]
        self.wkv = [a.wkv for a in attns]
        self.weights = [a.kv_norm.weight.data for a in attns]
        self.freqs = [
            torch.view_as_real(a.freqs_cis).flatten(-2).contiguous() for a in attns
        ]
        self.layers = [a.layer_id for a in attns]
        self.eps = attns[0].eps
        self.v41 = pool.kv_layout.value == "v41"
        self.why = None
        if not 0 < len(attns) <= 4:
            self.why = f"{len(attns)} stages"
        elif sys.modules[DSPARK].is_unified_kv_triton():
            self.why = "unified KV"
        elif pool.uniform_fp8 or pool.kv_layout.value not in ("v4", "v41"):
            self.why = f"a {pool.kv_layout.value} pool"
        elif any(a.eps != self.eps for a in attns):
            self.why = "the stages' eps differ"
        elif any(w.dtype != torch.bfloat16 or w.numel() != 512 for w in self.weights):
            self.why = "kv_norm weights not [512] bf16"
        elif any(f.dim() != 2 or f.shape[1] != 64 for f in self.freqs):
            self.why = "rope tables not [positions, 64]"
        # the one launch: aiter's MXFP8 route of the stacked wkv, plain rows, SGLang's RMSNorm
        self.fused = None
        if self.why is None:
            self.fused = _fused_why(model, self.wkv)
            if self.fused is None:
                from sgl_amd_mono.deepseek_v41.gfx950.inject import kv

                stacked = sys.modules[DRAFT_MODEL]._stacked_wkv_weight(
                    wkv_linears=self.wkv
                )
                stage0 = model.stages[0]
                device = stacked.weight.device
                self.w = stacked.weight.view(torch.uint8)
                self.ws = stacked.mxfp8_scale.view(torch.uint8)
                self.norm_w, self.norm_eps = (
                    stage0.main_norm.weight,
                    stage0.main_norm.variance_epsilon,
                )
                self.scratch = kv.buffers(device, len(self.wkv))


def _fused_why(model, wkv) -> str | None:
    """Why the stacked wkv does not take the one launch (None: it does)."""
    from sgl_amd_mono.deepseek_v41.gfx950.inject import kv

    if len(wkv) > kv.MAX_STAGES:
        return f"{len(wkv)} stages"
    method = getattr(wkv[0], "quant_method", None)
    backend = getattr(method, "mxfp8_dense_backend", None)
    if backend is None or not backend.is_gfx95_mxfp8_aiter():
        return "not aiter's MXFP8 route"
    if getattr(wkv[0], "mxfp8_aiter_preshuffled", False):
        return "preshuffled wkv"
    stacked = sys.modules[DRAFT_MODEL]._stacked_wkv_weight(wkv_linears=wkv)
    if stacked.mxfp8_scale is None:
        return "no MXFP8 scales"
    if (
        tuple(stacked.weight.shape) != (512 * len(wkv), 5120)
        or stacked.weight.element_size() != 1
    ):
        return f"stacked wkv {tuple(stacked.weight.shape)}"
    if tuple(stacked.mxfp8_scale.shape) != (16 * len(wkv), 160):
        return f"wkv scales {tuple(stacked.mxfp8_scale.shape)}, not 32 x 32 blocks"
    norm = getattr(model.stages[0], "main_norm", None)
    w = getattr(norm, "weight", None)
    if (
        w is None
        or w.dtype != torch.bfloat16
        or w.numel() != 5120
        or not hasattr(norm, "variance_epsilon")
    ):
        return "main_norm not a [5120] bf16 RMSNorm"
    return None


def _store(st, kvs, pos, caches, page_size, loc=None, committed=None):
    from sgl_amd_mono.deepseek_v41.gfx950.inject import ops

    step = kvs[0].element_size()
    stride = (kvs[1].data_ptr() - kvs[0].data_ptr()) // step if len(kvs) > 1 else 0
    if all(
        kv.stride() == kvs[0].stride()
        and kv.data_ptr() == kvs[0].data_ptr() + s * stride * step
        for s, kv in enumerate(kvs)
    ):  # the stages' rows a stride apart in one tensor: one launch
        slots = {} if committed is None else committed._asdict()
        args = (st.weights, st.freqs, pos, loc, caches, st.eps, page_size, st.v41)
        ops().swa_store(kvs[0], stride, *args, **slots)
        return
    loc = committed.slots() if committed is not None else loc
    for s, kv in enumerate(kvs):
        one = (st.weights[s : s + 1], st.freqs[s : s + 1], pos, loc, caches[s : s + 1])
        ops().swa_store(kv, 0, *one, st.eps, page_size, st.v41)


def _stages(model, pool):
    st = model.__dict__.get("_mono_inject")
    if st is None:
        st = model.__dict__["_mono_inject"] = _Stages(model, pool)
        chose("inject_store", st.why is None, st.why)
    return st if st.why is None else None


def _write_target_hidden_kv(
    orig, self, *, main_hidden, swa_loc, positions, pool, **kwargs
):
    committed = swa_loc if isinstance(swa_loc, _Committed) else None
    st = None if kwargs else _stages(self, pool)
    if st is None:
        if committed is not None:
            swa_loc = committed.slots()
        return orig(
            self,
            main_hidden=main_hidden,
            swa_loc=swa_loc,
            positions=positions,
            pool=pool,
            **kwargs,
        )
    from sgl_amd_mono.deepseek_v41.gfx950.inject import kv

    caches = [pool.get_swa_raw_buffer(layer) for layer in st.layers]
    loc = None if committed is not None else swa_loc.to(torch.int32).contiguous()
    rows = main_hidden.shape[0]
    why = st.fused or (None if kv.MIN_ROWS <= rows <= kv.MAX_ROWS else f"{rows} rows")
    if chose("inject_kv", why is None, why):
        projected, _ = self.stages[0].main_proj(main_hidden)
        slots = {} if committed is None else committed._asdict()
        kv.ops().inject_kv(
            projected, st.norm_w, st.norm_eps, st.w, st.ws, *st.scratch, st.weights,
            st.freqs, positions.contiguous(), loc, caches, st.eps, pool.swa_page_size,
            st.v41, **slots,
        )  # fmt: skip
        return
    kvs = sys.modules[DSPARK].CommitKvProj.execute(
        main_x=self.project_target_hidden(main_hidden),
        wkv_linears=st.wkv,
        allow_strided_output=True,
    )
    _store(st, kvs, positions.contiguous(), caches, pool.swa_page_size, loc, committed)


def _inject_mla(orig, self, *args, **kwargs):
    """The committed rows' slots in the store's launch, where SGLang makes them with
    ``committed_swa_locations``."""
    pool, cache_loc = kwargs.get("pool"), kwargs.get("cache_loc")
    loc2d, commit = kwargs.get("cache_loc_2d"), kwargs.get("commit_lens")
    mapping = getattr(pool, "full_to_swa_index_mapping", None)
    if (
        args
        or kwargs.get("swa_loc") is not None
        or getattr(pool, "swa_req_ring_size", None) is not None
        or cache_loc is None
        or loc2d is None
        or commit is None
        or mapping is None
        or not cache_loc.is_cuda
        or not cache_loc.is_contiguous()
        or cache_loc.dtype != torch.int64
        or not commit.is_contiguous()
        or commit.dtype != torch.int32
        or mapping.dtype != torch.int64
        or not mapping.is_contiguous()
        or cache_loc.numel() != loc2d.numel()
    ):
        return orig(self, *args, **kwargs)
    with torch.inference_mode():
        self.draft_model.write_target_hidden_kv(
            main_hidden=kwargs["target_hidden"],
            swa_loc=_Committed(cache_loc, mapping, commit, loc2d.shape[1]),
            positions=kwargs["positions"],
            pool=pool,
        )


def install() -> None:
    if not flag(ENV) or torch.version.hip is None:
        return
    import importlib

    dspark = importlib.import_module(DSPARK).DeepseekV4ForCausalLMDSpark
    route(dspark, "write_target_hidden_kv", _write_target_hidden_kv)
    injector = importlib.import_module(INJECTOR).TargetHiddenKvInjector
    route(injector, "_inject_mla", _inject_mla)
