# SPDX-License-Identifier: Apache-2.0
"""A verify's committed KV into every DSpark draft stage's SWA cache in one launch
(``gfx950/inject``), its slots computed in the launch; bitwise SGLang's per-stage store,
checked on random rows at the first call. ``SGLANG_AMD_MONO_DSV41_INJECT=0`` keeps SGLang's."""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

from sgl_amd_mono.deepseek_v41.glue import agreed, flag, route

ENV = "SGLANG_AMD_MONO_DSV41_INJECT"
DSPARK = "sglang.srt.models.deepseek_v4_dspark"
INJECTOR = "sglang.srt.speculative.dspark_components.dspark_kv_inject"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4_dspark.py": (
        "DeepseekV4ForCausalLMDSpark",
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
    ),
    "kernels/ops/speculative/dspark/commit_swa.py": ("committed_swa_locations",),
    "kernels/ops/attention/dsv4/norm_rope.py": ("fused_k_norm_rope_flashmla",),
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
        self.freqs_cis = [a.freqs_cis for a in attns]
        self.freqs = [
            torch.view_as_real(f).flatten(-2).contiguous() for f in self.freqs_cis
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


def _disagrees(st, pool) -> str | None:
    """SGLang's per-stage store vs the one launch, on random rows into scratch pages."""
    from sglang.kernels.ops.attention.dsv4.norm_rope import fused_k_norm_rope_flashmla

    dev = st.weights[0].device
    g = torch.Generator(device=dev).manual_seed(11)
    rows, width = 12, 6
    page_bytes = pool.get_swa_raw_buffer(st.layers[0]).stride(0)
    pages, ps = 3, pool.swa_page_size
    n = len(st.layers)
    kv = torch.randn(rows, n * 512, generator=g, device=dev).mul_(3).bfloat16()
    kvs = [kv[:, s * 512 : (s + 1) * 512] for s in range(n)]
    pos = torch.randint(
        0, min(f.shape[0] for f in st.freqs), (rows,), generator=g, device=dev
    )
    full = torch.randperm(pages * ps, generator=g, device=dev)[:rows]
    mapping = torch.arange(pages * ps, device=dev, dtype=torch.int64)
    commit = torch.tensor([4, 6], dtype=torch.int32, device=dev)
    slots = _Committed(full, mapping, commit, width).slots()
    want = [
        torch.zeros(pages, page_bytes, dtype=torch.uint8, device=dev) for _ in range(n)
    ]
    got = [torch.zeros_like(w) for w in want]
    for s in range(n):
        fused_k_norm_rope_flashmla(
            kv=kvs[s].contiguous(),
            kv_weight=st.weights[s],
            eps=st.eps,
            freqs_cis=st.freqs_cis[s],
            positions=pos,
            out_loc=slots,
            kvcache=want[s],
            page_size=ps,
            layout=pool.kv_layout,
        )
    _store(st, kvs, pos, got, ps, committed=_Committed(full, mapping, commit, width))
    for s in range(n):
        if not torch.equal(want[s], got[s]):
            return f"stage {s}: {(want[s] != got[s]).sum().item()} bytes differ"
    return None


def _stages(model, pool):
    st = model.__dict__.get("_mono_inject")
    if st is None:
        st = model.__dict__["_mono_inject"] = _Stages(model, pool)
    if st.why is not None:
        return None
    ok = agreed("inject_store", lambda: _disagrees(st, pool))
    return st if ok else None


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
    kvs = sys.modules[DSPARK].CommitKvProj.execute(
        main_x=self.project_target_hidden(main_hidden),
        wkv_linears=st.wkv,
        allow_strided_output=True,
    )
    caches = [pool.get_swa_raw_buffer(layer) for layer in st.layers]
    loc = None if committed is not None else swa_loc.to(torch.int32).contiguous()
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
