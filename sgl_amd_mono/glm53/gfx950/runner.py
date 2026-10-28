# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from ROCm/aiter#6173 at 56d6390def (Apache-2.0):
# aiter/ops/flydsl/kernels/glm5_mono/glm/op.py
"""Host side of the GLM-5.3 launches: one runtime per (width, kind), shared by
every covered layer of that kind in a step, and one op per layer holding its
weights. The layer kinds are the MoE layers that run the indexer and those that
reuse the previous top-k, and the dense layers (from the step's tokens for the
first); the tail (final norm, LM head, logits) is one more runtime and op."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Optional

import msgspec
import torch

from .config import GLM5_GLOBAL_HEADS, GLM5_GLOBAL_INTER, HIDDEN, KV_LORA, PE_DIM
from .layout import INDEX_DIM, layout

TOPK = 2048
# SGLang's DSA index cache: 64-token pages of FP8 keys (16 x 16 tiles), then one
# f32 scale per token
INDEX_PAGE_TOKENS = 64
INDEX_PAGE_BYTES = INDEX_PAGE_TOKENS * (INDEX_DIM + 4)
KV_ROW = KV_LORA + PE_DIM


class Glm53Build(msgspec.Struct, frozen=True):
    """One compiled launch: the row width, TP size, the kind's layers per step,
    whether it runs the indexer (scoring up to ``index_max_seq`` keys a row),
    whether SGLang's FP8 weights are in AITER's (16, 16) layout, whether the MLA
    cache holds E4M3 (``kv_fp8``), and the guard's test fault injection. ``dense``:
    a dense MLP of that many expert-shaped slices instead of the MoE; ``tokens``:
    the layer starts from the step's tokens and ``vocab_rows`` embedding rows a
    rank; ``head``: the tail, on ``vocab_rows`` LM-head rows a rank."""

    rows: int
    tp: int
    launches_per_step: int
    with_indexer: bool
    index_max_seq: int
    fp8_shuffled: bool
    kv_fp8: bool = False
    fault_site: int = 0
    fault_after: int = 0
    stop: bool = True
    dense: int = 0
    tokens: bool = False
    head: bool = False
    vocab_rows: int = 0

    @property
    def heads(self) -> int:
        return GLM5_GLOBAL_HEADS // self.tp

    @property
    def inter(self) -> int:
        return GLM5_GLOBAL_INTER // self.tp


class Glm53Weights(msgspec.Struct, frozen=True):
    """One layer's tensors as the kernel reads them. Every FP8 weight carries f32
    block scales, 128 x 128 (W_UK's 128 x 64). q_a | kv_a, q_b, o_proj, the indexer's wq_b and wk and
    the experts are SGLang's tensors (``Glm53Build.fp8_shuffled``); W_UK, W_UV,
    the router and the indexer's weights_proj are packed copies, the k LayerNorm
    weight and bias f32 copies. ``index``: an indexer layer's pointer table.
    ``dense``: a dense layer's MLP as that many expert-shaped slices (copies) in the
    expert tensors, its router tensors unread; a tokens layer's ``w_r`` is this
    rank's embedding rows."""

    g_in: torch.Tensor
    g_q: torch.Tensor
    g_kv: torch.Tensor
    g_post: torch.Tensor
    w_qkv_a: torch.Tensor
    s_qkv_a: torch.Tensor
    w_q_b: torch.Tensor
    s_q_b: torch.Tensor
    w_uk: torch.Tensor
    s_uk: torch.Tensor
    w_uv: torch.Tensor
    s_uv: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    w_r: torch.Tensor
    bias: torch.Tensor
    w_ug: torch.Tensor
    s_ug: torch.Tensor
    w_dn: torch.Tensor
    s_dn: torch.Tensor
    shuffled: bool
    index: Optional[torch.Tensor] = None
    keep: tuple = ()
    dense: int = 0


class Glm53Caches(msgspec.Struct, frozen=True):
    """One layer's SGLang caches: the MLA cache [slots, 576] (latent, then rope),
    bf16 or, as SGLang's HIP DSA backends keep it, E4M3 without scales, and, on a
    layer that runs the indexer, the uint8 index cache [pages, 64, 132]."""

    kv: torch.Tensor
    idx: Optional[torch.Tensor] = None


class Glm53Step(msgspec.Struct, frozen=True):
    """A step's rows for every launch of one width (the spec's ``_Built._step`` fills them):
    int64 positions and MLA slots (-1: a pad row), int32 ``indptr`` [W + 1] (row
    s attends keys indptr[s] ..< indptr[s + 1]), int32 request rows of
    ``page_table`` (SGLang's int32 64-token page table), and the int32 top-k
    buffer the indexer layers write and the others read."""

    positions: torch.Tensor
    slots: torch.Tensor
    indptr: torch.Tensor
    req_ids: torch.Tensor
    page_table: torch.Tensor
    topk: torch.Tensor


def compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    from flydsl.utils.env import CompileEnvManager

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    var = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(var)
    os.environ[var] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


class Glm53Runtime:
    """The compiled launch for one (width, kind) plus the state its layers share:
    scratch mailboxes, the symmetric peer buffer, and the guard block, whose word 0
    is the step epoch (``advance_step``)."""

    def __init__(self, key: Glm53Build, rank: int, group, device: torch.device):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer

        from .kernel import build_glm5_monokernel

        ensure_writable_flydsl_cache()
        self.key, self.width, self.rank = key, key.rows, rank
        self.kv_dtype = torch.float8_e4m3fn if key.kv_fp8 else torch.bfloat16
        self.sc, self.sy = layout(
            key.rows,
            key.heads,
            key.tp,
            TOPK,
            key.with_indexer,
            key.index_max_seq,
            inter=key.inter,
            output_heads=key.heads,
            tokens=key.tokens,
            head_vocab=key.vocab_rows * key.tp if key.head else 0,
        )
        self.launch = build_glm5_monokernel(
            key.rows,
            key.heads,
            key.tp,
            TOPK,
            launches_per_step=key.launches_per_step,
            with_indexer=key.with_indexer,
            index_max_seq=key.index_max_seq,
            attention_weight="fp8_block128",
            kv_cache_layout="atom",
            kv_cache_dtype="fp8" if key.kv_fp8 else "bf16",
            inter=key.inter,
            output_heads=key.heads,
            uv_scale_rows=128,
            # the other layers read the top-k buffer, not the index cache
            index_paged=key.with_indexer,
            index_block_size=INDEX_PAGE_TOKENS,
            index_block_bytes=INDEX_PAGE_BYTES,
            index_shuffled=key.with_indexer,
            fp8_shuffled=key.fp8_shuffled,
            fault_site=key.fault_site,
            fault_after=key.fault_after,
            stop=key.stop,
            dense_experts=key.dense,
            tokens=key.tokens,
            vocab_rows=key.vocab_rows,
            tail=key.head,
        )
        self.peer = PeerBuffer(self.sy["_bytes"], group, rank, key.tp, device)
        self.peer.bytes.zero_()
        # the hand-off mailboxes, uncached as the peer buffer is: a line is read at its
        # home memory, not through the producer's L2. The tail has none, and its
        # buffer still needs an address (an uncached allocation of 0 bytes has none)
        self._scratch = PeerBuffer(max(self.sc["_bytes"], 256), group, rank, 1, device)
        self.scratch = self._scratch.bytes
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        # two outputs, by layer parity: a layer never writes what it reads; the
        # tail's: the gathered f32 logits and the final norm's rows
        if key.head:
            vocab = key.vocab_rows * key.tp
            self.outs = [
                torch.empty(key.rows, vocab, dtype=torch.float32, device=device),
                torch.empty(key.rows, HIDDEN, dtype=torch.bfloat16, device=device),
            ]
        else:
            self.outs = [
                torch.empty(key.rows, HIDDEN, dtype=torch.bfloat16, device=device)
                for _ in range(2)
            ]

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline (below the server's watchdog timeout)."""
        from sgl_amd_mono.common import guard

        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def precompile(self) -> None:
        compile_only(self.launch)

    def advance_step(self) -> None:
        self.epoch[:1].add_(1)

    def close(self) -> None:
        self.scratch = None
        self._scratch.close()
        self.peer.close()


def _check(name: str, t: torch.Tensor, shape: tuple, dtype: torch.dtype) -> None:
    if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"GLM-5.3 monokernel needs {name} as contiguous {dtype} {shape}, "
            f"got {t.dtype} {tuple(t.shape)}"
        )


class Glm53LayerOp:
    """One covered layer: its weights and its ordinal among its kind's layers."""

    def __init__(self, runtime: Glm53Runtime, weights: Glm53Weights, layer: int):
        key = runtime.key
        if (weights.index is not None) != key.with_indexer:
            raise ValueError("an indexer runtime needs indexer weights, and only it")
        if weights.dense != key.dense:
            raise ValueError(
                f"weights of {weights.dense} dense slices on a runtime of {key.dense}"
            )
        if weights.shuffled != key.fp8_shuffled:
            raise ValueError(
                f"weights with shuffled={weights.shuffled} on a runtime with "
                f"fp8_shuffled={key.fp8_shuffled}"
            )
        self.rt, self.w, self.layer = runtime, weights, layer

    def launch(
        self,
        h: torch.Tensor,
        caches: Glm53Caches,
        cos: torch.Tensor,
        sin: torch.Tensor,
        step: Glm53Step,
    ) -> torch.Tensor:
        """The previous layer's hidden stream ``h`` [W, HIDDEN] in (a tokens layer:
        the step's token ids, int32 [W], -1 a pad row); this layer's out (h plus its
        attention and MLP)."""
        rt, w, W = self.rt, self.w, self.rt.width
        if rt.key.tokens:
            _check("token ids", h, (W,), torch.int32)
        else:
            _check("h", h, (W, HIDDEN), torch.bfloat16)
        _check("positions", step.positions, (W,), torch.int64)
        _check("slots", step.slots, (W,), torch.int64)
        _check("indptr", step.indptr, (W + 1,), torch.int32)
        _check("req_ids", step.req_ids, (W,), torch.int32)
        if step.topk.dtype != torch.int32 or step.topk.numel() < W * TOPK:
            raise ValueError("needs an int32 top-k buffer of W x 2048 entries")
        pt = step.page_table
        if pt.dtype != torch.int32 or pt.dim() != 2 or pt.stride(1) != 1:
            raise ValueError("needs an int32 page table with contiguous rows")
        kv, idx = caches.kv, caches.idx
        if kv.dtype != rt.kv_dtype or kv.dim() != 2 or kv.stride() != (KV_ROW, 1):
            raise ValueError(f"needs the MLA cache as {rt.kv_dtype} [slots, {KV_ROW}]")
        if rt.key.with_indexer and (
            idx is None
            or idx.dtype != torch.uint8
            or tuple(idx.shape[1:]) != (INDEX_PAGE_TOKENS, INDEX_DIM + 4)
            or not idx.is_contiguous()
        ):
            raise ValueError(
                f"needs the index cache as uint8 [pages, {INDEX_PAGE_TOKENS}, "
                f"{INDEX_DIM + 4}]"
            )
        x_out = rt.outs[self.layer % 2]
        p = lambda t: t.data_ptr()  # noqa: E731
        rt.launch(
            p(h),
            p(x_out),
            p(step.positions),
            p(step.positions),
            p(step.slots),
            p(step.indptr),
            p(kv),
            p(kv),
            p(idx) if rt.key.with_indexer else p(step.topk),
            p(cos),
            p(sin),
            p(w.g_in),
            p(w.g_q),
            p(w.g_kv),
            p(w.g_post),
            p(w.w_qkv_a),
            p(w.s_qkv_a),
            p(w.w_q_b),
            p(w.s_q_b),
            p(w.w_uk),
            p(w.s_uk),
            p(w.w_uv),
            p(w.s_uv),
            p(w.w_o),
            p(w.s_o),
            p(w.w_r),
            p(w.bias),
            p(w.w_ug),
            p(w.s_ug),
            p(w.w_dn),
            p(w.s_dn),
            p(rt.scratch),
            rt.peer.local,
            p(rt.peer.addresses),
            p(w.index) if w.index is not None else 0,
            p(rt.epoch),
            p(pt),
            p(step.req_ids),
            p(step.topk),
            rt.rank,
            self.layer,
            pt.stride(0),
            stream=torch.cuda.current_stream(),
        )
        return x_out


class Glm53HeadWeights(msgspec.Struct, frozen=True):
    """The tail's tensors, SGLang's in place: the final norm's weight and this
    rank's rows of the LM head (bf16 [vocab_rows, HIDDEN])."""

    g_norm: torch.Tensor
    w_head: torch.Tensor


class Glm53HeadOp:
    """The tail on its runtime (``Glm53Build.head``)."""

    def __init__(self, runtime: Glm53Runtime, weights: Glm53HeadWeights):
        key = runtime.key
        if not key.head:
            raise ValueError("a tail op needs a head runtime")
        _check("final norm", weights.g_norm, (HIDDEN,), torch.bfloat16)
        _check("LM head", weights.w_head, (key.vocab_rows, HIDDEN), torch.bfloat16)
        self.rt, self.w = runtime, weights

    def launch(self, x: torch.Tensor):
        """The last layer's rows ``x`` [W, HIDDEN] in; the gathered logits (f32 [W,
        vocab]) and the final norm's rows (bf16 [W, HIDDEN]) out."""
        rt, w, W = self.rt, self.w, self.rt.width
        _check("x", x, (W, HIDDEN), torch.bfloat16)
        logits, normed = rt.outs
        p = lambda t: t.data_ptr()  # noqa: E731
        z = p(x)  # the arguments the tail does not read
        rt.launch(
            p(x),
            p(logits),
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            p(w.g_norm),
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            p(normed),
            z,
            p(w.w_head),
            z,
            z,
            z,
            z,
            z,
            p(rt.scratch),
            rt.peer.local,
            p(rt.peer.addresses),
            0,
            p(rt.epoch),
            z,
            z,
            z,
            rt.rank,
            0,
            0,
            stream=torch.cuda.current_stream(),
        )
        return logits, normed
