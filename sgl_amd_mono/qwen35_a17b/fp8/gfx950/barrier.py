# SPDX-License-Identifier: Apache-2.0
"""The two guarded peer barriers that open each decode or target-verify step.

Each phase publishes its own tag, then polls that tag on every rank. Phases 0
and 1 use different pair slots and run once each per step, in that order, so a
rank that finishes a phase cannot overwrite the tag the slower ranks are still
polling. A slot must not be written twice without the other phase in between:
the second write would need no peer to have read the first. A fault at site 1
stalls inside the phase 0 poll, after its tag is published and before the step
runs any layer; the other ranks then block in phase 1 until the deadline flags
them and each rank stops with a store to address 0x0.
"""

from __future__ import annotations

import inspect
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, load_ptr64, rsrc
from sgl_amd_mono.common.plan import THREADS, kernel_sources, key_tuple
from sgl_amd_mono.common.sync import Mailbox, preg, publish
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.layout import PHASE_BYTES, PHASE_FLAGS, TP

SOURCES = kernel_sources(__name__)
# The barrier does no work past the handshake, so one CTA is enough. Every extra
# CTA only adds threads that poll the same uncached peer tags and slow the wait.
GRID = int(os.environ.get("SGLANG_AMD_MONO_BARRIER_GRID", 1))


def _ensure_cache() -> None:
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = os.path.expanduser("~/.flydsl/cache")
    os.makedirs(path, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = path


def compile_only(launch) -> None:
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


def build_barrier(kind: int, rank: int, fault, tp: int = TP):
    """A launch for the step's phase ``kind`` (0 first, 1 second) on this rank."""
    fault_site = -1 if fault is None else int(fault[0])
    fault_after = 0 if fault is None else int(fault[1])
    # scalars only: FlyDSL drops a tuple closed over by the launch from its cache key
    keyed = key_tuple(
        {
            "kind": kind,
            "rank": rank,
            "fault_site": -1 if fault is None else int(fault[0]),
            "fault_after": 0 if fault is None else int(fault[1]),
            "grid": GRID,
            "tp": tp,
        },
        SOURCES,
    )
    _ensure_cache()

    @flyc.kernel(
        name=f"qwen35_fp8_phase_{kind}_r{rank}_tp{tp}", known_block_size=[THREADS, 1, 1]
    )
    def phase_kernel(peers: Int64, epoch: Int64, sym: Int64):
        _ = keyed
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=PHASE_FLAGS,
                fault_site=None if fault_site < 0 else fault_site,
                fault_after=fault_after,
            )
        ):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            ep = fx.Int32(
                bo.buffer_load(
                    rsrc(epoch), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
                )
            )
            # Phase 0 owns pair `rank`; phase 1 owns pair `rank + TP`. Tags are
            # 2*epoch and 2*epoch+1. Publish before the poll so the first step
            # needs no seed, and so one rank cannot replace a tag others still
            # need: the next writer of a slot has already seen every peer finish
            # the previous read of it.
            mine = ep * fx.Int32(2) + fx.Int32(kind)
            slot = rank if kind == 0 else rank + tp
            publish(
                Mailbox(mine).put,
                preg(sym, 0, "self"),
                slot,
                fx.Int32(0),
                (bid == 0) & (tid == 0),
            )
            Mailbox(mine).poll(
                [
                    (preg(load_ptr64(peers, p), 0, f"r{p}"), slot - rank + p, 1)
                    for p in range(tp)
                ]
            )

    @flyc.jit
    def launch(
        peers: Int64,
        epoch: Int64,
        sym: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        phase_kernel(peers, epoch, sym).launch(
            grid=(GRID,), block=(THREADS,), stream=stream
        )

    return launch


class PhaseRuntime:
    """The two phase barriers, the peer buffer they publish into, and the epoch."""

    def __init__(
        self, rank: int, group, device: torch.device, deadline_s: float, tp: int = TP
    ):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank = rank
        fault = guard.fault_injection(rank)
        self.tp = tp
        self.launch = {
            0: build_barrier(0, rank, fault, tp),
            1: build_barrier(1, rank, fault, tp),
        }
        self.peer = PeerBuffer(PHASE_BYTES, group, rank, tp, device)
        self.peer.bytes.zero_()
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        self.set_deadline(deadline_s)
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=group)
        for launch in self.launch.values():
            compile_only(launch)

    def set_deadline(self, seconds: float) -> None:
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def advance(self) -> None:
        self.epoch[:1].add_(1)

    def enter(self, kind: int) -> None:
        self.launch[kind](
            self.peer.addresses.data_ptr(),
            self.epoch.data_ptr(),
            self.peer.local,
            stream=torch.cuda.current_stream(),
        )

    def close(self) -> None:
        self.peer.close()
