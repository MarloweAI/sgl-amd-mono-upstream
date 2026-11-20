# SPDX-License-Identifier: Apache-2.0
"""One rank's host state for the Kimi-K3 launches: the scratch every launch
shares, the MoE task queue, the step epoch and the TP group's peer memory.

Mailbox tags carry the epoch the model bumps once a forward (``step_begin``,
which a CUDA graph captures), so no buffer is zeroed between steps. A launch's
own slot distinguishes it from the step's other launches.
"""

from __future__ import annotations

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.kimi_k3 import K2_MAX_ROWS


class K3MonoRunner:
    """The device buffers every Kimi-K3 launch on this rank shares.

    Built once, outside any graph capture: the peer handshake allocates and
    runs a collective, and FlyDSL traces a kernel on its first launch.
    """

    def __init__(self, tp: int, rank: int, group, device) -> None:
        from sgl_amd_mono.kimi_k3 import MAX_BLOCKS
        from sgl_amd_mono.kimi_k3.gfx950 import layer
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_combine, mla_front
        from sgl_amd_mono.kimi_k3.gfx950.stages import moe

        self.tp, self.rank, self.device = tp, rank, device
        # tests only: the wait to stall and how many launches to let
        # through first, compiled into this rank's build key
        self.fault_site, self.fault_after = guard.fault_injection(rank) or (0, 0)
        # The launches run one after another, so one buffer the widest of them
        # fits, then K2's tagged x rows.
        front = mla_front.scratch_bytes(
            mla_front.MlaFrontBuild(tokens=mla_front.G_ROWS, nblocks=MAX_BLOCKS, scale=1.0)
        )
        self.scratch_bytes = (max(layer.scratch_alloc_bytes(), front) + 255) // 256 * 256
        # this rank's decode-context-parallel rank (the DCP front's key share)
        self.dcp_rank = 0
        # The hand-off mailboxes uncached, as GLM-5.3-Flash's: a consumer reads a
        # line at its home memory, not through the producer XCD's L2. A one-rank
        # peer buffer, never freed while the process lives (see PeerBuffer).
        self._scratch = PeerBuffer(self.scratch_bytes, group, rank, 1, device)
        self.scratch = self._scratch.bytes
        self.scratch.zero_()
        # [the step epoch, the guard block (deadline, error words)]
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        # The MoE launch's task queue. The layer launch takes none -- its ABI
        # has no queue -- and this is here for the kernel checks' MoE launch.
        self.queue = torch.zeros(moe.QUEUE_BYTES // 4, dtype=torch.int32, device=device)
        # past the two parity halves, the guard's flag word a rank, then K2's
        # sentinel regions, then the MLA front's all-to-all cells. The kernels
        # read them at the same offsets, from the same functions.
        self.peer = PeerBuffer(mla_combine.peer_end(), group, rank, tp, device)
        layer.init_peer(self.peer.bytes)
        mla_combine.init_peer(self.peer.bytes)
        # One prefix buffer, not one a layer: the layers run in sequence.
        self.prefix = torch.empty(
            K2_MAX_ROWS, layer.HIDDEN, dtype=torch.bfloat16, device=device
        )

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline, below the server's watchdog timeout."""
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def step_begin(self) -> None:
        """Once a forward, before its first launch (a graph captures it).

        Word 0 only: the rest of the buffer is the guard block.
        """
        self.epoch[0].add_(1)

    def launch_args(self) -> dict:
        """The state every launch takes."""
        return {
            "scratch": self.scratch,
            "peers": self.peer.addresses,
            "rank": self.rank,
            "epoch": self.epoch,
        }

    def close(self) -> None:
        self.peer.close()
        if self._scratch is not None:
            self._scratch.close()
