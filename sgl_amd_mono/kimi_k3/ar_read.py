# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's all-reduces with the glue after them (TP8, gfx950), for the widths SGLang's own layers
run (``ROWS``):

* ``ArRead``: o_proj's all-reduce and the FFN read of the attention-residual bank in one launch a
  layer, in place of the all-reduce and ``attn_res_hip._agg_kernel`` (``gfx950/stages/ar_read.py``).
  SGLang's K3 decoder layer offers its FFN read to it (a ReadoutFusion that also reads).
* ``MoeAr``: the latent MoE's flat [latent | shared] all-reduce with the latent's RMSNorm in one
  launch, reading the partials where the MoE wrote them (``gfx950/stages/ar_norm.py``).
* ``SharedPushAr``: the same reduce with the shared half pushed one-shot to every rank by the launch that
  computes it (its tile epilogue), so only the latent half's all-reduce follows the routed experts; its
  launch sums the pushed shared partials in rank order (``gfx950/stages/shared_push.py``).

A width outside ``ROWS`` keeps SGLang's all-reduce and glue.
"""

import torch

from sgl_amd_mono.kimi_k3.gfx950.route import SHARED_PUSH_ROWS
from sgl_amd_mono.kimi_k3.gfx950.stages import ar_read as K

# Above the row caps of the KDA and MLA layer launches, up to the widest graphed verify.
ROWS = range(49, K.MAX_ROWS + 1)
# SharedPushAr's: the rows the fused shared chain pushes the shared partial at (its one definition).
PUSH_ROWS = SHARED_PUSH_ROWS


def _p(t) -> int:
    return t.data_ptr()


class ArRead:
    """One rank's fused launches: the peer buffer every layer's launch shares, the step epoch
    (bumped by the first layer's call, in the graph) and the launch.

    ``layers``: the decoder layers' indices (each its own launch slot); ``eps``: the score and
    output RMSNorm epsilons. Built by every rank in the same
    order, outside graph capture (the peer handshake runs on ``cpu_group``)."""

    def __init__(self, cpu_group, rank: int, device, layers, eps, deadline_s: float = 20.0):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer

        dev = torch.device(device)
        self.rank = rank
        self.first = min(layers)
        self.slots = max(layers) + 1
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        ticks = guard.deadline_ticks(deadline_s)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self.peer = PeerBuffer(K.peer_bytes(), cpu_group, rank, K.TP, dev, init=K.arm)
        self.launch = K.build(float(eps[0]), float(eps[1]), self.slots)

    def covers(self, rows: int) -> bool:
        return rows in ROWS

    def read(self, layer: int, x, res, bank, nvb: int, cw, w_o):
        """(out, prefix): the FFN input (post_attention_layernorm of the bank's mix) and the
        written residual, from this rank's o_proj partial ``x`` [m, H], the pending residual
        ``res`` (None after a write layer's attention read) and the bank [m, blocks, H] (its
        first ``nvb`` snapshots)."""
        m = x.shape[0]
        assert x.is_contiguous() and bank.is_contiguous() and 1 <= nvb <= min(K.NB_MAX, bank.shape[1])
        assert res is None or (res.is_contiguous() and res.shape == x.shape)
        out = torch.empty_like(x)
        pre = torch.empty_like(x)
        if layer == self.first:
            self.epoch[0:1].add_(1)
        self.launch(
            _p(x), _p(res if res is not None else x), _p(bank), _p(cw), _p(w_o), _p(pre), _p(out),
            m, nvb, bank.shape[1], int(res is not None), _p(self.peer.addresses), self.rank, _p(self.epoch), layer,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return out, pre


class MoeAr:
    """One rank's flat MoE all-reduce launches ([latent | shared] in one launch, the latent's RMSNorm
    in it; ``gfx950/stages/ar_norm.py``): their peer buffer, step epoch (bumped by the first MoE
    layer's call) and launch. ``layers``: the MoE layers' indices (each its own launch slot); ``eps``:
    routed_expert_norm's epsilon. Built like ``ArRead``."""

    def __init__(self, cpu_group, rank: int, device, layers, eps: float, deadline_s: float = 20.0):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k3.gfx950.stages import ar_norm as N

        dev = torch.device(device)
        self.rank = rank
        self.first = min(layers)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        ticks = guard.deadline_ticks(deadline_s)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self.peer = PeerBuffer(N.moe_peer_bytes(), cpu_group, rank, K.TP, dev, init=N.moe_arm)
        self.launch = N.build_moe(float(eps), max(layers) + 1)

    def covers(self, rows: int) -> bool:
        return rows in ROWS

    def reduce(self, layer: int, latent, shared, w):
        """(latent sum, shared sum, RMSNorm of the latent sum with ``w``) from this rank's partials
        latent [m, 3584] and shared [m, 7168] (bf16, contiguous rows)."""
        m = latent.shape[0]
        assert latent.is_contiguous() and shared.is_contiguous() and shared.shape[0] == m
        out_l, out_n, out_s = torch.empty_like(latent), torch.empty_like(latent), torch.empty_like(shared)
        if layer == self.first:
            self.epoch[0:1].add_(1)
        self.launch(
            _p(latent), _p(shared), _p(w), _p(out_l), _p(out_s), _p(out_n), m, _p(self.peer.addresses),
            self.rank, _p(self.epoch), layer, stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return out_l, out_s, out_n


class SharedPushAr:
    """One rank's split MoE all-reduce: the launch that computes the shared partial pushes it to every rank
    (``push_args``), then ``reduce`` runs the latent half with its RMSNorm and sums the pushed shared
    partials. Its peer buffer, step epoch (bumped at the first MoE layer's ``push_args``) and launch; built
    like ``MoeAr``."""

    def __init__(self, cpu_group, rank: int, device, layers, eps: float, deadline_s: float = 20.0):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k3.gfx950.stages import shared_push as S

        dev = torch.device(device)
        self.rank = rank
        self.first = min(layers)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        ticks = guard.deadline_ticks(deadline_s)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self.peer = PeerBuffer(S.peer_bytes(), cpu_group, rank, K.TP, dev, init=S.arm)
        self.launches = max(layers) + 1
        self.split_launch = S.build_split(float(eps), self.launches)
        self.slice_launch = S.build_split(float(eps), self.launches, gather=False)

    def covers(self, rows: int) -> bool:
        return rows in PUSH_ROWS

    def push_args(self, layer: int):
        """(peers, rank, epoch, slot): the arguments of ``shared_push.context`` for a launch that pushes this
        layer's shared partial itself (its tile epilogue: ``push_tile`` / ``flag_tile``), as int64 addresses and
        ints; the call before that launch, which also starts the step's epoch at the first MoE layer. The launch
        builds ``context`` with ``launches`` = ``self.launches``."""
        if layer == self.first:
            self.epoch[0:1].add_(1)
        return self.peer.addresses.data_ptr(), self.rank, self.epoch.data_ptr(), layer

    def reduce(self, layer: int, latent, shared, w):
        """``MoeAr.reduce``'s (latent sum, shared sum, RMSNorm of the latent sum) after this layer's shared
        partial was pushed by a launch given ``push_args(layer)``."""
        m = latent.shape[0]
        assert latent.is_contiguous() and shared.is_contiguous() and shared.shape[0] == m
        out_l, out_n, out_s = torch.empty_like(latent), torch.empty_like(latent), torch.empty_like(shared)
        self.split_launch(
            _p(latent), _p(shared), _p(w), _p(out_l), _p(out_s), _p(out_n), m, _p(self.peer.addresses),
            self.rank, _p(self.epoch), layer, stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return out_l, out_s, out_n

    def reduce_slice(self, layer: int, latent, shared, w):
        """``reduce`` for a consumer that reads only this rank's columns of the shared sum: (latent sum,
        this rank's slice of the shared sum [m, 896] (columns [896 rank, 896 (rank + 1)), contiguous rows),
        RMSNorm of the latent sum), with no gather of the shared sum."""
        m = latent.shape[0]
        assert latent.is_contiguous() and shared.is_contiguous() and shared.shape[0] == m
        out_l, out_n = torch.empty_like(latent), torch.empty_like(latent)
        out_s = shared.new_empty((m, shared.shape[1] // K.TP))
        self.slice_launch(
            _p(latent), _p(shared), _p(w), _p(out_l), _p(out_s), _p(out_n), m, _p(self.peer.addresses),
            self.rank, _p(self.epoch), layer, stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return out_l, out_s, out_n
