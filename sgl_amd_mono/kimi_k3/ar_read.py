# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's o_proj all-reduce fused with the FFN read of the attention-residual bank (TP8, gfx950),
for the widths SGLang's own layers run (``ROWS``): one launch a layer in place of the all-reduce
and ``attn_res_hip._agg_kernel``. The kernel: ``gfx950/stages/ar_read.py``.

SGLang's K3 decoder layer offers its FFN read to ``ArRead.read`` (a ReadoutFusion that also
reads); a width outside ``ROWS`` keeps SGLang's all-reduce and read.
"""

import torch

from sgl_amd_mono.kimi_k3.gfx950.stages import ar_read as K

# Above the row caps of the KDA and MLA layer launches, up to the widest graphed verify.
ROWS = range(49, K.MAX_ROWS + 1)


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
