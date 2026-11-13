# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/moe_shape.py
"""The MoE build key (``MoeBuild``) and route shape (``RouteShape``)."""

import flydsl.expr as fx
import msgspec

from sgl_amd_mono.common.plan import BLOCKS, WAVES
from sgl_amd_mono.deepseek_v41.gfx950.stages import moe_plan as mplan
from sgl_amd_mono.deepseek_v41.gfx950.stages.dims import (
    UG_PART,
    Dims,
)
from sgl_amd_mono.deepseek_v41.gfx950.stages.gemv import (
    ROWS,
    TILE,
    token_tiles,
)
from sgl_amd_mono.deepseek_v41.gfx950.stages.router import PARTS as ROUTER_PARTS

EXPERTS, TOPK = 384, 6  # DeepSeek-V4.1-Flash routing
ROUTER0 = 8  # the router's first task: past the xq tasks (a token each)
MASK_WORDS = 2  # an expert's token bitmap in the route table: tokens <= 64
ROW_SLICES = 40  # a token row's CTAs in the per-row MoE: 128 hidden rows each (``RouteShape.row_moe``)
ROW_CTAS = 240  # the per-row MoE's CTAs at most: 6 rows
ROW_SHARED = 18  # the per-row MoE's shared expert units: on the last CTAs, past every row's ug units
# the per-row MoE's fewest rows a TP: a row's ug bytes a CTA are fixed (its topk experts' share over
# its 40 CTAs), the deduplicated split's grow with the experts a step routes; at TP4 (twice TP8's
# bytes a row) the per-row split is level from 4 rows
ROW_MIN_TOKENS = {4: 4, 8: 1}
# a route lane's experts sorted descending, ties in index order
SORT_NETS = {
    2: [(0, 1)],
    6: [
        (0, 1),
        (2, 3),
        (4, 5),
        (0, 2),
        (1, 4),
        (3, 5),
        (0, 1),
        (2, 3),
        (4, 5),
        (1, 2),
        (3, 4),
        (2, 3),
    ],
}


class MoeBuild(msgspec.Struct, frozen=True):
    tokens: int
    # the TP size: every width a rank holds (``Dims``)
    tp: int = 4
    # routed experts and a token's top-k: the target's, or the DSpark draft's
    experts: int = EXPERTS
    topk: int = TOPK
    # FP4 groups an ug task (``RouteShape.ug_groups``)
    ug_groups: int = 2


class RouteShape(msgspec.Struct, frozen=True):
    """A build's routing: task counts, placements and the LDS route table layout."""

    experts: int
    topk: int
    tokens: int
    tp: int
    # FP4 groups an ug task: 1 (K quarters) doubles the tasks so CTAs end ug closer (sglang#43984)
    ug_groups: int = 2

    @property
    def dims(self) -> Dims:
        return Dims(self.tp)

    @property
    def ug_parts(self) -> int:
        """An expert's ug tasks (``moe_plan.ug_parts``)."""
        return mplan.ug_parts(self.dims.inter_real // UG_PART, self.ug_groups)

    @property
    def ug_k_slices(self) -> int:
        """Return the K slices of an ug task's waves."""
        return WAVES // (2 * self.ug_groups)

    @property
    def picks(self) -> int:
        return self.tokens * self.topk

    @property
    def max_u(self) -> int:
        return min(self.picks, self.experts)

    @property
    def router_groups(self) -> tuple[int, int]:
        """Return ``stage_router``'s token groups and their tokens."""
        per_group = self.experts // ROWS * ROUTER_PARTS
        groups = max(1, min(len(token_tiles(self.tokens)), BLOCKS // per_group))
        tokens = -(-self.tokens // groups)
        assert (groups - 1) * tokens < self.tokens
        return groups, tokens

    @property
    def router_tasks(self) -> int:
        # (token group, 16-row tile, K part): ``stage_router``
        return self.router_groups[0] * self.experts // ROWS * ROUTER_PARTS

    @property
    def router0(self) -> int:
        """Past the xq tasks (a token each from CTA 0)."""
        return max(ROUTER0, self.tokens)

    @property
    def route0(self) -> int:
        return self.router0 + self.router_tasks

    @property
    def row_moe(self) -> bool:
        """The MoE split per (token row, slot), TileRT's: no expert dedup, the routing on every CTA,
        a CTA's down one (row, 128 hidden rows) of its row's picks. TP4 and TP8, one token tile, the
        target's route."""
        return (
            self.tp in ROW_MIN_TOKENS
            and self.tokens >= ROW_MIN_TOKENS[self.tp]
            and not self.multi
            and self.tokens * ROW_SLICES <= ROW_CTAS
            and (self.experts, self.topk) == (EXPERTS, TOPK)
        )

    @property
    def row_ug_ctas(self) -> int:
        """The per-row MoE's ug CTAs: those below the shared units."""
        return min(self.tokens * ROW_SLICES, self.shared0)

    @property
    def shared0(self) -> int:
        if self.row_moe:
            # the last CTAs: past every row's ug units
            return BLOCKS - self.shared_units
        return self.route0 + max(8, self.tokens)

    @property
    def shared_units(self) -> int:
        """The shared expert's (task, token tile) units (``stage_shared``); the per-row MoE's are
        ``ROW_SHARED`` units of ``shared_unit_tasks`` tasks."""
        if self.row_moe:
            return ROW_SHARED
        return self.dims.shared_tasks * len(token_tiles(self.tokens))

    @property
    def shared_unit_tasks(self) -> int:
        """A shared unit's tasks, run one after the other on its CTA (TP4's per-row MoE: two)."""
        if self.row_moe:
            assert self.dims.shared_tasks % ROW_SHARED == 0
            return self.dims.shared_tasks // ROW_SHARED
        return 1

    @property
    def ug0(self) -> int:
        """The first ug round's placement: the shared units' CTAs last."""
        return self.shared0 + self.shared_units

    @property
    def sort_net(self):
        return SORT_NETS[self.experts // 64]

    @property
    def tab_words(self) -> int:
        return self.tile_slot0 + self.max_tiles if self.multi else self.queue0 + 1

    @property
    def multi(self) -> bool:
        """True past one token tile: an ug task is then a (pick tile, part)."""
        return self.tokens > TILE

    @property
    def slot_tiles(self) -> int:
        """A slot's pick tiles at most (a token picks an expert once)."""
        return len(token_tiles(self.tokens))

    @property
    def max_tiles(self) -> int:
        """Return the maximum number of pick tiles of U."""
        return min(self.picks, self.max_u + self.picks // TILE)

    @property
    def tile_first0(self) -> int:
        """Past one token tile: each slot's first pick tile, then their count."""
        return self.queue0 + 1

    @property
    def tile_slot0(self) -> int:
        """Past one token tile: each pick tile's slot of U."""
        return self.tile_first0 + self.max_u + 1

    @property
    def queue0(self) -> int:
        """The ug task this CTA took from the queue (``take_ug_task``)."""
        return self.mask0 + MASK_WORDS * self.experts

    @property
    def expert_slot0(self) -> int:
        """The route table's per-expert slots (a slot where the expert is in U)."""
        return 3 * self.picks + 1 + self.max_u

    @property
    def order0(self) -> int:
        """The picks (t topk + k) in slot order, a slot's ascending."""
        return self.expert_slot0 + self.experts

    @property
    def first0(self) -> int:
        """Each slot's first position in the pick order, then the pick count."""
        return self.order0 + self.picks

    @property
    def mask0(self) -> int:
        """Each expert's tokens that picked it, a bit a token (``MASK_WORDS``)."""
        return self.first0 + self.max_u + 1

    @property
    def tile_picks(self) -> int:
        """A slot's picks at most: a token picks an expert once."""
        return min(self.tokens, 16)

    def route_w(self, tab, s, t, k):
        return fx.ptr_load(tab + (t * 2 * self.topk + self.topk + k)).bitcast(
            fx.Float32
        )

    def n_union(self, tab, s):
        return fx.ptr_load(tab + 2 * s * self.topk)

    def union_expert(self, tab, s, u):
        return fx.ptr_load(tab + (2 * s * self.topk + 1 + u))

    def slot_first(self, tab, u):
        return fx.ptr_load(tab + (self.first0 + u))

    def pick_at(self, tab, i):
        return fx.ptr_load(tab + (self.order0 + i))

    def tile_first(self, tab, u):
        return fx.ptr_load(tab + (self.tile_first0 + u))

    def tile_slot(self, tab, i):
        return fx.ptr_load(tab + (self.tile_slot0 + i))


def route_shape(key: MoeBuild) -> RouteShape:
    return RouteShape(key.experts, key.topk, key.tokens, key.tp, key.ug_groups)
