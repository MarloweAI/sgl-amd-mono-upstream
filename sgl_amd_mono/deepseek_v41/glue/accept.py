# SPDX-License-Identifier: Apache-2.0
"""The static verify graph's accept: SGLang's one-launch tail without the accept-site TP
broadcast, a site SGLang's ``init`` and ``rng`` presets leave off too. An explicit
``SGLANG_SPEC_TP_SYNC`` or ``SGLANG_AMD_MONO_DSV41_ACCEPT=0`` keeps SGLang's."""

from __future__ import annotations

import os
import sys

import torch

from sgl_amd_mono.deepseek_v41.glue import flag, route

ENV = "SGLANG_AMD_MONO_DSV41_ACCEPT"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
SPEC_UTILS = "sglang.srt.speculative.spec_utils"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/speculative/dspark_components/dspark_verify.py": (
        "DsparkVerifyEpilogue",
        "__init__(tp_sync)",
        "__init__(fused_accept_tail)",
        "TargetVerifyExecutor",
        "begin_simulated_step",
        "_simulate_acc_len",
        "_simulated_correct_len",
        "gamma",
        "target_worker",
        "SIMULATE_ACC_METHOD",
        "sample_simulated_acc_len",
    ),
    "srt/speculative/spec_tp_sync.py": ("SpecTpSyncSite", "DSPARK_ACCEPT_GRAPH"),
    "srt/managers/tp_worker.py": ("random_seed",),
    "srt/environ.py": ("SGLANG_SPEC_TP_SYNC", "SGLANG_DSPARK_FUSED_ACCEPT_TAIL"),
}

_GEN: list = []


class _WithoutAcceptSite:
    """A SpecTpSync without the accept-site broadcast."""

    def __init__(self, sync, site):
        self._sync, self._site = sync, site

    def enabled(self, site) -> bool:
        return site != self._site and self._sync.enabled(site)

    def sync(self, site, values):
        return values if site == self._site else self._sync.sync(site, values)

    def __getattr__(self, name):
        return getattr(self._sync, name)


def _epilogue_init(orig, self, *args, **kwargs):
    sync = kwargs.get("tp_sync")
    if not args and sync is not None:
        site = sys.modules[VERIFY].SpecTpSyncSite.DSPARK_ACCEPT_GRAPH
        kwargs["tp_sync"] = _WithoutAcceptSite(sync, site)
        if os.environ.get("SGLANG_DSPARK_FUSED_ACCEPT_TAIL") is None:
            kwargs["fused_accept_tail"] = True
    return orig(self, *args, **kwargs)


def _begin_simulated_step(orig, self):
    """The simulated length from one seeded stream, alike on every rank: with the
    broadcast off, nothing else makes the ranks agree."""
    if self._simulated_correct_len is None:
        return orig(self)
    if not _GEN:
        seed = int(self.target_worker.random_seed)
        _GEN.append(torch.Generator(device="cpu").manual_seed(seed))
    verify = sys.modules[VERIFY]
    n = verify.sample_simulated_acc_len(
        self._simulate_acc_len,
        verify.SIMULATE_ACC_METHOD,
        self.gamma + 1,
        generator=_GEN[0],
    )
    self._simulated_correct_len.fill_(n - 1)


def install() -> None:
    if not flag(ENV) or "SGLANG_SPEC_TP_SYNC" in os.environ:
        return
    import importlib

    verify = importlib.import_module(VERIFY)
    route(verify.DsparkVerifyEpilogue, "__init__", _epilogue_init)
    # SGLang with this generator draws the length and DSpark's bonus seed alike on every
    # rank; replacing its step would leave the bonus seed unset
    spec_utils = importlib.import_module(SPEC_UTILS)
    if not hasattr(spec_utils, "simulated_acc_len_generator"):
        route(
            verify.TargetVerifyExecutor, "begin_simulated_step", _begin_simulated_step
        )
