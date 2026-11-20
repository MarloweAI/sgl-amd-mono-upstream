"""The Kimi-K3 forward this plugin reproduces, pinned by source digest.

``_Built`` replaces the whole model forward for a routed step. So
``_backbone``, ``_layer`` and ``_pre_o_proj`` are a copy of what SGLang's own
forward does. ``SGLANG_READS`` catches a rename. It cannot catch a changed
branch. A changed branch here gives wrong tokens on routed steps and right
tokens everywhere else, which is the hardest kind of bug to find.

This test fails whenever one of those functions changes at all, including on a
comment. That is deliberate, and it is noisy.

When it fails, do this. Read the SGLang function next to the plugin function
named below. Decide whether the change affects a routed step. Update
``_Built`` if it does. Then put the new digest here, in the same commit. Never
update a digest without reading the diff. That is the whole point of the test.
"""

import hashlib
import inspect

import pytest
from sglang.srt.models import kimi_k3

# (class, method) -> sha256 prefix of inspect.getsource, and what in
# sgl_amd_mono/kimi_k3/spec.py reproduces it
PINNED = {
    ("KimiK3ForConditionalGeneration", "forward"): (
        "e35d07d1e2292cff",
        "_Built.forward",
    ),
    ("KimiK3LinearForCausalLM", "forward"): (
        "427cb949b79cf431",
        "_Built.language_forward",
    ),
    ("KimiK3LinearModel", "forward"): (
        "6d19ecd8cf8bb53e",
        "_Built._backbone",
    ),
    ("KimiK3DecoderLayer", "forward"): (
        "373da46f0dd3bf07",
        "_Built._layer",
    ),
    ("KimiK3DecoderLayer", "_run_self_attn"): (
        "d0d26643498322e1",
        "_Built._pre_o_proj",
    ),
}


def _digest(cls: str, method: str) -> str:
    src = inspect.getsource(getattr(getattr(kimi_k3, cls), method))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


@pytest.mark.parametrize(
    ("cls", "method"), sorted(PINNED), ids=lambda p: p if isinstance(p, str) else str(p)
)
def test_the_reproduced_forward_has_not_drifted(cls, method):
    want, mirror = PINNED[cls, method]
    got = _digest(cls, method)
    assert got == want, (
        f"{cls}.{method} changed ({want} -> {got}). "
        f"sgl_amd_mono/kimi_k3/spec.py's {mirror} reproduces it: read the diff, "
        f"update {mirror} if a routed decode or verify step is affected, then put "
        f"{got} here in the same commit."
    )
