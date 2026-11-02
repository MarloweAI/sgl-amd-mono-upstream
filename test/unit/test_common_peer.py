"""``PeerBuffer.close()`` keeps the uncached memory under a closed buffer, on CPU.

Freeing it is what the fix removes: pages a kernel had used uncached, once freed and
reused by ordinary allocations, gave later kernels stale cache lines. The test builds a
buffer without HIP and checks that closing it frees nothing.
"""

import pytest

from sgl_amd_mono.common import peer


class _NoHip:
    """A HIP runtime that fails on any call."""

    def __getattr__(self, name):
        raise AssertionError(f"close() called {name}")


def _buffer(local: int, opened: list[int]) -> peer.PeerBuffer:
    buf = object.__new__(peer.PeerBuffer)
    buf.rank, buf.local, buf._opened, buf.bytes = 0, local, opened, object()
    return buf


@pytest.fixture
def retired(monkeypatch):
    monkeypatch.setattr(peer, "_RETIRED", [], raising=False)
    monkeypatch.setattr(peer, "hip_runtime", lambda: _NoHip())
    return peer._RETIRED


def test_close_keeps_the_buffer_and_its_peer_mappings(retired):
    buf = _buffer(0x1000, [0x2000, 0x3000])
    buf.close()
    assert retired == [(0x1000, [0x2000, 0x3000])]
    assert (buf.local, buf._opened, buf.bytes) == (0, [], None)


def test_close_is_idempotent(retired):
    buf = _buffer(0x1000, [])
    buf.close()
    buf.close()
    assert retired == [(0x1000, [])]
