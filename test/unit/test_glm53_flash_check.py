"""The GLM-5.3-Flash serving check's index-cache comparison on CPU: the dequantized
keys of a page (FP8 value x the pool's power-of-two scale). A pool requantized at the
next scale passes ``dsa.index.cosine``; a pool's scale doubled with its values kept
fails it."""

import torch

from sgl_amd_mono.glm53_flash import check
from sgl_amd_mono.glm53_flash.spec import NUMERICS

# the byte of pool p, dim c in a page: AITER's 16-pool x 16-byte tiles
_P, _C = torch.arange(64)[:, None], torch.arange(128)[None, :]
_POS = ((_P // 16) * 2048 + (_C // 16) * 256 + (_P % 16) * 16 + _C % 16).reshape(-1)


def _page(keys: torch.Tensor) -> torch.Tensor:
    """An index-cache page from keys [64 pools, 128]: power-of-two scales of each pool's
    amax / 448, FP8 values in AITER's 16-pool x 16-byte tiles, then the scales."""
    scale = torch.exp2(torch.ceil(torch.log2(keys.abs().amax(1) / 448)))
    q = (keys / scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    page = torch.zeros(1, 64 * 132, dtype=torch.uint8)
    page[0, _POS] = q.reshape(-1)
    page[0, 64 * 128 :] = scale.view(torch.uint8)
    return page


def test_index_keys_decode_the_tiled_page():
    keys = torch.randn(64, 128, generator=torch.Generator().manual_seed(0))
    got = check._index_keys(_page(keys))[0]
    # in the page's byte order: pool p, dim c at its tile position
    want = torch.empty(64 * 128)
    want[_POS] = keys.reshape(-1)
    pools = _POS.view(64, 128)
    cos = torch.nn.functional.cosine_similarity(got[pools], want[pools], dim=1)
    assert cos.min() > 0.999, cos.min()


def test_a_pool_at_the_next_scale_passes_and_a_wrong_scale_fails():
    bound = NUMERICS.bounds["dsa.index.cosine"]
    keys = torch.randn(64, 128, generator=torch.Generator().manual_seed(1))
    ref = _page(keys)
    # pool 5 requantized at twice its scale (its amax at a power of two): the same keys
    # to FP8 precision
    nudged = ref.clone()
    scales = nudged[0, 64 * 128 :].view(torch.float32)
    values = (keys[5] / (2 * scales[5])).to(torch.float8_e4m3fn).view(torch.uint8)
    nudged[0, _POS.view(64, 128)[5]] = values
    scales[5] *= 2
    ok = check._outputs("index", nudged, ref)["index"]["cosine"]
    assert ok >= bound, ok
    # pool 5's scale doubled with its values kept (a wrong scale on one row's pool)
    wrong = ref.clone()
    wrong[0, 64 * 128 :].view(torch.float32)[5] *= 2
    bad = check._outputs("index", wrong, ref)["index"]["cosine"]
    assert bad < bound, bad
