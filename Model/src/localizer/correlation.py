from __future__ import annotations

import torch
import torch.nn.functional as F


def dense_correlation(search_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
    b, c, hs, ws = search_feat.shape
    br, cr, hr, wr = ref_feat.shape
    assert (b, c) == (br, cr), "search and reference must share batch and channels"
    assert hr <= hs and wr <= ws, (
        f"reference feature map ({hr}x{wr}) must not exceed the search "
        f"feature map ({hs}x{ws}); got an oversized reference"
    )

    out = F.conv2d(
        search_feat.reshape(1, b * c, hs, ws),
        ref_feat.reshape(b * c, 1, hr, wr),
        groups=b * c,
    )
    return out.reshape(b, c, hs - hr + 1, ws - wr + 1)
