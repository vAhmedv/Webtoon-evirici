"""Window-local ve global bbox koordinat dönüşümleri."""

from __future__ import annotations

from .bbox import BBox


def window_bbox_to_global(
    bbox: BBox,
    window_y_start: int,
) -> BBox:
    """Window-local bbox'ı global koordinata çevirir.

    X koordinatı değişmez. Y koordinatına window.y_start eklenir.

    Args:
        bbox: Window-local BBox.
        window_y_start: Window'ın global Y başlangıcı.

    Returns:
        Global koordinatlı BBox.
    """
    return BBox(
        x1=bbox.x1,
        y1=bbox.y1 + window_y_start,
        x2=bbox.x2,
        y2=bbox.y2 + window_y_start,
    )


def global_bbox_to_window(
    bbox: BBox,
    window_y_start: int,
) -> BBox:
    """Global bbox'ı window-local koordinata çevirir.

    Args:
        bbox: Global BBox.
        window_y_start: Window'ın global Y başlangıcı.

    Returns:
        Window-local BBox.
    """
    return BBox(
        x1=bbox.x1,
        y1=bbox.y1 - window_y_start,
        x2=bbox.x2,
        y2=bbox.y2 - window_y_start,
    )