"""A1 ikinci-görüş süzgeci: kopya/uzak/zerre elenir, komşu-kayıp aday kalır."""

from core.detection import BBox, Detection, RegionType
from core.detection.second_opinion import (
    bbox_edge_gap,
    max_iou_with,
    select_second_opinion,
)


def _det(x1, y1, x2, y2, conf=0.6):
    return Detection(
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=conf,
        type=RegionType.UNKNOWN,
        source_window_id=1,
        mask=None,
        metadata={},
    )


def test_gap_zero_on_overlap() -> None:
    assert bbox_edge_gap(BBox(x1=0, y1=0, x2=10, y2=10), BBox(x1=5, y1=5, x2=15, y2=15)) == 0
    assert bbox_edge_gap(BBox(x1=0, y1=0, x2=10, y2=10), BBox(x1=13, y1=0, x2=20, y2=10)) == 3


def test_max_iou_empty_is_zero() -> None:
    assert max_iou_with(BBox(x1=0, y1=0, x2=10, y2=10), []) == 0.0


def test_duplicate_rejected() -> None:
    ctd = [_det(100, 100, 200, 160)]
    yolo = [_det(105, 102, 195, 158, conf=0.9)]
    assert select_second_opinion(yolo, ctd, 1024, 800) == []


def test_far_box_rejected() -> None:
    ctd = [_det(100, 100, 200, 160)]
    yolo = [_det(600, 600, 660, 640)]
    assert select_second_opinion(yolo, ctd, 1024, 800) == []


def test_near_miss_accepted() -> None:
    # IS sınıfı: blok kutusunun dibinde, CTD ile örtüşmeyen küçük kutu.
    ctd = [_det(100, 120, 500, 360)]
    yolo = [_det(150, 60, 210, 100)]
    got = select_second_opinion(yolo, ctd, 1024, 800)
    assert len(got) == 1


def test_speck_rejected() -> None:
    ctd = [_det(100, 120, 500, 360)]
    yolo = [_det(150, 100, 154, 104)]
    assert select_second_opinion(yolo, ctd, 1024, 800) == []


def test_empty_inputs() -> None:
    assert select_second_opinion([], [_det(0, 0, 10, 10)], 1024, 800) == []
    assert select_second_opinion([_det(0, 0, 50, 50)], [], 1024, 800) == []
