"""Text-Block grouping module.

Groups post-OCR consecutive lines inside the same speech bubble/panel into
a single TextBlock (Translation Unit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Sequence

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection.bbox import BBox
from core.detection.detection import Region, RegionStatus, RegionType


@dataclass(frozen=True)
class TextBlock:
    """Tek bir konuşma balonu / metin bloğunu temsil eden grup.

    Attributes:
        id: Gruplanmış blok kimliği.
        member_ids: Bloğa ait sıralı Region ID'leri.
        members: Bloğa ait sıralı Region nesneleri (okuma sırasına göre).
        merged_bbox: Bloğun tüm üyelerini kapsayan global BBox.
        source_text: Bloğa ait birleştirilmiş İngilizce kaynak metin.
        translation: Bloğa ait birleştirilmiş Türkçe çeviri (varsa).
        metadata: Ek grup verileri.
    """

    id: int
    member_ids: tuple[int, ...]
    members: tuple[Region, ...]
    merged_bbox: BBox
    source_text: str
    translation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _compute_merged_bbox(boxes: Sequence[BBox]) -> BBox:
    """Birden fazla BBox'ı kapsayan birleşik BBox üretir."""
    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _metadata_boxes(region: Region) -> list[BBox]:
    values = region.metadata.get("ctd_block_bboxes") or [region.metadata.get("ctd_block_bbox")]
    result: list[BBox] = []
    if not isinstance(values, list):
        return result
    for value in values:
        if isinstance(value, list) and len(value) == 4:
            try:
                result.append(BBox(*(int(round(float(part))) for part in value)))
            except (TypeError, ValueError):
                continue
    return result


def _same_ctd_container(r1: Region, r2: Region) -> bool:
    ids1 = set(str(value) for value in r1.metadata.get("ctd_block_ids", []) if value)
    ids2 = set(str(value) for value in r2.metadata.get("ctd_block_ids", []) if value)
    if ids1 and ids2 and ids1.intersection(ids2):
        return True
    boxes1, boxes2 = _metadata_boxes(r1), _metadata_boxes(r2)
    for box1 in boxes1:
        for box2 in boxes2:
            if box1.iou(box2) >= 0.55:
                return True
            merged = _compute_merged_bbox((box1, box2))
            intersection = box1.intersection(box2)
            overlap = intersection.area if intersection is not None else 0
            if overlap / max(1, min(box1.area, box2.area)) >= 0.75 and merged.area <= max(box1.area, box2.area) * 1.35:
                return True
    return False


def _explicit_different_containers(r1: Region, r2: Region) -> bool:
    ids1 = set(str(value) for value in r1.metadata.get("ctd_block_ids", []) if value)
    ids2 = set(str(value) for value in r2.metadata.get("ctd_block_ids", []) if value)
    return bool(ids1 and ids2 and ids1.isdisjoint(ids2))


def _is_hyphen_continuation(upper: Region, lower: Region) -> bool:
    first = (upper.text or "").strip()
    second = (lower.text or "").strip()
    if not first or not second:
        return False
    return bool(re.search(r"[-–—=]\s*[\"'\)]*$", first))


def _looks_like_continuation(upper: Region, lower: Region) -> bool:
    first = (upper.text or "").strip()
    second = (lower.text or "").strip()
    if not first or not second:
        return False
    if _is_hyphen_continuation(upper, lower):
        return True
    # A terminal sentence is a strong boundary; otherwise an aligned following
    # line is allowed to continue, including conjunction-led lines.
    if re.search(r"[.!?][\"')\]]?$", first):
        return False
    return bool(re.search(r"[A-Za-z0-9]", second))


def _are_adjacent_lines(r1: Region, r2: Region, is_immediate_consecutive: bool = False) -> bool:
    """İki bölgenin aynı konuşma balonunda üst üste gelen iki satır olup olmadığını kontrol eder.

    Muhafazakâr (conservative) eşleşme kriterleri:
    - Yakın dikey mesafe
    - Yatay çakışma veya hizalanma
    - Benzer satır yüksekliği
    """
    b1, b2 = r1.global_bbox, r2.global_bbox
    h1, h2 = b1.height, b2.height
    w1, w2 = b1.width, b2.width

    if h1 <= 0 or h2 <= 0 or w1 <= 0 or w2 <= 0:
        return False

    # Satır yükseklik oranı çok farklıysa birleştirme (örn: dev font + küçük altyazı)
    height_ratio = min(h1, h2) / max(h1, h2)
    if height_ratio < 0.4:
        return False

    min_h = min(h1, h2)
    
    # Dikey mesafe kontrolü (r1 üstte, r2 altta)
    if b2.y1 >= b1.y1:
        v_gap = b2.y1 - b1.y2
    else:
        v_gap = b1.y1 - b2.y2

    same_container = _same_ctd_container(r1, r2)
    different_containers = _explicit_different_containers(r1, r2)
    upper, lower = (r1, r2) if b1.y1 <= b2.y1 else (r2, r1)
    is_hyphen = _is_hyphen_continuation(upper, lower)
    continuation = _looks_like_continuation(upper, lower)

    # Explicit different CTD containers block merging UNLESS there is a strong hyphenated continuation between immediate consecutive lines
    if different_containers:
        if not (is_hyphen and is_immediate_consecutive):
            return False
    if _metadata_boxes(r1) and _metadata_boxes(r2) and not same_container and not continuation:
        return False

    # CTD container/continuation evidence recovers an occasional missed last line,
    # while the default remains the previous conservative 1.6x threshold.
    gap_factor = 2.2 if (same_container or is_hyphen) else (1.85 if continuation else 1.6)
    max_allowed_gap = int(min_h * gap_factor)
    if v_gap > max_allowed_gap or v_gap < -int(min_h * 0.4):
        return False

    # Yatay çakışma & hizalanma kontrolü
    overlap_x = max(0, min(b1.x2, b2.x2) - max(b1.x1, b2.x1))
    overlap_ratio = overlap_x / min(w1, w2)

    cx1 = (b1.x1 + b1.x2) / 2.0
    cx2 = (b2.x1 + b2.x2) / 2.0
    center_x_diff = abs(cx1 - cx2)

    left_diff = abs(b1.x1 - b2.x1)
    right_diff = abs(b1.x2 - b2.x2)
    min_edge_diff = min(left_diff, right_diff)

    if same_container or is_hyphen:
        min_overlap = 0.05
        max_center_diff = max(w1, w2) * 0.85
        max_edge_diff = max(w1, w2) * 0.85
    elif continuation:
        min_overlap = 0.08
        max_center_diff = max(w1, w2) * 0.75
        max_edge_diff = max(w1, w2) * 0.75
    else:
        min_overlap = 0.20
        max_center_diff = max(w1, w2) * 0.60
        max_edge_diff = max(w1, w2) * 0.60

    if overlap_ratio < min_overlap and center_x_diff > max_center_diff and min_edge_diff > max_edge_diff:
        return False

    return True


def group_text_blocks(
    regions: Sequence[Region],
    coords: GlobalCoordinateSystem,
) -> list[TextBlock]:
    """AUTO durumundaki hikaye metni bölgelerini muhafazakâr biçimde TextBlock'lara gruplar.

    Args:
        regions: Canonical Region listesi.
        coords: Global koordinat sistemi.

    Returns:
        Oluşturulan TextBlock listesi.
    """
    # Yalnızca geçerli hikaye metinlerini grupla (SKIP edilenler hariç)
    candidates: list[Region] = []
    for r in regions:
        if r.status == RegionStatus.SKIP:
            continue
        if r.type in (RegionType.SFX, RegionType.WATERMARK):
            continue
        if not r.text or not r.text.strip():
            continue
        candidates.append(r)

    if not candidates:
        return []

    # Sayfalara göre grupla
    page_buckets: dict[int, list[Region]] = {}
    for r in candidates:
        center_y = (r.global_bbox.y1 + r.global_bbox.y2) // 2
        page_idx, _ = coords.global_to_page(center_y)
        page_buckets.setdefault(page_idx, []).append(r)

    blocks: list[TextBlock] = []
    block_id_counter = 1

    for page_idx in sorted(page_buckets.keys()):
        page_regions = page_buckets[page_idx]
        
        # Üstten alta sırala
        page_regions.sort(key=lambda r: (r.global_bbox.y1, r.global_bbox.x1))

        # Adjacency graph / connected components
        n = len(page_regions)
        parent = list(range(n))

        def find(i: int) -> int:
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(n):
            for j in range(i + 1, n):
                r1 = page_regions[i]
                r2 = page_regions[j]
                
                # Çok uzak dikey mesafedeyse erken döngü kırma
                if r2.global_bbox.y1 - r1.global_bbox.y2 > max(r1.global_bbox.height, r2.global_bbox.height) * 2.5:
                    break

                if _are_adjacent_lines(r1, r2, is_immediate_consecutive=(j == i + 1)):
                    union(i, j)

        # Component'leri topla
        components: dict[int, list[Region]] = {}
        for idx in range(n):
            root = find(idx)
            components.setdefault(root, []).append(page_regions[idx])

        for comp_regions in components.values():
            # Reading order sıralama (İngilizce webtoon: top-to-bottom primary, left-to-right secondary)
            comp_regions.sort(key=lambda r: (r.global_bbox.y1, r.global_bbox.x1))

            member_ids = tuple(r.id for r in comp_regions)
            merged_box = _compute_merged_bbox([r.global_bbox for r in comp_regions])
            source_text = " ".join(r.text.strip() for r in comp_regions if r.text and r.text.strip())

            blocks.append(
                TextBlock(
                    id=block_id_counter,
                    member_ids=member_ids,
                    members=tuple(comp_regions),
                    merged_bbox=merged_box,
                    source_text=source_text,
                    metadata={
                        "page_index": page_idx,
                        "member_count": len(comp_regions),
                        "ctd_line_mapping": [
                            {"region_id": region.id, **line}
                            for region in comp_regions
                            for line in region.metadata.get("ctd_line_memberships", [])
                            if isinstance(line, dict)
                        ],
                    },
                )
            )
            block_id_counter += 1

    return blocks
