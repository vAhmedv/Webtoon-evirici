"""Faz 2 render guard testleri: tamamı sentetik fixture, gerçek bölüm verisi yok.

Anti-overfit: hiçbir test sayfa/bölüm numarası, piksel koordinat ezberi veya
bölüme özel metin içermez. Eşikler (`OVERLAP_IOU_THRESHOLD`) göreli geometriye
dayanır; tüm manhwa'lara uygulanır.
"""

import pytest
from collections.abc import Iterator
from loguru import logger
from PIL import Image

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.imaging.renderer import (
    TextRenderer,
    _clean_orphan_quotes,
    _group_overlapping,
    _has_word_content,
    _normalize_render_text,
)


def _member(region_id: int, x1: int, y1: int, x2: int, y2: int) -> Region:
    return Region(
        id=region_id,
        global_bbox=BBox(x1, y1, x2, y2),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text=f"KAYNAK {region_id}",
        metadata={},
    )


def _block(block_id: int, x1: int, y1: int, x2: int, y2: int) -> TextBlock:
    member = _member(block_id, x1, y1, x2, y2)
    return TextBlock(
        id=block_id,
        member_ids=(member.id,),
        members=(member,),
        merged_bbox=BBox(x1, y1, x2, y2),
        source_text=f"SOURCE {block_id}",
    )


def _canvas() -> Image.Image:
    return Image.new("RGB", (400, 400), "white")


@pytest.fixture
def warnings_sink() -> Iterator[list[str]]:
    """loguru WARNING+ kayıtlarını yakalar (caplog loguru'yu görmez)."""
    msgs: list[str] = []
    handler_id = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    yield msgs
    logger.remove(handler_id)


# --- Yardımcı fonksiyon birim testleri ---


def test_has_word_content() -> None:
    assert _has_word_content("MERHABA DÜNYA")
    assert _has_word_content("123")
    assert not _has_word_content("…?!")
    assert not _has_word_content("   ...   ")
    assert not _has_word_content("")


def test_clean_orphan_quotes_drops_lone_double_quote() -> None:
    assert _clean_orphan_quotes('iz"bırakmak') == "izbırakmak"
    assert _clean_orphan_quotes('"NE ...?!') == "NE ...?!"


def test_clean_orphan_quotes_keeps_balanced_and_apostrophe() -> None:
    assert _clean_orphan_quotes('"ALINTI"') == '"ALINTI"'
    # ASCII kesme işareti Türkçe tamlamalarda meşru, dokunulmaz.
    assert _clean_orphan_quotes("VRMMO'su") == "VRMMO'su"
    assert _clean_orphan_quotes("DÜNYANIN EN POPÜLERİ") == "DÜNYANIN EN POPÜLERİ"


def test_normalize_render_text() -> None:
    assert _normalize_render_text("  Merhaba   DÜNYA ") == "merhaba dünya"


def test_group_overlapping_singletons() -> None:
    a = _block(1, 10, 10, 100, 60)
    b = _block(2, 200, 200, 300, 260)
    groups = _group_overlapping([(a, "metin a", a.merged_bbox), (b, "metin b", b.merged_bbox)])
    assert len(groups) == 2


def test_group_overlapping_merges_and_transitive() -> None:
    # a~b çakışır, b~c çakışır, a~c çakışmaz → geçişlilikle tek grup.
    a = _block(1, 10, 10, 110, 60)
    b = _block(2, 60, 10, 160, 60)
    c = _block(3, 110, 10, 210, 60)
    assert a.merged_bbox.iou(b.merged_bbox) > 0
    groups = _group_overlapping(
        [(a, "a", a.merged_bbox), (b, "b", b.merged_bbox), (c, "c", c.merged_bbox)],
        iou_threshold=0.01,
    )
    assert len(groups) == 1
    assert len(groups[0]) == 3


# --- render_blocks entegrasyon testleri ---


def test_duplicate_block_id_renders_once(warnings_sink) -> None:
    renderer = TextRenderer()
    block = _block(7, 50, 50, 350, 120)
    _, rendered, _ = renderer.render_blocks(
        _canvas(), [(block, "AYNI METİN BURADA"), (block, "AYNI METİN BURADA")]
    )
    assert rendered == 1
    assert any("iki kez" in msg for msg in warnings_sink)


def test_overlapping_same_text_renders_once() -> None:
    renderer = TextRenderer()
    a = _block(1, 50, 50, 350, 120)
    b = _block(2, 60, 55, 340, 115)  # ~tam çakışma
    _, rendered, _ = renderer.render_blocks(
        _canvas(), [(a, "AYNI CÜMLE BURADA"), (b, "AYNI CÜMLE BURADA")]
    )
    assert rendered == 1


def test_overlapping_different_text_keeps_longest(warnings_sink) -> None:
    renderer = TextRenderer()
    short = _block(1, 50, 50, 350, 120)
    long = _block(2, 60, 55, 340, 115)
    _, rendered, _ = renderer.render_blocks(
        _canvas(), [(short, "KISA"), (long, "BU ÇOK DAHA UZUN BİR CÜMLE BURADA")]
    )
    assert rendered == 1
    assert any("çakışan" in msg for msg in warnings_sink)


def test_separate_bubbles_both_render() -> None:
    renderer = TextRenderer()
    a = _block(1, 50, 50, 350, 120)
    b = _block(2, 50, 200, 350, 270)  # ayrı balon, çakışma yok
    _, rendered, _ = renderer.render_blocks(
        _canvas(), [(a, "BİRİNCİ BALON METNİ"), (b, "İKİNCİ BALON METNİ")]
    )
    assert rendered == 2


def test_punctuation_only_text_skipped() -> None:
    renderer = TextRenderer()
    block = _block(1, 50, 50, 350, 120)
    _, rendered, _ = renderer.render_blocks(_canvas(), [(block, "…?!")])
    assert rendered == 0
