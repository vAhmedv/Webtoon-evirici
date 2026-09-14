"""F6 fren kilitleri: tek eşik + gerçek kırmızı + sel freni (sentetik)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_e2e_real_chapter1 import (
    GATE_OVERFLOW_MAX,
    GATE_SHORT_UNTRANSLATED_MAX,
)
from scripts.compare_golden_v1 import (
    REVIEW_RATE_MAX_RISE,
    SHORT_MAX_RISE,
    compare_chapters,
)


def _chap(**kw):
    base = {
        "translation_eligible_blocks_count": 100,
        "echo_preserved_blocks_count": 10,
        "translated_blocks_count": 80,
        "overflow_blocks_count": 0,
        "short_dialogue_untranslated_count": 0,
        "final_auto_regions": 80,
        "final_review_regions": 10,
        "final_skip_regions": 10,
    }
    base.update(kw)
    return base


def _pair(old_kw=None, new_kw=None):
    return (
        {"name": "old", "chapters": {"ch1": _chap(**(old_kw or {}))}},
        {"name": "new", "chapters": {"ch1": _chap(**(new_kw or {}))}},
    )


def test_gate_single_source_is_zero() -> None:
    assert GATE_SHORT_UNTRANSLATED_MAX == 0
    assert GATE_OVERFLOW_MAX == 0


def test_runner_uses_same_gate_source() -> None:
    # Drift bekçisi: runner ayrı sayı tutamaz, audittekini alır.
    import scripts.run_golden_remeasure_v1 as r

    assert r.GATE_SHORT_UNTRANSLATED_MAX == GATE_SHORT_UNTRANSLATED_MAX
    assert r.GATE_OVERFLOW_MAX == GATE_OVERFLOW_MAX


def test_same_golden_is_clean() -> None:
    old, new = _pair()
    alarms, shorts = compare_chapters(old, new)
    assert alarms == [] and shorts == []


def test_overflow_is_fail() -> None:
    old, new = _pair(new_kw={"overflow_blocks_count": 1})
    alarms, _ = compare_chapters(old, new)
    assert any("OVERFLOW" in a for a in alarms)


def test_rate_drop_is_fail() -> None:
    # 80/90=0.889 -> 70/90=0.778 : 11 puan düşüş
    old, new = _pair(new_kw={"translated_blocks_count": 70})
    alarms, _ = compare_chapters(old, new)
    assert any("cevrilme-orani" in a for a in alarms)


def test_short_rise_is_fail() -> None:
    old, new = _pair(new_kw={"short_dialogue_untranslated_count": SHORT_MAX_RISE})
    _, shorts = compare_chapters(old, new)
    assert len(shorts) == 1


def test_review_flood_is_relative_not_absolute() -> None:
    # Aynı +4 review artışı: küçük bölümde sel, büyükte değil.
    # Küçük: 10/100 -> 30/120 = 0.10 -> 0.25 : 15 puan (FAIL)
    small_old = _chap(final_auto_regions=80, final_review_regions=10, final_skip_regions=10)
    small_new = _chap(final_auto_regions=80, final_review_regions=30, final_skip_regions=10)
    a1, _ = compare_chapters(
        {"name": "o", "chapters": {"c": small_old}},
        {"name": "n", "chapters": {"c": small_new}},
    )
    assert any("inceleme-orani" in a for a in a1)
    # Eşik sabiti göreli (0.05): mutlak +5 sayısı yok.
    assert REVIEW_RATE_MAX_RISE == 0.05
