"""Audit kapı-metrik testleri ((b) kuralı: çevrilemeyen vs basılamayan)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_e2e_real_chapter1 import (
    count_short_dialogue_untranslated,
    count_short_unprinted,
)


def _region(rid, text, status="review", rtype="dialogue", reason="", translation=None):
    return {
        "id": rid, "text": text, "status": status, "type": rtype,
        "review_reason": reason, "translation": translation,
    }


def test_untranslated_counts_only_tr_less_shorts() -> None:
    regions = [
        _region(1, "DAMMIT", translation=None),  # çevirisiz -> kayıp
        _region(2, "WHAT?!", translation="Ne?!"),  # çevirili -> basılamayan
        _region(3, "BUT ALLEN...", reason="translation_guard_review", translation="Ama Allen..."),
        _region(4, "BOOM", rtype="sfx"),
        _region(5, "THIS IS A LONG STORY BALLOON TEXT HERE"),
        _region(6, "HEY!", status="skip"),
    ]
    assert count_short_dialogue_untranslated(regions) == 1
    # Basılamayan: çevirisi hazır WHAT (inpaint tutması) + guard tekili.
    assert count_short_unprinted(regions) == 2


def test_cjk_fragments_are_design_holds() -> None:
    regions = [
        _region(1, "トh。", reason="ambiguous_cjk_review"),
        _region(2, "erh.", reason="verifier_cjk"),
    ]
    assert count_short_dialogue_untranslated(regions) == 0
    assert count_short_unprinted(regions) == 0
