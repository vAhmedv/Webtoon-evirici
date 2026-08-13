"""OCR normalizer ve agreement/disagreement unit testleri."""

from __future__ import annotations

from core.ocr_normalizer import normalize_ocr_text
from providers.ocr.agreement import decide_ocr_agreement
from providers.ocr.base import OCRResult


class TestNormalizeOCRText:
    """Güvenli whitespace normalizasyonu."""

    def test_newline_to_space(self) -> None:
        assert normalize_ocr_text("RELAX, KID. YOU\nSAW IT YOURSELF\nJUST NOW.") == (
            "RELAX, KID. YOU SAW IT YOURSELF JUST NOW."
        )

    def test_multiple_whitespace_collapse(self) -> None:
        assert normalize_ocr_text("HELLO   WORLD\t\tTEST") == "HELLO WORLD TEST"

    def test_trim(self) -> None:
        assert normalize_ocr_text("  HELLO WORLD  ") == "HELLO WORLD"

    def test_empty(self) -> None:
        assert normalize_ocr_text("") == ""
        assert normalize_ocr_text(None) == ""

    def test_unicode_whitespace(self) -> None:
        assert normalize_ocr_text("HELLO\u00a0WORLD") == "HELLO WORLD"

    def test_no_word_correction(self) -> None:
        # Kelime düzeltme YAPILMAMALI
        assert normalize_ocr_text("PLSHOVERS") == "PLSHOVERS"
        assert normalize_ocr_text("WHO TIAN") == "WHO TIAN"
        assert normalize_ocr_text("Huh San") == "Huh San"


class TestOCRAgreement:
    """VL-1.6 vs Paddle v5 agreement/disagreement."""

    def _result(self, raw: str, confidence: float | None = None) -> OCRResult:
        return OCRResult(
            text=raw,
            confidence=confidence,
            raw_text=raw,
            lines=[],
            warnings=[],
        )

    def test_exact_match_primary_wins(self) -> None:
        primary = self._result("RELAX, KID. YOU\nSAW IT YOURSELF\nJUST NOW.")
        verifier = self._result("RELAX, KID. YOU\nSAW IT YOURSELF\nJUST NOW.")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.source == "primary"
        assert verdict.requires_review is False
        assert verdict.needs_repair is False
        assert verdict.accepted_text == "RELAX, KID. YOU SAW IT YOURSELF JUST NOW."

    def test_format_only_difference_primary_wins(self) -> None:
        primary = self._result("RELAX, KID. YOU\nSAW IT YOURSELF\nJUST NOW.")
        verifier = self._result("RELAX, KID. YOU SAW IT YOURSELF JUST NOW.")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.source == "primary"
        assert verdict.requires_review is False
        assert verdict.needs_repair is False

    def test_case_only_difference_safe_agreement(self) -> None:
        primary = self._result("HELLO WORLD")
        verifier = self._result("hello world")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.accepted_text == "HELLO WORLD"
        assert verdict.requires_review is False
        assert verdict.needs_repair is False
        assert verdict.reason is None

    def test_relax_kid_newline_only_auto_accept(self) -> None:
        primary = self._result("RELAX, KID. YOU SAW IT YOURSELF JUST NOW.")
        verifier = self._result("RELAX, KID. YOU\nSAW IT YOURSELF\nJUST NOW.")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.accepted_text == "RELAX, KID. YOU SAW IT YOURSELF JUST NOW."
        assert verdict.requires_review is False
        assert verdict.needs_repair is False
        assert verdict.source == "primary"

    def test_critical_name_mismatch_needs_repair(self) -> None:
        # VL: LHO TIAN (hallucination), Paddle: LUO TIAN
        primary = self._result("MY NAME IS LHO\nTIAN. I'M NOT AN\nABILITY USER.")
        verifier = self._result("MY NAME IS LUO\nTIAN. I'M NOT AN\nABILITY USER.")
        verdict = decide_ocr_agreement(primary, verifier)
        # Otomatik seçim YOK
        assert verdict.accepted_text is None
        assert verdict.requires_review is True
        assert verdict.needs_repair is True
        assert verdict.reason == "word_difference"
        # Raw'lar korunur
        assert verdict.primary_raw == "MY NAME IS LHO\nTIAN. I'M NOT AN\nABILITY USER."
        assert verdict.verifier_raw == "MY NAME IS LUO\nTIAN. I'M NOT AN\nABILITY USER."

    def test_hu_san_mismatch_needs_repair(self) -> None:
        primary = self._result("HLI San, you're the fastest.")
        verifier = self._result("HU SAN, YOU'RE THE FASTEST.")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.accepted_text is None
        assert verdict.requires_review is True
        assert verdict.needs_repair is True
        assert verdict.reason == "word_difference"

    def test_word_difference_needs_repair_no_auto_select(self) -> None:
        # PUSHOVERS vs PLSHOVERS — Paddle otomatik kabul EDİLMEMELİ
        primary = self._result("CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER, AND THE REST OF THE TEAM ARE NO PUSHOVERS EITHER.")
        verifier = self._result("CAPTAIN GAO YUAN IS A\nPEAK LEVEL 1 ABILITY\nUSER, AND THE REST\nOF THE TEAM ARE NO\nPLSHOVERS EITHER.")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.accepted_text is None
        assert verdict.requires_review is True
        assert verdict.needs_repair is True
        assert verdict.reason == "word_difference"
        # Paddle sonucu otomatik accepted_text olmamalı
        assert verdict.accepted_text != "CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER, AND THE REST OF THE TEAM ARE NO PLSHOVERS EITHER."
        # provisional_text primary (VL) metnini taşır
        assert verdict.provisional_text == "CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER, AND THE REST OF THE TEAM ARE NO PUSHOVERS EITHER."
        assert verdict.repaired_text is None

    def test_primary_empty_verifier_filled_needs_repair(self) -> None:
        primary = self._result("")
        verifier = self._result("ROGER, CAPTAIN.")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.accepted_text is None
        assert verdict.requires_review is True
        assert verdict.needs_repair is True
        assert verdict.reason == "primary_empty_verifier_filled"
        assert verdict.provisional_text == "ROGER, CAPTAIN."

    def test_primary_cjk_needs_repair(self) -> None:
        primary = self._result("これはテストです")
        verifier = self._result("THIS IS A TEST.")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.accepted_text is None
        assert verdict.requires_review is True
        assert verdict.needs_repair is True
        assert "cjk" in verdict.reason

    def test_raw_texts_preserved(self) -> None:
        primary = self._result("LHO TIAN")
        verifier = self._result("LUO TIAN")
        verdict = decide_ocr_agreement(primary, verifier)
        assert verdict.primary_raw == "LHO TIAN"
        assert verdict.verifier_raw == "LUO TIAN"
        assert verdict.primary_normalized == "LHO TIAN"
        assert verdict.verifier_normalized == "LUO TIAN"

    def test_single_pass_clean_ocr_auto_accepted(self) -> None:
        primary = self._result("WHAT THE...?")
        verdict = decide_ocr_agreement(primary, verifier=None)
        assert verdict.requires_review is False
        assert verdict.needs_repair is False
        assert verdict.accepted_text == "WHAT THE...?"

    def test_single_pass_structurally_suspicious_ocr_concatenated(self) -> None:
        primary = self._result("CRAPTEDWEAPONS CAN BE GRANTED TO OTHERS.")
        verdict = decide_ocr_agreement(primary, verifier=None)
        assert verdict.requires_review is True
        assert verdict.needs_repair is True
        assert "concatenated_token" in (verdict.reason or "")

    def test_single_pass_structurally_suspicious_ocr_invalid_consonants(self) -> None:
        primary = self._result("CRAFTED WEAPONS CAN PE CDANTED TO OTUEDS")
        verdict = decide_ocr_agreement(primary, verifier=None)
        assert verdict.requires_review is True
        assert verdict.needs_repair is True
        assert "invalid_start_consonants" in (verdict.reason or "")

    def test_single_pass_structurally_suspicious_ocr_invalid_cluster(self) -> None:
        primary = self._result("APOKTTON OPTHE EXPIS SHARED WITH THE CREATOR.")
        verdict = decide_ocr_agreement(primary, verifier=None)
        assert verdict.requires_review is True
        assert verdict.needs_repair is True
        assert "invalid_consonant_cluster" in (verdict.reason or "")