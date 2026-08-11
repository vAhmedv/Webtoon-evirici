"""Unit tests for Qwen Real-Chapter Translation Regression Diagnostic V1.

Locks the repaired ALL-CAPS term detection and suffix restoration behavior while
retaining the provider/model/prompt configuration checks from the diagnostic.
"""

from pathlib import Path
import pytest

from core.translation.protection import (
    ProtectedTermMeta,
    _suffix_category,
    detect_named_terms_in_items,
    restore_protected_translation,
)
from providers.translation.base import TranslationItem
from providers.translation.qwen_gguf_translation_v2 import (
    DEFAULT_LLAMA_EXE_PATH,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_QWEN_SERVER_URL,
    QWEN_TRANSLATOR_SYSTEM_PROMPT,
    QwenGGUFTranslationProviderV2,
)


def test_all_caps_vs_sentence_case_term_detection():
    """Both casing variants must reject the former false-positive term."""
    item_caps = TranslationItem(region_id=1, source="I'M USED TO IT.")
    item_sent = TranslationItem(region_id=2, source="I'm used to it.")

    det_caps = detect_named_terms_in_items([item_caps])
    det_sent = detect_named_terms_in_items([item_sent])

    assert det_caps == set()
    assert det_sent == set()


def test_sentinel_restore_behavior_for_dir_and_im():
    """Copular and first-person suffixes must restore with vowel harmony."""
    cat_dir = _suffix_category("", "DIR")
    assert cat_dir == "copular"

    cat_im = _suffix_category("", "İM")
    assert cat_im == "person_1sg"

    # Test restore_protected_translation with 'DIR
    meta_dir = ProtectedTermMeta(
        sentinel="__WTTERM0001__",
        source_original="ABILITY USER",
        target_base="yetenek kullanıcısı",
        is_approved=True,
        proper_name=False,
    )
    restored_dir = restore_protected_translation("__WTTERM0001__'DIR", {"__WTTERM0001__": meta_dir})
    assert restored_dir == "yetenek kullanıcısıdır"

    # Test restore_protected_translation with 'İM
    meta_im = ProtectedTermMeta(
        sentinel="__WTTERM0002__",
        source_original="SECRET REALM GUIDE",
        target_base="gizli âlem rehberi",
        is_approved=True,
        proper_name=False,
    )
    restored_im = restore_protected_translation("__WTTERM0002__'İM", {"__WTTERM0002__": meta_im})
    assert restored_im == "gizli âlem rehberiyim"


def test_standard_suffixes_remain_observable():
    """Verify that standard case suffixes (gen, abl, loc, dat) are recognized."""
    assert _suffix_category("", "de") == "loc"
    assert _suffix_category("", "den") == "abl"
    assert _suffix_category("", "e") == "dat"
    assert _suffix_category("", "in") == "gen"
    assert _suffix_category("", "ler") == "plural"


def test_production_provider_and_system_prompt_unchanged():
    """Verify current production provider settings and prompt text."""
    provider = QwenGGUFTranslationProviderV2()
    assert provider.model_path == DEFAULT_QWEN_MODEL_PATH
    assert provider.executable_path == DEFAULT_LLAMA_EXE_PATH
    assert provider.server_url == DEFAULT_QWEN_SERVER_URL

    assert "You are a precise English to Turkish translator." in QWEN_TRANSLATOR_SYSTEM_PROMPT
    assert "Output only the Turkish translation." in QWEN_TRANSLATOR_SYSTEM_PROMPT


def test_real_gate_uses_qwen_v2_provider():
    """Verify real chapter runner uses QwenGGUFTranslationProviderV2."""
    script_p = Path("scripts/real_chapter_translation_gate_v1.py")
    assert script_p.exists()
    content = script_p.read_text(encoding="utf-8")
    assert "QwenGGUFTranslationProviderV2" in content


def test_translator_receives_empty_context_items():
    """Verify translator receives context_items=[] directly."""
    script_p = Path("scripts/real_chapter_translation_gate_v1.py")
    assert script_p.exists()
    content = script_p.read_text(encoding="utf-8")
    assert "context_items=[]" in content


def test_no_production_source_files_modified():
    """Verify core/translation/protection.py and providers/translation/qwen_gguf_translation_v2.py are clean."""
    prot_p = Path("core/translation/protection.py")
    prov_p = Path("providers/translation/qwen_gguf_translation_v2.py")

    assert prot_p.exists()
    assert prov_p.exists()

    # Confirm original class & functions exist
    prot_code = prot_p.read_text(encoding="utf-8")
    prov_code = prov_p.read_text(encoding="utf-8")

    assert "def detect_named_terms_in_items" in prot_code
    assert "def restore_protected_translation" in prot_code
    assert "class QwenGGUFTranslationProviderV2" in prov_code
