"""Semantic Context V3 controlled-English bridge tests."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from core.translation.semantic_context import (
    SemanticContextRequest,
    decide_controlled_bridge,
    parse_controlled_english_resolution,
    render_controlled_english_bridge_prompt,
    resolve_controlled_bridge_with_fallback,
    validate_controlled_target,
)
from providers.translation.qwen_semantic_resolver import QwenSemanticResolverProvider


def _request(
    target: str = "Frost Chain can hold three targets at once.",
    *,
    previous: tuple[str, ...] = ("The enemy mages are preparing a counter-attack.",),
    following: tuple[str, ...] = ("Use it now before they cast their spell.",),
    named_terms: tuple[str, ...] = ("Frost Chain",),
) -> SemanticContextRequest:
    return SemanticContextRequest(
        previous_context=previous,
        target_source=target,
        next_context=following,
        named_terms=named_terms,
    )


def _v3_json(
    controlled_target: str,
    *,
    rewrite_needed: bool = True,
    confidence: float = 0.95,
    question_word: str | None = None,
    tense_aspect: str | None = "modal capability",
) -> str:
    return json.dumps(
        {
            "rewrite_needed": rewrite_needed,
            "confidence": confidence,
            "risk_types": ["lexical_sense"] if rewrite_needed else [],
            "semantic_notes": [
                {
                    "span": "hold",
                    "resolved_meaning": "keep from moving",
                    "evidence": "Nearby context shows immobilization.",
                }
            ],
            "question_word": question_word,
            "tense_aspect": tense_aspect,
            "referents": [],
            "controlled_target": controlled_target,
        }
    )


def _decision(request: SemanticContextRequest, raw: str):
    return decide_controlled_bridge(
        request,
        parse_controlled_english_resolution(raw),
    )


def test_v3_prompt_renders_controlled_english_bridge() -> None:
    prompt = render_controlled_english_bridge_prompt(_request())
    assert "English controlled-language bridge" in prompt
    assert "keep three targets from moving at the same time" in prompt
    assert "Who added the cost of the meal to my room bill?" in prompt
    assert '"controlled_target"' in prompt
    assert "You do NOT translate into Turkish." in prompt


def test_safe_controlled_english_rewrite_is_accepted() -> None:
    request = _request("Frost Chain can hold three targets at once.")
    decision = _decision(
        request,
        _v3_json("Frost Chain can keep three targets from moving at the same time."),
    )
    assert decision.rewrite_used is True
    assert decision.selected_target == "Frost Chain can keep three targets from moving at the same time."
    assert decision.validation_failures == ()


def test_rewrite_needed_false_leaves_original_byte_identical() -> None:
    request = _request("The spell wore off.")
    decision = _decision(
        request,
        _v3_json("The spell wore off.", rewrite_needed=False),
    )
    assert decision.rewrite_used is False
    assert decision.selected_target == "The spell wore off."


def test_malformed_json_fallback_returns_original() -> None:
    request = _request()
    outcome = resolve_controlled_bridge_with_fallback(
        request,
        lambda req: "{invalid_json",
    )
    assert outcome.decision.rewrite_used is False
    assert outcome.decision.selected_target == request.target_source
    assert outcome.malformed_json is True


def test_low_confidence_fallback_returns_original() -> None:
    request = _request()
    decision = _decision(
        request,
        _v3_json("Frost Chain can keep three targets from moving at the same time.", confidence=0.5),
    )
    assert decision.rewrite_used is False
    assert decision.selected_target == request.target_source
    assert decision.rejection_reason == "low_confidence"


def test_named_term_loss_rejected() -> None:
    request = _request("Frost Chain can hold three targets at once.", named_terms=("Frost Chain",))
    decision = _decision(
        request,
        _v3_json("Ice Magic can keep three targets from moving at the same time."),
    )
    assert decision.rewrite_used is False
    assert "named_term_loss" in decision.validation_failures


def test_number_changed_rejected() -> None:
    request = _request("Frost Chain can hold three targets at once.")
    decision = _decision(
        request,
        _v3_json("Frost Chain can keep five targets from moving at the same time."),
    )
    assert decision.rewrite_used is False
    assert "number_changed" in decision.validation_failures


def test_who_to_who_accepted() -> None:
    request = _request("Who charged the meal to my room?", named_terms=())
    decision = _decision(
        request,
        _v3_json("Who added the cost of the meal to my room bill?", question_word="who"),
    )
    assert decision.rewrite_used is True
    assert decision.selected_target == "Who added the cost of the meal to my room bill?"


def test_who_to_how_rejected() -> None:
    request = _request("Who charged the meal to my room?", named_terms=())
    decision = _decision(
        request,
        _v3_json("How was the meal charged to my room bill?", question_word="how"),
    )
    assert decision.rewrite_used is False
    assert "question_type_changed" in decision.validation_failures


def test_actual_tense_change_rejected() -> None:
    request = _request("They have been waiting since dawn.", named_terms=())
    decision = _decision(
        request,
        _v3_json("They waited since dawn."),
    )
    assert decision.rewrite_used is False
    assert "tense_aspect_changed" in decision.validation_failures


def test_pronoun_expansion_does_not_trigger_false_tense_rejection() -> None:
    request = _request("I don't trust it.", named_terms=())
    decision = _decision(
        request,
        _v3_json("I don't trust the cracked compass."),
    )
    assert decision.rewrite_used is True
    assert decision.selected_target == "I don't trust the cracked compass."


def test_lexical_negation_reform_is_not_automatically_rejected() -> None:
    request = _request("The rest of the team are no pushovers either.", named_terms=())
    decision = _decision(
        request,
        _v3_json("The rest of the team are also not easy to defeat."),
    )
    assert decision.rewrite_used is True
    assert decision.selected_target == "The rest of the team are also not easy to defeat."


def test_uncertain_polarity_equivalence_falls_back_conservatively() -> None:
    request = _request("The rest of the team are no pushovers either.", named_terms=())
    failures, validator_uncertain = validate_controlled_target(
        request.target_source,
        "The rest of the team are also hard to defeat.",
    )
    assert "polarity_changed" in failures
    assert validator_uncertain is True


def test_moderate_expansion_allowed() -> None:
    request = _request("Frost Chain can hold three targets at once.")
    failures, _ = validate_controlled_target(
        request.target_source,
        "Frost Chain can keep three targets from moving at the exact same time.",
    )
    assert "controlled_target_too_long" not in failures


def test_paragraph_or_context_dump_rejected() -> None:
    request = _request("Frost Chain can hold three targets at once.")
    dump = "Frost Chain can keep three targets from moving. This is because Frost Chain is an ice spell that immobilizes enemies in a 10 meter radius when cast by a level 5 mage."
    failures, _ = validate_controlled_target(
        request.target_source,
        dump,
    )
    assert "controlled_target_too_long" in failures


def test_turkish_output_rejected() -> None:
    request = _request("Frost Chain can hold three targets at once.")
    failures, _ = validate_controlled_target(
        request.target_source,
        "Frost Chain üç hedefi aynı anda tutabilir.",
    )
    assert "controlled_target_not_english" in failures


def test_gguf_backend_retained() -> None:
    resolver = QwenSemanticResolverProvider(managed=False)
    assert resolver.backend == "qwen3.5-9b-gguf-llamacpp"
    assert "Transformers" not in resolver.backend
