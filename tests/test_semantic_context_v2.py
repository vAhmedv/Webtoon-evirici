"""Semantic Context V2 translation-risk resolver tests."""
from __future__ import annotations

import json

import pytest

from core.translation.semantic_context import (
    SemanticContextRequest,
    decide_translation_risk_rewrite,
    parse_translation_risk_resolution,
    render_translation_risk_resolver_prompt,
    resolve_translation_risk_with_fallback,
)
from providers.translation.translategemma_gguf_translation import (
    TranslateGemmaGGUFTranslationProvider,
)


def _request(
    target: str = "The lock can secure two panels.",
    *,
    previous: tuple[str, ...] = ("The panels keep swinging open.",),
    following: tuple[str, ...] = ("Use it before the storm arrives.",),
    named_terms: tuple[str, ...] = (),
) -> SemanticContextRequest:
    return SemanticContextRequest(
        previous_context=previous,
        target_source=target,
        next_context=following,
        named_terms=named_terms,
    )


def _resolver_json(
    clarified_target: str,
    *,
    rewrite_needed: bool = True,
    confidence: float = 0.95,
    question_type: str | None = None,
    tense_aspect: str | None = "modal capability",
) -> str:
    return json.dumps(
        {
            "rewrite_needed": rewrite_needed,
            "confidence": confidence,
            "risk_types": ["lexical_sense"] if rewrite_needed else [],
            "semantic_notes": [
                {
                    "span": "secure",
                    "resolved_meaning": "keep fixed in place",
                    "evidence": "Nearby context says the panels keep moving.",
                }
            ],
            "question_type": question_type,
            "tense_aspect": tense_aspect,
            "referents": [],
            "clarified_target": clarified_target,
        }
    )


def _decision(
    request: SemanticContextRequest,
    raw: str,
):
    return decide_translation_risk_rewrite(
        request,
        parse_translation_risk_resolution(raw),
    )


def test_v2_prompt_targets_translation_risk_without_ambiguity_gate() -> None:
    prompt = render_translation_risk_resolver_prompt(_request())

    assert "translation model could plausibly interpret with the wrong meaning" in prompt
    assert "rewrite_needed" in prompt
    assert '"ambiguous"' not in prompt
    assert "You do NOT translate into Turkish." in prompt


def test_safe_high_confidence_rewrite_is_accepted() -> None:
    request = _request("The lock can hold two panels.")

    decision = _decision(
        request,
        _resolver_json("The lock can secure two panels."),
    )

    assert decision.rewrite_used is True
    assert decision.selected_target == "The lock can secure two panels."
    assert decision.validation_failures == ()


def test_rewrite_needed_false_uses_original() -> None:
    request = _request()

    decision = _decision(
        request,
        _resolver_json(request.target_source, rewrite_needed=False),
    )

    assert decision.rewrite_used is False
    assert decision.selected_target == request.target_source
    assert decision.rejection_reason == "rewrite_not_needed"


def test_high_confidence_alone_is_not_enough() -> None:
    request = _request()

    decision = _decision(
        request,
        _resolver_json(
            request.target_source,
            rewrite_needed=False,
            confidence=1.0,
        ),
    )

    assert decision.rewrite_used is False
    assert decision.rejection_reason == "rewrite_not_needed"


def test_low_confidence_rewrite_uses_original() -> None:
    request = _request("The lock can hold two panels.")

    decision = _decision(
        request,
        _resolver_json("The lock can secure two panels.", confidence=0.84),
    )

    assert decision.rewrite_used is False
    assert decision.rejection_reason == "low_confidence"


def test_same_clarified_target_uses_original() -> None:
    request = _request()

    decision = _decision(request, _resolver_json(request.target_source))

    assert decision.rewrite_used is False
    assert decision.rejection_reason == "unchanged_target"


def test_named_term_identity_change_is_rejected() -> None:
    request = _request(
        "Silver Ward can secure two panels.",
        named_terms=("Silver Ward",),
    )

    decision = _decision(
        request,
        _resolver_json("Golden Ward can secure two panels."),
    )

    assert decision.rewrite_used is False
    assert "named_term_loss" in decision.validation_failures


def test_number_change_is_rejected() -> None:
    request = _request()

    decision = _decision(
        request,
        _resolver_json("The lock can secure three panels."),
    )

    assert decision.rewrite_used is False
    assert "number_changed" in decision.validation_failures


def test_polarity_change_is_rejected() -> None:
    request = _request("The lock cannot secure two panels.")

    decision = _decision(
        request,
        _resolver_json("The lock can secure two panels."),
    )

    assert decision.rewrite_used is False
    assert "polarity_changed" in decision.validation_failures


def test_who_to_how_is_rejected() -> None:
    request = _request("Who charged the fee to the account?")

    decision = _decision(
        request,
        _resolver_json(
            "How was the fee charged to the account?",
            question_type="how",
            tense_aspect="simple past passive",
        ),
    )

    assert decision.rewrite_used is False
    assert "question_type_changed" in decision.validation_failures


def test_who_to_who_rewording_is_allowed() -> None:
    request = _request("Who charged the fee to the account?")

    decision = _decision(
        request,
        _resolver_json(
            "Who added the fee to the account?",
            question_type="who",
            tense_aspect="simple past",
        ),
    )

    assert decision.rewrite_used is True
    assert decision.selected_target == "Who added the fee to the account?"


def test_question_to_statement_is_rejected() -> None:
    request = _request("Who charged the fee to the account?")

    decision = _decision(
        request,
        _resolver_json(
            "The clerk added the fee to the account.",
            question_type="who",
            tense_aspect="simple past",
        ),
    )

    assert decision.rewrite_used is False
    assert "question_type_changed" in decision.validation_failures


def test_turkish_clarified_target_is_rejected() -> None:
    request = _request()

    decision = _decision(
        request,
        _resolver_json("Kilit iki paneli sabit tutabilir."),
    )

    assert decision.rewrite_used is False
    assert "clarified_target_not_english" in decision.validation_failures


def test_excessively_long_rewrite_is_rejected() -> None:
    request = _request("The lock can secure two panels.")
    clarified = (
        "The lock can securely and permanently secure two panels against every possible "
        "storm, intruder, accident, vibration, impact, and magical event."
    )

    decision = _decision(request, _resolver_json(clarified))

    assert decision.rewrite_used is False
    assert "clarified_target_too_long" in decision.validation_failures


def test_context_sentence_copy_is_rejected() -> None:
    context_sentence = "The eastern gate was already sealed."
    request = _request(
        "The guard stayed.",
        previous=(context_sentence,),
        following=(),
    )

    decision = _decision(
        request,
        _resolver_json("The guard stayed; the eastern gate was already sealed."),
    )

    assert decision.rewrite_used is False
    assert "context_sentence_copied" in decision.validation_failures


def test_temporal_identity_and_aspect_change_is_rejected() -> None:
    request = _request("They have been waiting since dawn.")

    decision = _decision(
        request,
        _resolver_json(
            "They have been waiting since sunrise.",
            tense_aspect="present perfect progressive",
        ),
    )

    assert decision.rewrite_used is False
    assert "tense_aspect_changed" in decision.validation_failures


def test_json_or_markdown_garbage_is_rejected() -> None:
    request = _request()

    decision = _decision(
        request,
        _resolver_json('```json {"clarified_target": "The lock can secure two panels."} ```'),
    )

    assert decision.rewrite_used is False
    assert "structured_or_control_garbage" in decision.validation_failures


def test_malformed_json_falls_back_to_original() -> None:
    request = _request()

    outcome = resolve_translation_risk_with_fallback(request, lambda _: "not-json")

    assert outcome.malformed_json is True
    assert outcome.decision.selected_target == request.target_source
    assert outcome.decision.rewrite_used is False


def test_qwen_failure_falls_back_to_original() -> None:
    request = _request()

    def failing_resolver(_: SemanticContextRequest) -> str:
        raise RuntimeError("model failure")

    outcome = resolve_translation_risk_with_fallback(request, failing_resolver)

    assert outcome.resolver_failed is True
    assert outcome.decision.selected_target == request.target_source
    assert outcome.decision.rewrite_used is False


def test_v2_schema_is_exact() -> None:
    payload = json.loads(_resolver_json(_request().target_source))
    payload["ambiguous"] = False

    with pytest.raises(ValueError, match="schema_mismatch"):
        parse_translation_risk_resolution(json.dumps(payload))


def test_semantic_context_v2_keeps_micro_batch_disabled() -> None:
    provider = TranslateGemmaGGUFTranslationProvider(
        managed=False,
        micro_batch_enabled=False,
        prompt_variant="minimal_faithful",
    )

    assert provider.micro_batch_enabled is False
    assert provider.metrics.micro_batch_requests == 0
