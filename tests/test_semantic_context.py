from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from core.translation.semantic_context import (
    LocalContextRegion,
    SemanticContextRequest,
    decide_clarification,
    parse_semantic_resolution,
    resolve_with_fallback,
    select_local_context,
)
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.translategemma_gguf_translation import (
    TranslateGemmaGGUFTranslationProvider,
)


def _resolver_json(
    clarified_target: str,
    *,
    ambiguous: bool = True,
    confidence: float = 0.95,
    question_type: str | None = None,
) -> str:
    return json.dumps(
        {
            "ambiguous": ambiguous,
            "confidence": confidence,
            "semantic_notes": [
                {
                    "span": "test span",
                    "intended_sense": "context-supported sense",
                    "evidence": "nearby English context",
                }
            ],
            "question_type": question_type,
            "tense_aspect": "preserved",
            "referents": [],
            "register_hint": None,
            "clarified_target": clarified_target,
        }
    )


def _request(
    target: str,
    *,
    named_terms: tuple[str, ...] = (),
) -> SemanticContextRequest:
    return SemanticContextRequest(
        previous_context=("Nearby English context.",),
        target_source=target,
        next_context=(),
        named_terms=named_terms,
    )


def test_strict_json_parsing():
    parsed = parse_semantic_resolution(
        _resolver_json("The chain can restrain three targets.")
    )
    assert parsed.ambiguous is True
    assert parsed.confidence == 0.95
    assert parsed.clarified_target == "The chain can restrain three targets."
    assert parsed.semantic_notes[0].intended_sense == "context-supported sense"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "```json\n{}\n```",
        json.dumps({"ambiguous": True}),
        json.dumps([]),
    ],
)
def test_malformed_resolver_response_falls_back(raw: str):
    request = _request("Keep the original target.")
    outcome = resolve_with_fallback(request, lambda _: raw)
    assert outcome.malformed_json is True
    assert outcome.decision.selected_target == request.target_source
    assert outcome.decision.rejection_reason == "malformed_json"


def test_confidence_threshold_accepts_only_at_or_above_threshold():
    request = _request("The chain can hold them.")
    low = parse_semantic_resolution(
        _resolver_json("The chain can restrain them.", confidence=0.849)
    )
    accepted = parse_semantic_resolution(
        _resolver_json("The chain can restrain them.", confidence=0.85)
    )
    assert decide_clarification(request, low).rejection_reason == "low_confidence"
    assert decide_clarification(request, accepted).clarification_used is True


def test_unchanged_target_bypasses_clarification():
    request = _request("Nothing needs clarification.")
    resolution = parse_semantic_resolution(_resolver_json(request.target_source))
    decision = decide_clarification(request, resolution)
    assert decision.selected_target == request.target_source
    assert decision.clarification_used is False
    assert decision.rejection_reason == "unchanged_target"


def test_low_confidence_falls_back_to_original():
    request = _request("The guard can hold them.")
    outcome = resolve_with_fallback(
        request,
        lambda _: _resolver_json(
            "The guard can restrain them.",
            confidence=0.5,
        ),
    )
    assert outcome.decision.selected_target == request.target_source
    assert outcome.decision.rejection_reason == "low_confidence"


def test_named_term_preservation_rejects_rewrite():
    request = _request(
        "Frost Chain can hold three targets.",
        named_terms=("Frost Chain",),
    )
    resolution = parse_semantic_resolution(
        _resolver_json("Ice Chain can restrain three targets.")
    )
    decision = decide_clarification(request, resolution)
    assert "named_term_loss" in decision.validation_failures
    assert decision.selected_target == request.target_source


def test_number_preservation_rejects_change():
    request = _request("The chain can hold three targets.")
    resolution = parse_semantic_resolution(
        _resolver_json("The chain can restrain two targets.")
    )
    decision = decide_clarification(request, resolution)
    assert "number_changed" in decision.validation_failures


def test_negation_preservation_rejects_change():
    request = _request("Do not open the gate.")
    resolution = parse_semantic_resolution(_resolver_json("Open the gate."))
    decision = decide_clarification(request, resolution)
    assert "polarity_changed" in decision.validation_failures


def test_question_type_preservation_rejects_who_to_how():
    request = _request("Who charged the meal to my room?")
    resolution = parse_semantic_resolution(
        _resolver_json(
            "How was the meal charged to my room?",
            question_type="how",
        )
    )
    decision = decide_clarification(request, resolution)
    assert "question_type_changed" in decision.validation_failures
    assert decision.selected_target == request.target_source


def test_turkish_clarified_target_is_rejected():
    request = _request("He can carry the box.")
    resolution = parse_semantic_resolution(_resolver_json("Kutuyu taşıyabilir."))
    decision = decide_clarification(request, resolution)
    assert "clarified_target_not_english" in decision.validation_failures


def test_all_caps_dialogue_is_not_misclassified_as_a_named_entity():
    request = _request("THE CHAIN CAN HOLD THEM.")
    resolution = parse_semantic_resolution(
        _resolver_json("THE CHAIN CAN RESTRAIN THEM.")
    )
    assert decide_clarification(request, resolution).clarification_used is True


def test_unsupported_new_named_entity_is_rejected():
    request = _request("The guard can restrain them.")
    resolution = parse_semantic_resolution(
        _resolver_json("The guard can restrain them for Zeus.")
    )
    decision = decide_clarification(request, resolution)
    assert "unsupported_named_entity" in decision.validation_failures


def test_empty_clarified_target_is_rejected():
    request = _request("Keep this sentence.")
    resolution = parse_semantic_resolution(_resolver_json(""))
    decision = decide_clarification(request, resolution)
    assert decision.validation_failures == ("empty_clarified_target",)
    assert decision.selected_target == request.target_source


def test_resolver_exception_falls_back_to_original():
    request = _request("Keep this sentence.")

    def failing_resolver(_: SemanticContextRequest) -> str:
        raise RuntimeError("resolver unavailable")

    outcome = resolve_with_fallback(request, failing_resolver)
    assert outcome.resolver_failed is True
    assert outcome.decision.selected_target == request.target_source
    assert outcome.decision.rejection_reason == "resolver_failure"


def test_local_context_filters_non_prose_and_respects_bounds():
    regions = [
        LocalContextRegion(1, 1, "First line.", "dialogue", "scene-a"),
        LocalContextRegion(2, 2, "BANG", "sfx", "scene-a"),
        LocalContextRegion(3, 3, "Second line.", "narration", "scene-a"),
        LocalContextRegion(4, 4, "Target line.", "dialogue", "scene-a"),
        LocalContextRegion(5, 5, "WATERMARK", "watermark", "scene-a"),
        LocalContextRegion(6, 6, "Next line.", "dialogue", "scene-a"),
        LocalContextRegion(7, 7, "Other scene.", "dialogue", "scene-b"),
    ]
    previous, following = select_local_context(regions, 4)
    assert previous == ("First line.", "Second line.")
    assert following == ("Next line.",)


def test_semantic_context_translate_path_keeps_micro_batch_disabled():
    provider = TranslateGemmaGGUFTranslationProvider(
        managed=False,
        micro_batch_enabled=False,
        prompt_variant="minimal_faithful",
    )
    inp = TranslationInput(
        items=[
            TranslationItem(1, "The guard can restrain them.", reading_order=1),
            TranslationItem(2, "The door remained shut.", reading_order=2),
        ],
        chapter_id="semantic-context-unit",
    )
    with (
        patch.object(provider, "_check_health", return_value=True),
        patch.object(
            provider,
            "_query_official_translation",
            return_value=("Çeviri.", 10, 2, 0.01),
        ),
    ):
        output = provider.translate(inp)
    assert len(output.results) == 2
    assert provider.metrics.generation_call_count == 2
    assert provider.metrics.micro_batch_requests == 0
