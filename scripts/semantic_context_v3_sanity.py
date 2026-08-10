"""Semantic Context V3 One-Item Sanity Test (ID 011 HOLD).

Executes one real Qwen GGUF resolver call on ID 011 using the V3 prompt.
"""
import json
import time

from core.translation.semantic_context import (
    SemanticContextRequest,
    render_controlled_english_bridge_prompt,
    resolve_controlled_bridge_with_fallback,
)
from providers.translation.qwen_semantic_resolver import QwenSemanticResolverProvider

# ID 011 Dataset Item
PREVIOUS_CONTEXT = (
    "MIRA CAST FROST CHAIN AT THE LEADER AND HIS TWO GUARDS.",
    "THE ICE CRYSTALS FASTENED AROUND THEIR LIMBS IMMEDIATELY.",
)
TARGET_SOURCE = "Frost Chain can hold three targets at once."
NEXT_CONTEXT = ("NOW IS OUR CHANCE TO ATTACK!",)
NAMED_TERMS = ("Frost Chain",)


def run_sanity():
    print("=== STARTING SEMANTIC CONTEXT V3 SANITY TEST (ID 011) ===")

    req = SemanticContextRequest(
        previous_context=PREVIOUS_CONTEXT,
        target_source=TARGET_SOURCE,
        next_context=NEXT_CONTEXT,
        named_terms=NAMED_TERMS,
    )

    resolver = QwenSemanticResolverProvider(
        server_url="http://127.0.0.1:8082",
        prompt_renderer=render_controlled_english_bridge_prompt,
    )

    t0 = time.perf_counter()
    print("Loading Qwen GGUF resolver on port 8082...")
    resolver.load()
    load_sec = time.perf_counter() - t0
    print(f"Loaded in {load_sec:.2f}s")

    print(f"\nTARGET: {TARGET_SOURCE}")
    outcome = resolve_controlled_bridge_with_fallback(req, resolver.resolve)

    resolver.unload()

    print("\n--- RAW QWEN RESPONSE ---")
    print(outcome.raw_response)
    print("-------------------------")

    print(f"\nRewrite Needed: {outcome.resolution.rewrite_needed if outcome.resolution else None}")
    print(f"Confidence: {outcome.resolution.confidence if outcome.resolution else None}")
    print(f"Controlled Target: {outcome.decision.selected_target}")
    print(f"Rewrite Used: {outcome.decision.rewrite_used}")
    print(f"Rejection Reason: {outcome.decision.rejection_reason}")
    print(f"Validation Failures: {outcome.decision.validation_failures}")

    # Sanity Assertions
    assert outcome.resolver_failed is False, "Qwen resolver failed"
    assert outcome.malformed_json is False, "Malformed JSON"
    assert outcome.resolution is not None, "Resolution is None"
    assert outcome.decision.selected_target != "", "Selected target is empty"
    assert "Frost Chain" in outcome.decision.selected_target, "Frost Chain lost"
    assert "three" in outcome.decision.selected_target or "3" in outcome.decision.selected_target, "Number 3 lost"

    print("\n=== SANITY TEST PASSED ===")


if __name__ == "__main__":
    run_sanity()
