"""Regression tests for the cheap, post-run E2E metric summarizer."""

from scripts.audit_e2e_real_chapter1 import summarize_final_region_states


def test_final_unknown_metrics_describe_serialized_state_not_transitions() -> None:
    regions = [
        {"type": "unknown", "status": "skip"},
        {"type": "unknown", "status": "review"},
        {"type": "unknown", "status": "auto"},
        {"type": "dialogue", "status": "auto"},
        {"type": "narration", "status": "review"},
    ]

    metrics = summarize_final_region_states(regions)

    assert metrics == {
        "final_serialized_region_count": 5,
        "final_unknown_skip_regions": 1,
        "final_unknown_review_regions": 1,
        "final_unknown_auto_regions": 1,
    }
    assert sum(value for key, value in metrics.items() if key.startswith("final_unknown_")) == 3
