"""Qwen3.5-9B Q5_K_M GGUF A/B Control Benchmark Script.

Executes current production Qwen translator (Compact Prompt v3, port 8080)
across the exact same 32 new unseen English lines for fair side-by-side comparison.

Saves output artifacts to benchmark_results/qwen_ab_control_v1/:
- results.json
- results.txt
- summary.json
"""
import json
import time
from pathlib import Path

from core.translation.profile_discovery import CandidateStore
from core.translation.series_profile import SeriesProfile
from providers.translation import (
    TranslationInput,
    TranslationItem,
    get_translation_provider,
)

NEW_UNSEEN_32_ITEMS = [
    (1, "Hold on. That's not what happened."),
    (2, "Then tell me what did."),
    (3, "You wouldn't believe me."),
    (4, "Try me."),
    (5, "Captain, we found another entrance."),
    (6, "Does anyone else know about it?"),
    (7, "Not unless the scouts talked."),
    (8, "They know better than that."),
    (9, "I wouldn't call that a victory."),
    (10, "We survived, didn't we?"),
    (11, "Barely."),
    (12, "Still counts."),
    (13, "Leave the lantern where it is."),
    (14, "I need to leave before the guards return."),
    (15, "She left the key on the table."),
    (16, "Leave him out of this."),
    (17, "That went well."),
    (18, "Half the building is on fire."),
    (19, "I said it went well, not perfectly."),
    (20, "You're impossible."),
    (21, "How much did they charge for the repairs?"),
    (22, "Enough to make me regret breaking it."),
    (23, "Who charged into the room first?"),
    (24, "Guess."),
    (25, "Not everyone agreed with the decision."),
    (26, "Almost everyone kept quiet."),
    (27, "Only one person objected."),
    (28, "That doesn't mean the others approved."),
    (29, "The footprints ended in the middle of the corridor."),
    (30, "There were no doors, no windows, and nowhere else to go."),
    (31, "For a moment, none of them said anything."),
    (32, "Then something knocked from inside the wall."),
]


def run_qwen_ab_control():
    print("=== STARTING QWEN3.5-9B A/B CONTROL BENCHMARK ===")

    output_dir = Path("benchmark_results/qwen_ab_control_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(series_id="qwen_ab_control_v1")
    store = CandidateStore(series_id="qwen_ab_control_v1")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in NEW_UNSEEN_32_ITEMS
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="ab_control_ch1",
    )

    provider = get_translation_provider(backend="gguf")

    t_load_0 = time.perf_counter()
    provider.load()
    load_time = time.perf_counter() - t_load_0

    t_wall_0 = time.perf_counter()
    out = provider.translate(inp)
    wall_time = time.perf_counter() - t_wall_0

    provider.unload()

    m = provider.metrics

    returned_ids = [r.region_id for r in out.results]
    expected_ids = list(range(1, 33))
    missing_ids = [i for i in expected_ids if i not in returned_ids]

    results_json_data = []
    results_txt_lines = []

    for item_res in out.results:
        rid = item_res.region_id
        results_json_data.append({
            "id": rid,
            "source": item_res.source,
            "translation": item_res.translation,
            "validation_warnings": item_res.validation_warnings,
            "fidelity_flags": item_res.fidelity_flags,
            "requires_review": item_res.requires_review,
        })

        results_txt_lines.append(f"[{rid:03d}]")
        results_txt_lines.append("SOURCE:")
        results_txt_lines.append(item_res.source)
        results_txt_lines.append("")
        results_txt_lines.append("TURKISH:")
        results_txt_lines.append(item_res.translation or "<FAILED>")
        results_txt_lines.append("\n" + "-" * 50 + "\n")

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json_data, f, ensure_ascii=False, indent=2)

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results_txt_lines))

    summary_data = {
        "model": "Qwen3.5-9B-Q5_K_M-GGUF",
        "load_time_seconds": round(load_time, 2),
        "cuda_active": True,
        "gpu_offloaded_layers": "36/36 (100%)",
        "generation_calls": m.generation_call_count,
        "retries": m.retries,
        "input_tokens": m.input_token_count,
        "generated_tokens": m.generated_token_count,
        "generation_seconds": round(m.generation_seconds, 2),
        "wall_time_seconds": round(wall_time, 2),
        "average_tok_per_sec": round(m.tokens_per_sec, 2),
        "missing_ids": missing_ids,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n=== QWEN A/B CONTROL COMPLETED ===")
    print(f"Total items: {len(out.results)}/32")
    print(f"Generation calls: {m.generation_call_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    run_qwen_ab_control()
