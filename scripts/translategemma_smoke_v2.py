"""TranslateGemma 6-Item Real Production Smoke Test.

Executes the official TranslateGemma provider (/completion endpoint + raw template)
over the 6 new smoke items.
Saves results to benchmark_results/translategemma_smoke_v2/:
- results.json
- summary.json
"""
import json
import time
from pathlib import Path

from providers.translation import (
    TranslationInput,
    TranslationItem,
    get_translation_provider,
)

SMOKE_6_ITEMS = [
    (1, "I don't fight with this ability."),
    (2, "Then what is it for?"),
    (3, "It's called Silver Thread."),
    (4, "Silver Thread?"),
    (5, "I can use it to detect movement."),
    (6, "That's more useful than it sounds."),
]


def run_smoke_v2():
    print("=== STARTING TRANSLATEGEMMA 6-ITEM REAL SMOKE TEST ===")

    output_dir = Path("benchmark_results/translategemma_smoke_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in SMOKE_6_ITEMS
    ]

    inp = TranslationInput(items=items, chapter_id="smoke_v2_ch1")

    provider = get_translation_provider()

    t_load_0 = time.perf_counter()
    provider.load()
    load_time = time.perf_counter() - t_load_0

    t_wall_0 = time.perf_counter()
    out = provider.translate(inp)
    wall_time = time.perf_counter() - t_wall_0

    provider.unload()

    m = provider.metrics

    results_json_data = []

    for item_res in out.results:
        results_json_data.append({
            "id": item_res.region_id,
            "source": item_res.source,
            "translation": item_res.translation,
            "validation_warnings": item_res.validation_warnings,
            "requires_review": item_res.requires_review,
        })

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json_data, f, ensure_ascii=False, indent=2)

    summary_data = {
        "execution": {
            "total_items": len(out.results),
            "generation_calls": m.generation_call_count,
            "system_ui_bypass_count": m.system_ui_bypass_count,
            "term_only_bypass_count": m.term_only_bypass_count,
            "retries": m.retries,
            "input_tokens": m.input_token_count,
            "generated_tokens": m.generated_token_count,
            "generation_seconds": round(m.generation_seconds, 2),
            "wall_time_seconds": round(wall_time, 2),
            "load_time_seconds": round(load_time, 2),
        },
        "results": results_json_data,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n=== TRANSLATEGEMMA 6-ITEM SMOKE TEST COMPLETED ===")
    for r in out.results:
        print(f"[{r.region_id:03d}] {r.source} -> {r.translation}")

    print(f"\nGen calls: {m.generation_call_count}, Term bypass: {m.term_only_bypass_count}, Retries: {m.retries}")


if __name__ == "__main__":
    run_smoke_v2()
