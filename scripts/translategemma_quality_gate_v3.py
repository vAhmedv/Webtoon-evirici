"""TranslateGemma Quality Gate Benchmark v3 (20 Items).

Executes the official TranslateGemma provider (/completion + raw template) over the 20-item Quality Gate v3 test set.
Saves results to benchmark_results/translategemma_quality_gate_v3/:
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

QUALITY_GATE_V3_20_ITEMS = [
    (1, "I wasn't trying to stop you."),
    (2, "Then what were you trying to do?"),
    (3, "Buy us some time."),
    (4, "You could have warned me."),
    (5, "It's called Hollow Step."),
    (6, "Hollow Step?"),
    (7, "Activate Hollow Step."),
    (8, "Hollow Step only works once every ten seconds."),
    (9, "I learned Iron Veil from the old master."),
    (10, "Iron Veil reduces the force of incoming attacks."),
    (11, "That would have been useful five minutes ago."),
    (12, "I know."),
    (13, "PASSIVE SKILL ACQUIRED: BLOOD SENSE"),
    (14, "TITLE ACQUIRED: PALE HUNTER"),
    (15, "CLASS ADVANCEMENT AVAILABLE"),
    (16, "ABILITY COOLDOWN: 12 SECONDS"),
    (17, "Leave the Spirit Stones here."),
    (18, "Leave him out of this."),
    (19, "They charged us forty Spirit Stones."),
    (20, "The beast charged before we could move."),
]


def run_quality_gate_v3():
    print("=== STARTING TRANSLATEGEMMA QUALITY GATE V3 (20 ITEMS) ===")

    output_dir = Path("benchmark_results/translategemma_quality_gate_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(
        series_id="quality_gate_v3",
        glossary={
            "SPIRIT STONE": "Ruh Taşı",
        },
    )

    store = CandidateStore(series_id="quality_gate_v3")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in QUALITY_GATE_V3_20_ITEMS
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="quality_gate_v3_ch1",
    )

    provider = get_translation_provider()

    t_load_0 = time.perf_counter()
    provider.load()
    load_time = time.perf_counter() - t_load_0

    t_wall_0 = time.perf_counter()
    out = provider.translate(inp)
    wall_time = time.perf_counter() - t_wall_0

    provider.unload()

    m = provider.metrics

    returned_ids = [r.region_id for r in out.results]
    expected_ids = list(range(1, 21))
    missing_ids = [i for i in expected_ids if i not in returned_ids]
    duplicate_ids = [i for i in returned_ids if returned_ids.count(i) > 1]
    empty_outputs = [r.region_id for r in out.results if not r.translation]

    results_json_data = []

    for item_res in out.results:
        rid = item_res.region_id

        app_terms_used = {}
        for k, v in profile.glossary.items():
            if k in item_res.source.upper():
                app_terms_used[k] = v

        results_json_data.append({
            "id": rid,
            "source": item_res.source,
            "translation": item_res.translation,
            "context_ids": [],
            "nearby_source_ids_for_human_review": [i for i in range(max(1, rid - 3), rid)],
            "approved_terms": app_terms_used,
            "validation_warnings": item_res.validation_warnings,
            "requires_review": item_res.requires_review,
        })

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json_data, f, ensure_ascii=False, indent=2)

    results_txt_lines = []
    for item_data in results_json_data:
        rid = item_data["id"]
        results_txt_lines.append(f"[{rid:03d}]\n")

        app_terms = item_data["approved_terms"]
        if app_terms:
            results_txt_lines.append("APPROVED TERMS:")
            for k, v in app_terms.items():
                results_txt_lines.append(f"{k} => {v}")
            results_txt_lines.append("")

        results_txt_lines.append("SOURCE:")
        results_txt_lines.append(item_data["source"])
        results_txt_lines.append("")

        results_txt_lines.append("TURKISH:")
        results_txt_lines.append(item_data["translation"] or "<FAILED>")
        results_txt_lines.append("\n" + "-" * 50 + "\n")

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results_txt_lines))

    summary_data = {
        "execution": {
            "synthetic_items": f"{len(out.results)}/{len(QUALITY_GATE_V3_20_ITEMS)}",
            "generation_calls": m.generation_call_count,
            "system_ui_bypass_count": m.system_ui_bypass_count,
            "term_only_bypass_count": m.term_only_bypass_count,
            "retries": m.retries,
            "input_tokens": m.input_token_count,
            "generated_tokens": m.generated_token_count,
            "generation_seconds": round(m.generation_seconds, 2),
            "wall_time_seconds": round(wall_time, 2),
            "average_tok_per_sec": round(m.tokens_per_sec, 2),
            "load_time_seconds": round(load_time, 2),
            "backend": provider.name,
            "model": m.translation_model,
        },
        "structural_validation": {
            "missing_ids": missing_ids,
            "duplicate_ids": duplicate_ids,
            "empty_outputs": empty_outputs,
            "requires_review_count": sum(1 for r in out.results if r.requires_review),
        },
        "paths": {
            "results_json": str(output_dir / "results.json"),
            "results_txt": str(output_dir / "results.txt"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n=== TRANSLATEGEMMA QUALITY GATE V3 COMPLETED ===")
    print(f"Total items: {len(out.results)}/20")
    print(f"Generation calls: {m.generation_call_count}, System UI bypass: {m.system_ui_bypass_count}, Term-only bypass: {m.term_only_bypass_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    run_quality_gate_v3()
