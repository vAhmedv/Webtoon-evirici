"""TranslateGemma Quality Gate Benchmark v2 (36 Items).

Executes the official TranslateGemma provider over the 36-item Quality Gate v2 test set.
Saves results to benchmark_results/translategemma_quality_gate_v2/:
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

QUALITY_GATE_V2_36_ITEMS = [
    (1, "My ability isn't meant for fighting."),
    (2, "Then why are you carrying a sword?"),
    (3, "Because swords are useful."),
    (4, "Fair enough."),
    (5, "It's called Glass Thread."),
    (6, "Glass Thread?"),
    (7, "Activate Glass Thread."),
    (8, "Glass Thread breaks if I move too far away."),
    (9, "I learned Silent Pulse from my teacher."),
    (10, "Silent Pulse only works while I'm standing still."),
    (11, "That sounds inconvenient."),
    (12, "You have no idea."),
    (13, "PASSIVE SKILL ACQUIRED: SHADOW SENSE"),
    (14, "TITLE ACQUIRED: ASH WARDEN"),
    (15, "CLASS ADVANCEMENT AVAILABLE"),
    (16, "ABILITY COOLDOWN: 17 SECONDS"),
    (17, "The Guild Master didn't order the attack."),
    (18, "He only told us to be ready."),
    (19, "Not everyone understood the difference."),
    (20, "Apparently you didn't."),
    (21, "Leave the Spirit Stones with the clerk."),
    (22, "Leave her out of this."),
    (23, "We have to leave before sunrise."),
    (24, "Leave the rest to me."),
    (25, "They charged me thirty Spirit Stones for this."),
    (26, "The wolf charged before I could draw my blade."),
    (27, "Who charged the repair bill to my room?"),
    (28, "Ask the innkeeper."),
    (29, "Fantastic. The stairs just collapsed."),
    (30, "You don't sound very happy about it."),
    (31, "What gave it away?"),
    (32, "The screaming helped."),
    (33, "A moment earlier, the chamber had been empty."),
    (34, "Now there was a handprint on the inside of the glass."),
    (35, "Nobody had entered the room."),
    (36, "Then a second handprint appeared beside it."),
]


def run_quality_gate_v2():
    print("=== STARTING TRANSLATEGEMMA QUALITY GATE V2 (36 ITEMS) ===")

    output_dir = Path("benchmark_results/translategemma_quality_gate_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(
        series_id="quality_gate_v2",
        glossary={
            "GUILD MASTER": "Lonca Lideri",
            "SPIRIT STONE": "Ruh Taşı",
        },
    )

    store = CandidateStore(series_id="quality_gate_v2")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in QUALITY_GATE_V2_36_ITEMS
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="quality_gate_v2_ch1",
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
    expected_ids = list(range(1, 37))
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
            "context_ids": [],  # Truthful: direct translation mode does not send reference context
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
            "synthetic_items": f"{len(out.results)}/{len(QUALITY_GATE_V2_36_ITEMS)}",
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

    print("\n=== TRANSLATEGEMMA QUALITY GATE V2 COMPLETED ===")
    print(f"Total items: {len(out.results)}/36")
    print(f"Generation calls: {m.generation_call_count}, System UI bypass: {m.system_ui_bypass_count}, Term-only bypass: {m.term_only_bypass_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    run_quality_gate_v2()
