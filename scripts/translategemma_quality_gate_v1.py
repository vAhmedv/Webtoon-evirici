"""TranslateGemma New Real Quality Gate Benchmark (40 Items).

Executes the official TranslateGemma provider over the 40-item test set.
Saves results to benchmark_results/translategemma_quality_gate_v1/:
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

QUALITY_GATE_40_ITEMS = [
    (1, "I don't have a combat ability."),
    (2, "That doesn't mean I'm helpless."),
    (3, "What does your ability do?"),
    (4, "It's called Phantom Thread."),
    (5, "Phantom Thread?"),
    (6, "I can attach it to anything I can see."),
    (7, "Activate Phantom Thread."),
    (8, "Don't use Phantom Thread again until your Mana Core stabilizes."),
    (9, "I learned Iron Pulse from my brother."),
    (10, "Iron Pulse isn't supposed to hurt the user."),
    (11, "Then I'm obviously doing it wrong."),
    (12, "That's one way to put it."),
    (13, "PASSIVE SKILL ACQUIRED: ECHO SENSE"),
    (14, "TITLE ACQUIRED: GRAVE WALKER"),
    (15, "CLASS ADVANCEMENT AVAILABLE"),
    (16, "ABILITY COOLDOWN: 24 SECONDS"),
    (17, "The Guild Master ordered everyone out."),
    (18, "I never said the Guild Master approved the plan."),
    (19, "Only Inner Disciples are allowed inside the Secret Realm."),
    (20, "Not every Inner Disciple comes back."),
    (21, "Leave the Mana Stones on the table."),
    (22, "Leave him out of this."),
    (23, "We need to leave before dawn."),
    (24, "Leave it to me."),
    (25, "They charged us fifty Mana Stones."),
    (26, "The beast charged before I could move."),
    (27, "Who charged this expense to my account?"),
    (28, "Ask the accountant."),
    (29, "You're kidding."),
    (30, "I wish I were."),
    (31, "Oh, wonderful. The ceiling is collapsing."),
    (32, "You have a strange definition of wonderful."),
    (33, "I didn't say she opened the gate."),
    (34, "I said she knew who did."),
    (35, "Almost everyone believed her."),
    (36, "Not everyone was convinced."),
    (37, "The hallway had been empty a moment earlier."),
    (38, "Now a single wet footprint marked the dust."),
    (39, "Nothing moved, but something behind the wall was breathing."),
    (40, "Then the breathing stopped."),
]


def run_quality_gate():
    print("=== STARTING TRANSLATEGEMMA QUALITY GATE V1 (40 ITEMS) ===")

    output_dir = Path("benchmark_results/translategemma_quality_gate_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(
        series_id="quality_gate_v1",
        glossary={
            "ABILITY USER": "Yetenek Kullanıcısı",
            "MANA CORE": "Mana Çekirdeği",
            "GUILD MASTER": "Lonca Lideri",
            "INNER DISCIPLE": "İç Mürit",
            "SECRET REALM": "Gizli Diyar",
        },
    )

    store = CandidateStore(series_id="quality_gate_v1")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in QUALITY_GATE_40_ITEMS
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="quality_gate_ch1",
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
    expected_ids = list(range(1, 41))
    missing_ids = [i for i in expected_ids if i not in returned_ids]
    duplicate_ids = [i for i in returned_ids if returned_ids.count(i) > 1]
    empty_outputs = [r.region_id for r in out.results if not r.translation]

    results_json_data = []
    items_by_id = {item.region_id: item.source for item in items}

    for item_res in out.results:
        rid = item_res.region_id
        ctx_ids = [i for i in range(max(1, rid - 3), rid)]

        app_terms_used = {}
        for k, v in profile.glossary.items():
            if k in item_res.source.upper():
                app_terms_used[k] = v

        results_json_data.append({
            "id": rid,
            "source": item_res.source,
            "translation": item_res.translation,
            "context_ids": ctx_ids,
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

        ctx_ids = item_data["context_ids"]
        if ctx_ids:
            results_txt_lines.append("CONTEXT:")
            for cid in ctx_ids:
                if cid in items_by_id:
                    results_txt_lines.append(f"{cid:03d} | {items_by_id[cid]}")
            results_txt_lines.append("")

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
            "synthetic_items": f"{len(out.results)}/{len(QUALITY_GATE_40_ITEMS)}",
            "generation_calls": m.generation_call_count,
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

    print("\n=== TRANSLATEGEMMA QUALITY GATE COMPLETED ===")
    print(f"Total items: {len(out.results)}/40")
    print(f"Generation calls: {m.generation_call_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    run_quality_gate()
