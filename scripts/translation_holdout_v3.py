"""New Holdout Quality Test v3 (36 Items).

Executes the production translation pipeline (Qwen3.5-9B GGUF llama-server + CUDA)
with Compact System Prompt v3 across 36 holdout items.

Saves output artifacts to benchmark_results/translation_holdout_v3/:
- results.json
- results.txt
- summary.json
"""
import json
import time
from pathlib import Path

from core.translation.profile_discovery import CandidateStore, record_validated_term_observations
from core.translation.series_profile import SeriesProfile
from providers.translation import (
    TranslationInput,
    TranslationItem,
    get_translation_provider,
)

HOLDOUT_ITEMS_V3 = [
    (1, "Easy. I only asked a question."),
    (2, "Then ask a better one."),
    (3, "You're in a good mood today."),
    (4, "Don't ruin it."),
    (5, "Sir, the western gate is still open."),
    (6, "I ordered it closed an hour ago."),
    (7, "The mechanism is jammed."),
    (8, "Then get someone who can fix it."),
    (9, "Did Arin Sol leave this note?"),
    (10, "It has her seal."),
    (11, "That doesn't prove she wrote it."),
    (12, "No, but it proves she was here."),
    (13, "The Moonwell is dry."),
    (14, "It hasn't been dry in three hundred years."),
    (15, "Then something changed underground."),
    (16, "Or someone changed it."),
    (17, "That was clever."),
    (18, "Was that a compliment?"),
    (19, "Don't get used to it."),
    (20, "Too late."),
    (21, "Put the knife down."),
    (22, "I'm not holding a knife."),
    (23, "...Right."),
    (24, "Should I be worried that you had to check?"),
    (25, "Everyone except the captain returned."),
    (26, "Almost nobody was injured."),
    (27, "Only two horses are missing."),
    (28, "We still have at least an hour."),
    (29, "I didn't tell him to open the vault."),
    (30, "I told him not to leave it unguarded."),
    (31, "Those are very different instructions."),
    (32, "Apparently not to him."),
    (33, "The corridor seemed shorter on the way back."),
    (34, "Perhaps it was only the darkness, but the walls felt closer than before."),
    (35, "Somewhere behind them, metal scraped slowly across stone."),
    (36, "Nobody turned around."),
]


def run_holdout_v3():
    print("=== STARTING PRODUCTION TRANSLATION HOLDOUT V3 (36 ITEMS) ===")

    output_dir = Path("benchmark_results/translation_holdout_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(
        series_id="synthetic_translation_holdout_v3",
        known_names={
            "ARIN SOL": "Arin Sol",
        },
        glossary={
            "MOONWELL": "Ay Pınarı",
            # Intentionally Unrelated Terms (MUST NOT be injected for unrelated items)
            "SUN TEMPLE": "Güneş Tapınağı",
            "VOID KEY": "Boşluk Anahtarı",
        },
    )

    store = CandidateStore(series_id="synthetic_translation_holdout_v3")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in HOLDOUT_ITEMS_V3
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="holdout_v3_ch1",
    )

    provider = get_translation_provider(backend="gguf")

    t_load_0 = time.perf_counter()
    provider.load()
    load_time = time.perf_counter() - t_load_0

    t_wall_0 = time.perf_counter()
    out = provider.translate(inp)
    wall_time = time.perf_counter() - t_wall_0

    # Record validated observations into benchmark CandidateStore
    for item_res in out.results:
        record_validated_term_observations(
            candidate_store=store,
            chapter_id="holdout_v3_ch1",
            region_id=item_res.region_id,
            source_text=item_res.source,
            translated_text=item_res.translation or "",
            raw_term_usages=item_res.term_usages,
            term_id_map=item_res.term_id_map,
            fidelity_flags=item_res.fidelity_flags,
            requires_review=item_res.requires_review,
        )

    provider.unload()

    m = provider.metrics

    returned_ids = [r.region_id for r in out.results]
    expected_ids = list(range(1, 37))
    missing_ids = [i for i in expected_ids if i not in returned_ids]
    duplicate_ids = [i for i in returned_ids if returned_ids.count(i) > 1]
    requires_review_count = sum(1 for r in out.results if r.requires_review)
    fidelity_flagged_count = sum(1 for r in out.results if r.fidelity_flags)

    unrelated_terms = {"SUN TEMPLE", "VOID KEY"}
    unrelated_leaked = False
    for res in out.results:
        for src_t in res.term_id_map.values():
            if src_t.strip().upper() in unrelated_terms:
                unrelated_leaked = True

    cand_counts = {"discovered": 0, "provisional": 0, "ready_for_review": 0, "approved": 0}
    for c in store.candidates.values():
        st = c.status
        if st in cand_counts:
            cand_counts[st] += 1
        elif st == "confirmed":
            cand_counts["approved"] += 1

    results_json_data = []
    items_by_id = {item.region_id: item.source for item in items}

    for item_res in out.results:
        rid = item_res.region_id
        ctx_ids = [i for i in range(max(1, rid - 3), rid)]

        app_terms_used = {}
        for k, v in profile.known_names.items():
            if k in item_res.source.upper():
                app_terms_used[k] = v
        for k, v in profile.glossary.items():
            if k in item_res.source.upper() and k not in unrelated_terms:
                app_terms_used[k] = v

        results_json_data.append({
            "id": rid,
            "source": item_res.source,
            "translation": item_res.translation,
            "context_ids": ctx_ids,
            "approved_terms": app_terms_used,
            "validation_warnings": item_res.validation_warnings,
            "fidelity_flags": item_res.fidelity_flags,
            "requires_review": item_res.requires_review,
            "term_usages": item_res.term_usages,
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
        results_txt_lines.append("")

        results_txt_lines.append(f"WARNINGS: {item_data['validation_warnings']}")
        results_txt_lines.append(f"FIDELITY FLAGS: {item_data['fidelity_flags']}")
        results_txt_lines.append(f"REQUIRES REVIEW: {item_data['requires_review']}")
        results_txt_lines.append("-" * 50 + "\n")

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results_txt_lines))

    summary_data = {
        "execution": {
            "synthetic_items": f"{len(out.results)}/{len(HOLDOUT_ITEMS_V3)}",
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
            "json_failures": sum(1 for r in out.results if "json_parse_failure" in r.validation_warnings),
            "requires_review_count": requires_review_count,
            "fidelity_flagged_count": fidelity_flagged_count,
            "unrelated_glossary_leaked": unrelated_leaked,
        },
        "termbase": {
            "discovered": cand_counts["discovered"],
            "provisional": cand_counts["provisional"],
            "ready_for_review": cand_counts["ready_for_review"],
            "approved_automatically": cand_counts["approved"],
        },
        "paths": {
            "results_json": str(output_dir / "results.json"),
            "results_txt": str(output_dir / "results.txt"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n=== HOLDOUT V3 COMPLETED SUCCESSFULLY ===")
    print(f"Total items: {len(out.results)}/36")
    print(f"Generation calls: {m.generation_call_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    run_holdout_v3()
