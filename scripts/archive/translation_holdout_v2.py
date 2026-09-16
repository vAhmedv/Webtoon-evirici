"""New Holdout Quality Test v2 (48 Items).

Executes the production translation pipeline (Qwen3.5-9B GGUF llama-server + CUDA)
with Prompt v2 across 48 holdout items.

Saves output artifacts to benchmark_results/translation_holdout_v2/:
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

HOLDOUT_ITEMS = [
    (1, "Take it easy. Nobody's blaming you."),
    (2, "That's not what I meant."),
    (3, "You always say that after things go wrong."),
    (4, "And somehow you always survive to complain about it."),
    (5, "Commander, the scouts haven't returned yet."),
    (6, "How long have they been gone?"),
    (7, "Nearly three hours."),
    (8, "Then we stop waiting at sunset."),
    (9, "Is that Jian Wei's coat?"),
    (10, "No. It belongs to Mei Ren."),
    (11, "I borrowed it from her this morning."),
    (12, "You could have mentioned that earlier."),
    (13, "The Spirit Gate will remain closed until dawn."),
    (14, "Two Spirit Gates were destroyed during the siege."),
    (15, "Protect the Soul Lantern at all costs."),
    (16, "Without the Soul Lantern, we can't find our way back."),
    (17, "Don't touch that switch."),
    (18, "Why? Is it dangerous?"),
    (19, "Worse. It's connected to the alarm."),
    (20, "And you waited until now to tell me?"),
    (21, "Just perfect."),
    (22, "What?"),
    (23, "Nothing. Keep walking."),
    (24, "You're terrible at pretending you're not angry."),
    (25, "Who charged first?"),
    (26, "The cavalry."),
    (27, "And who charged you fifty coins for this?"),
    (28, "The innkeeper."),
    (29, "Leave the door open."),
    (30, "Leave him alone."),
    (31, "Leave the package with the guard."),
    (32, "We need to leave before sunrise."),
    (33, "I didn't say she stole the key."),
    (34, "I said she knew where it was."),
    (35, "There's a difference."),
    (36, "A rather important one."),
    (37, "Almost everyone made it out."),
    (38, "Not everyone was so lucky."),
    (39, "Only one guard stayed behind."),
    (40, "At least four prisoners are still missing."),
    (41, "If the bridge is still standing, we'll cross there."),
    (42, "If it isn't, we turn back."),
    (43, "Even if they offer us twice the money?"),
    (44, "Especially then."),
    (45, "The room looked untouched, but something about the dust bothered him."),
    (46, "A narrow trail ran from the window to the empty display case."),
    (47, "Whoever had entered hadn't searched the room at random."),
    (48, "They had known exactly what they were looking for."),
]


def run_holdout():
    print("=== STARTING PRODUCTION TRANSLATION HOLDOUT V2 (48 ITEMS) ===")

    output_dir = Path("benchmark_results/translation_holdout_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(
        series_id="synthetic_translation_holdout_v2",
        known_names={
            "JIAN WEI": "Jian Wei",
            "MEI REN": "Mei Ren",
        },
        glossary={
            "SPIRIT GATE": "Ruh Geçidi",
            "SOUL LANTERN": "Ruh Feneri",
            # Intentionally Unrelated Terms (MUST NOT be injected for unrelated items)
            "HEAVENLY TOWER": "Göksel Kule",
            "BLOOD SEAL": "Kan Mührü",
        },
    )

    store = CandidateStore(series_id="synthetic_translation_holdout_v2")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in HOLDOUT_ITEMS
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="holdout_v2_ch1",
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
            chapter_id="holdout_v2_ch1",
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
    expected_ids = list(range(1, 49))
    missing_ids = [i for i in expected_ids if i not in returned_ids]
    duplicate_ids = [i for i in returned_ids if returned_ids.count(i) > 1]
    requires_review_count = sum(1 for r in out.results if r.requires_review)
    fidelity_flagged_count = sum(1 for r in out.results if r.fidelity_flags)

    unrelated_terms = {"HEAVENLY TOWER", "BLOOD SEAL"}
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
            "synthetic_items": f"{len(out.results)}/{len(HOLDOUT_ITEMS)}",
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

    print("\n=== HOLDOUT V2 COMPLETED SUCCESSFULLY ===")
    print(f"Total items: {len(out.results)}/48")
    print(f"Generation calls: {m.generation_call_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    run_holdout()
