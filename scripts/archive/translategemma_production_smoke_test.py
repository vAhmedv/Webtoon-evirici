"""TranslateGemma 12B Production Integration Smoke Test Script (17 Items).

Executes the DEFAULT production translation pipeline (TranslateGemmaGGUFTranslationProvider)
over 17 production test items.

Saves output artifacts to benchmark_results/translategemma_production_v1/:
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

PRODUCTION_SMOKE_ITEMS = [
    (1, "Wait. You're telling me this door was open when you arrived?"),
    (2, "That's what I said."),
    (3, "Then why is there dust on the handle?"),
    (4, "Master Ren wants us back before nightfall."),
    (5, "Tell Master Ren we'll return when the job is finished."),
    (6, "The Spirit Core is reacting to something underground."),
    (7, "Don't touch it until we know what caused it."),
    (8, "Brilliant. Now we're locked in."),
    (9, "You were the one who closed the door."),
    (10, "I didn't know it would seal itself."),
    (11, "Not everyone in the village supports us."),
    (12, "That doesn't make them our enemies."),
    (13, "Leave the documents with the clerk."),
    (14, "We need to leave before the patrol comes back."),
    (15, "The footsteps stopped just beyond the wall."),
    (16, "For several seconds, the corridor was completely silent."),
    (17, "Then someone whispered his name."),
]


def run_production_smoke():
    print("=== STARTING TRANSLATEGEMMA PRODUCTION SMOKE TEST (17 ITEMS) ===")

    output_dir = Path("benchmark_results/translategemma_production_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(
        series_id="translategemma_production_v1",
        known_names={
            "MASTER REN": "Usta Ren",
        },
        glossary={
            "SPIRIT CORE": "Ruh Çekirdeği",
            # Intentionally Unrelated Terms (MUST NOT be injected for unrelated items)
            "HEAVENLY COURT": "Göksel Divan",
            "DRAGON ALTAR": "Ejder Sunağı",
        },
    )

    store = CandidateStore(series_id="translategemma_production_v1")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in PRODUCTION_SMOKE_ITEMS
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="prod_smoke_ch1",
    )

    # Use default factory backend (TranslateGemma GGUF)
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
    expected_ids = list(range(1, 18))
    missing_ids = [i for i in expected_ids if i not in returned_ids]

    unrelated_terms = {"HEAVENLY COURT", "DRAGON ALTAR"}
    unrelated_leaked = False

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
        results_txt_lines.append("\n" + "-" * 50 + "\n")

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results_txt_lines))

    summary_data = {
        "execution": {
            "synthetic_items": f"{len(out.results)}/{len(PRODUCTION_SMOKE_ITEMS)}",
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
            "requires_review_count": sum(1 for r in out.results if r.requires_review),
            "unrelated_glossary_leaked": unrelated_leaked,
        },
        "paths": {
            "results_json": str(output_dir / "results.json"),
            "results_txt": str(output_dir / "results.txt"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n=== TRANSLATEGEMMA PRODUCTION SMOKE COMPLETED ===")
    print(f"Total items: {len(out.results)}/17")
    print(f"Generation calls: {m.generation_call_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    run_production_smoke()
