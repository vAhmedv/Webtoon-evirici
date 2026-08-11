"""30-item real Qwen verification for Translation Pipeline Fix V1.

Uses the existing clean real-chapter artifact and the production Qwen V2
provider.  It does not run OCR, context resolution, TranslateGemma, batching, or
micro-batching.  Each retained item receives exactly one chat-completion call.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.translation.profile_discovery import CandidateStore
from core.translation.protection import ProtectedTermMeta, restore_protected_translation
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem, TranslationOutputItem
from providers.translation.qwen_gguf_translation_v2 import (
    DEFAULT_LLAMA_EXE_PATH,
    DEFAULT_QWEN_MODEL_PATH,
    QWEN_TRANSLATOR_SYSTEM_PROMPT,
    QwenGGUFTranslationProviderV2,
    _clean_qwen_output,
)


INPUT_PATH = Path("benchmark_results/real_chapter_translation_gate_v1_clean/valid_story_items.json")
OUTPUT_DIR = Path("benchmark_results/qwen_translation_pipeline_fix_v1")
EXPECTED_OUTPUT_FILES = {
    "input_manifest.json",
    "normalization_examples.json",
    "named_term_examples.json",
    "morphology_examples.json",
    "qwen_results.json",
    "before_after_comparison.json",
    "before_after_comparison.txt",
    "summary.json",
}
MANDATORY_AXE_IDS = (
    "axe_god_chapter1_005",
    "axe_god_chapter1_007",
    "axe_god_chapter1_019",
    "axe_god_chapter1_021",
    "axe_god_chapter1_026",
    "axe_god_chapter1_028",
)


def _write_json(name: str, value: Any) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _profile(series: str) -> SeriesProfile:
    if series.startswith("Axe God"):
        return SeriesProfile(
            series_id="axe_god",
            known_names={
                "LUO TIAN": "Luo Tian",
                "HU SAN": "Hu San",
                "GAO YUAN": "Gao Yuan",
                "YU": "Yu",
            },
            glossary={
                "ABILITY USER": "yetenek kullanıcısı",
                "SECRET REALM": "gizli âlem",
                "SECRET REALM GUIDE": "gizli âlem rehberi",
                "LEVEL 1": "1. seviye",
                "BLACKWIND RAVINE": "Blackwind Ravine",
                "FROST CHAIN": "Frost Chain",
                "AXE GOD": "Balta Tanrısı",
            },
        )
    return SeriesProfile(
        series_id="god_tier_crafter",
        known_names={"ETHAN": "Ethan", "LUCAS": "Lucas"},
        glossary={
            "CRAFTER": "Zanaatkar",
            "GOD-TIER": "Tanrı Seviyesi",
            "REINCARNATED": "Enkarne Olmuş",
            "SYSTEM": "Sistem",
            "SKILL": "Yetenek",
        },
    )


def _is_real_prose(item: dict[str, Any]) -> bool:
    text = str(item["semantic_v3"]["selected_english"]).strip()
    return len(re.findall(r"[A-Za-z]+", text)) >= 4 and bool(re.search(r"[.!?,]", text))


def _spread(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    eligible = [item for item in items if _is_real_prose(item)]
    if len(eligible) < count:
        raise RuntimeError(f"Not enough clean prose items: need {count}, found {len(eligible)}")
    if count == 1:
        return [eligible[len(eligible) // 2]]
    indexes = [round(idx * (len(eligible) - 1) / (count - 1)) for idx in range(count)]
    return [eligible[index] for index in indexes]


def select_gate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 15+15 clean items while covering all ten chapters."""
    by_id = {item["id"]: item for item in items}
    selected = [by_id[item_id] for item_id in MANDATORY_AXE_IDS]

    axe = [item for item in items if item["series"].startswith("Axe God")]
    axe_quotas = {"Chapter 2": 3, "Chapter 3": 2, "Chapter 4": 2, "Chapter 5": 2}
    for chapter, quota in axe_quotas.items():
        selected.extend(_spread([item for item in axe if item["chapter"] == chapter], quota))

    crafter = [item for item in items if item["series"].startswith("Reincarnated")]
    for chapter_number in range(1, 6):
        chapter = f"Chapter {chapter_number}"
        selected.extend(_spread([item for item in crafter if item["chapter"] == chapter], 3))

    if len(selected) != 30 or len({item["id"] for item in selected}) != 30:
        raise RuntimeError("Deterministic selection did not produce 30 unique items")
    if sum(item["series"].startswith("Axe God") for item in selected) != 15:
        raise RuntimeError("Axe God selection is not exactly 15 items")
    return selected


def validate_clean_input(all_items: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    valid_sources: dict[tuple[str, str], set[str]] = {}
    for item in all_items:
        valid_sources.setdefault((item["series"], item["chapter"]), set()).add(
            item["original_accepted_english"]
        )
    for item in selected:
        allowed = valid_sources[(item["series"], item["chapter"])]
        context = list(item.get("previous_context", [])) + list(item.get("next_context", []))
        if any(value not in allowed for value in context):
            raise RuntimeError(f"Dirty or cross-chapter context in {item['id']}")
        if any(re.search(r"\.(?:com|net|org)\b|scanlation|discord\.gg", value, re.I) for value in context):
            raise RuntimeError(f"Watermark context in {item['id']}")


def _single_real_call(
    provider: QwenGGUFTranslationProviderV2,
    prepared: Any,
) -> TranslationOutputItem:
    """Make exactly one real generation call and apply production final guards."""
    provider.metrics.generation_call_count += 1
    try:
        raw, input_tokens, output_tokens, seconds = provider._query_chat_completion(
            prepared.prepared_text
        )
    except Exception as exc:
        return TranslationOutputItem(
            region_id=prepared.item.region_id,
            source=prepared.item.source,
            translation=None,
            raw_model_response=str(exc)[:500],
            validation_warnings=["translation_server_error"],
            requires_review=True,
        )
    provider.metrics.input_token_count += input_tokens
    provider.metrics.generated_token_count += output_tokens
    provider.metrics.generation_seconds += seconds
    cleaned = _clean_qwen_output(raw, prepared.prepared_text)
    if not cleaned:
        return TranslationOutputItem(
            region_id=prepared.item.region_id,
            source=prepared.item.source,
            translation=None,
            raw_model_response=raw[:500],
            validation_warnings=["empty_translation"],
            requires_review=True,
        )
    return provider._finalize_prepared_item(prepared, cleaned, raw)


def _morphology_examples() -> list[dict[str, str]]:
    examples = []
    metas = {
        "ability": ProtectedTermMeta("__WTTERM0001__", "ABILITY USER", "yetenek kullanıcısı", True, False),
        "guide": ProtectedTermMeta("__WTTERM0002__", "SECRET REALM GUIDE", "gizli âlem rehberi", True, False),
        "name": ProtectedTermMeta("__WTTERM0003__", "GAO YUAN", "Gao Yuan", True, True),
    }
    mapping = {meta.sentinel: meta for meta in metas.values()}
    for raw in (
        "__WTTERM0001__DIR",
        "__WTTERM0001__'DIR",
        "__WTTERM0002__İM",
        "__WTTERM0002__'İM",
        "__WTTERM0003__DIR",
    ):
        examples.append({"raw": raw, "restored": restore_protected_translation(raw, mapping)})
    return examples


def _structural_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    # Deliberately case-sensitive: lowercase Turkish -dır/-yim is the repaired,
    # correct output; only the old raw uppercase sentinel attachment is faulty.
    morphology_pattern = re.compile(r"(?:kullanıcısıDIR|rehberiİM)")
    unrestored = [row["id"] for row in results if "__WTTERM" in str(row["final_restored"] or "")]
    morphology = [row["id"] for row in results if morphology_pattern.search(str(row["final_restored"] or ""))]
    false_to_it = [
        row["id"] for row in results
        if any(term.casefold() == "to it" for term in row["detected_named_terms"])
    ]
    return {
        "old_requires_review": sum(bool(row["old_stored_qwen"]["requires_review"]) for row in results),
        "new_requires_review": sum(bool(row["requires_review"]) for row in results),
        "old_english_leakage_warnings": sum(
            "untranslated_source_prose" in row["old_stored_qwen"]["warnings"] for row in results
        ),
        "new_english_leakage_warnings": sum(
            "untranslated_source_prose" in row["warnings"] for row in results
        ),
        "morphology_artifact_ids": morphology,
        "false_to_it_sentinel_ids": false_to_it,
        "unrestored_sentinel_ids": unrestored,
    }


def run() -> None:
    raw_input = INPUT_PATH.read_bytes()
    all_items = json.loads(raw_input.decode("utf-8"))
    selected = select_gate_items(all_items)
    validate_clean_input(all_items, selected)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unknown = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} - EXPECTED_OUTPUT_FILES
    if unknown:
        raise RuntimeError(f"Unexpected files already exist in output directory: {sorted(unknown)}")
    for name in EXPECTED_OUTPUT_FILES:
        (OUTPUT_DIR / name).unlink(missing_ok=True)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "git_head": head,
        "input_path": str(INPUT_PATH),
        "input_sha256": hashlib.sha256(raw_input).hexdigest(),
        "selection_policy": "15 per series; all 10 chapters; six mandatory regressions; deterministic spread",
        "item_count": len(selected),
        "series_counts": {
            "axe_god": sum(item["series"].startswith("Axe God") for item in selected),
            "god_tier_crafter": sum(item["series"].startswith("Reincarnated") for item in selected),
        },
        "chapters": sorted({f"{item['series']} / {item['chapter']}" for item in selected}),
        "items": [
            {"id": item["id"], "series": item["series"], "chapter": item["chapter"]}
            for item in selected
        ],
        "model_path": DEFAULT_QWEN_MODEL_PATH,
        "llama_executable": DEFAULT_LLAMA_EXE_PATH,
        "server_url": "http://127.0.0.1:8083",
        "system_prompt_sha256": hashlib.sha256(QWEN_TRANSLATOR_SYSTEM_PROMPT.encode()).hexdigest(),
        "reasoning": "off",
        "temperature": 0.0,
        "batching": "single_item_no_microbatch",
    }
    _write_json("input_manifest.json", manifest)

    provider = QwenGGUFTranslationProviderV2(server_url="http://127.0.0.1:8083")
    provider.load()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, item in enumerate(selected, 1):
            profile = _profile(item["series"])
            source = item["semantic_v3"]["selected_english"]
            trans_item = TranslationItem(
                region_id=item["id"],
                source=source,
                reading_order=index,
                known_names=profile.get_known_names_list(),
            )
            inp = TranslationInput(
                items=[trans_item],
                profile=profile,
                context_items=[],
                candidate_store=CandidateStore(series_id=profile.series_id),
                chapter_id=f"pipeline_fix_v1_{profile.series_id}",
            )
            prepared = provider._prepare_item(trans_item, inp)
            if prepared.system_ui_translation is not None or prepared.term_only_translation is not None:
                raise RuntimeError(f"Selected item would bypass real generation: {item['id']}")
            output = _single_real_call(provider, prepared)
            placeholder_map = {
                sentinel: asdict(meta) for sentinel, meta in prepared.placeholder_map.items()
            }
            semantic_flags = []
            if item["id"] == "axe_god_chapter1_005" and "param" not in str(output.translation or "").casefold():
                semantic_flags.append("INTRINSIC_MODEL_SEMANTIC_ERROR")
            if item["id"] == "axe_god_chapter1_007":
                semantic_flags.append("HUMAN_REVIEW_IDIOM_LIMITATION_CHECK")
            results.append(
                {
                    "id": item["id"],
                    "series": item["series"],
                    "chapter": item["chapter"],
                    "original_accepted_english": item["original_accepted_english"],
                    "v3_selected_english": source,
                    "old_stored_qwen": {
                        "translation": item["qwen35"]["translation"],
                        "warnings": list(item["qwen35"]["warnings"]),
                        "requires_review": bool(item["qwen35"]["requires_review"]),
                    },
                    "normalized_source": prepared.normalized_source,
                    "detected_named_terms": list(prepared.detected_named_terms),
                    "protected_source": prepared.prepared_text,
                    "placeholder_map": placeholder_map,
                    "new_raw_qwen": output.raw_model_response,
                    "final_restored": output.translation,
                    "warnings": list(output.validation_warnings),
                    "requires_review": output.requires_review,
                    "semantic_flags": semantic_flags,
                }
            )
            print(f"[{index:02d}/30] {item['id']} review={output.requires_review}", flush=True)
    finally:
        provider.unload()

    if provider.metrics.generation_call_count != 30:
        raise RuntimeError(
            f"Expected exactly 30 real Qwen calls, got {provider.metrics.generation_call_count}"
        )

    normalization = [
        {
            "id": row["id"],
            "original": row["v3_selected_english"],
            "normalized": row["normalized_source"],
            "changed": row["v3_selected_english"] != row["normalized_source"],
        }
        for row in results
    ]
    named_terms = [
        {
            "id": row["id"],
            "normalized_source": row["normalized_source"],
            "detected_named_terms": row["detected_named_terms"],
            "protected_source": row["protected_source"],
            "placeholder_map": row["placeholder_map"],
        }
        for row in results
    ]
    comparison = [
        {
            "id": row["id"],
            "source": row["v3_selected_english"],
            "normalized_source": row["normalized_source"],
            "old_translation": row["old_stored_qwen"]["translation"],
            "new_translation": row["final_restored"],
            "old_requires_review": row["old_stored_qwen"]["requires_review"],
            "new_requires_review": row["requires_review"],
            "warnings": row["warnings"],
            "semantic_flags": row["semantic_flags"],
        }
        for row in results
    ]
    metrics = _structural_metrics(results)
    structural_success = not any(
        metrics[key]
        for key in ("morphology_artifact_ids", "false_to_it_sentinel_ids", "unrestored_sentinel_ids")
    )
    summary = {
        "status": "STRUCTURAL_VERIFICATION_PASSED" if structural_success else "STRUCTURAL_VERIFICATION_FAILED",
        "quality_evaluation_ready": structural_success,
        "real_qwen_call_count": provider.metrics.generation_call_count,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "input_tokens": provider.metrics.input_token_count,
        "generated_tokens": provider.metrics.generated_token_count,
        "tokens_per_second": provider.metrics.tokens_per_sec,
        "structural_metrics": metrics,
        "semantic_scores_computed": False,
        "winner_selected": False,
        "production_model_switched": False,
        "notes": [
            "requires_review, leakage, morphology, and sentinel counts are structural diagnostics, not semantic quality scores.",
            "MY MONEY is flagged INTRINSIC_MODEL_SEMANTIC_ERROR when the required first-person possessive para form is absent.",
            "HIT THE JACKPOT remains explicitly queued for human idiom review without a phrase-specific hardcode.",
        ],
    }

    _write_json("normalization_examples.json", normalization)
    _write_json("named_term_examples.json", named_terms)
    _write_json("morphology_examples.json", _morphology_examples())
    _write_json("qwen_results.json", results)
    _write_json("before_after_comparison.json", comparison)
    _write_json("summary.json", summary)
    lines = ["QWEN TRANSLATION PIPELINE FIX V1 — BEFORE / AFTER", "=" * 72]
    for row in comparison:
        lines.extend(
            [
                f"[{row['id']}]",
                f"SOURCE: {row['source']}",
                f"NORMALIZED: {row['normalized_source']}",
                f"OLD: {row['old_translation']}",
                f"NEW: {row['new_translation']}",
                f"OLD_REVIEW={row['old_requires_review']} NEW_REVIEW={row['new_requires_review']}",
                f"WARNINGS={row['warnings']} FLAGS={row['semantic_flags']}",
                "-" * 72,
            ]
        )
    (OUTPUT_DIR / "before_after_comparison.txt").write_text("\n".join(lines), encoding="utf-8")

    actual = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if actual != EXPECTED_OUTPUT_FILES:
        raise RuntimeError(f"Output contract mismatch: {sorted(actual)}")
    if not structural_success:
        raise RuntimeError(f"Structural verification failed: {metrics}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def finalize_existing_outputs() -> None:
    """Recompute deterministic guards from saved results without model calls."""
    results = json.loads((OUTPUT_DIR / "qwen_results.json").read_text(encoding="utf-8"))
    for row in results:
        if row["id"] == "axe_god_chapter1_005":
            row["semantic_flags"] = (
                ["INTRINSIC_MODEL_SEMANTIC_ERROR"]
                if "param" not in str(row["final_restored"] or "").casefold()
                else []
            )
        elif row["id"] == "axe_god_chapter1_007":
            row["semantic_flags"] = ["HUMAN_REVIEW_IDIOM_LIMITATION_CHECK"]
    _write_json("qwen_results.json", results)

    comparison = [
        {
            "id": row["id"],
            "source": row["v3_selected_english"],
            "normalized_source": row["normalized_source"],
            "old_translation": row["old_stored_qwen"]["translation"],
            "new_translation": row["final_restored"],
            "old_requires_review": row["old_stored_qwen"]["requires_review"],
            "new_requires_review": row["requires_review"],
            "warnings": row["warnings"],
            "semantic_flags": row["semantic_flags"],
        }
        for row in results
    ]
    _write_json("before_after_comparison.json", comparison)
    lines = ["QWEN TRANSLATION PIPELINE FIX V1 — BEFORE / AFTER", "=" * 72]
    for row in comparison:
        lines.extend(
            [
                f"[{row['id']}]",
                f"SOURCE: {row['source']}",
                f"NORMALIZED: {row['normalized_source']}",
                f"OLD: {row['old_translation']}",
                f"NEW: {row['new_translation']}",
                f"OLD_REVIEW={row['old_requires_review']} NEW_REVIEW={row['new_requires_review']}",
                f"WARNINGS={row['warnings']} FLAGS={row['semantic_flags']}",
                "-" * 72,
            ]
        )
    (OUTPUT_DIR / "before_after_comparison.txt").write_text("\n".join(lines), encoding="utf-8")

    summary_path = OUTPUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = _structural_metrics(results)
    structural_success = not any(
        metrics[key]
        for key in ("morphology_artifact_ids", "false_to_it_sentinel_ids", "unrestored_sentinel_ids")
    )
    summary["status"] = (
        "STRUCTURAL_VERIFICATION_PASSED" if structural_success else "STRUCTURAL_VERIFICATION_FAILED"
    )
    summary["quality_evaluation_ready"] = structural_success
    summary["structural_metrics"] = metrics
    summary["notes"] = [
        "requires_review, leakage, morphology, and sentinel counts are structural diagnostics, not semantic quality scores.",
        "MY MONEY is flagged INTRINSIC_MODEL_SEMANTIC_ERROR when the required first-person possessive para form is absent.",
        "HIT THE JACKPOT remains explicitly queued for human idiom review without a phrase-specific hardcode.",
    ]
    elapsed = float(summary.get("elapsed_seconds") or 0)
    if elapsed > 0:
        summary["tokens_per_second"] = round(float(summary["generated_tokens"]) / elapsed, 2)
    _write_json("summary.json", summary)

    actual = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if actual != EXPECTED_OUTPUT_FILES or not structural_success:
        raise RuntimeError(f"Existing-output finalization failed: files={sorted(actual)} metrics={metrics}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing_outputs()
    else:
        run()
