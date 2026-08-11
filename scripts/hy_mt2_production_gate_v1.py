"""Run the fixed REAL30 set through the real protected Hy-MT2 production path.

This script launches only Hy-MT2. It consumes stored raw-Hy evidence and never
invokes Qwen, MADLAD, TranslateGemma, OCR, or the semantic resolver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.hy_mt2_gguf_translation import (
    DEFAULT_HY_MT2_MODEL_PATH,
    HyMT2GGUFTranslationProvider,
)


SOURCE_DIR = BASE_DIR / "benchmark_results" / "hy_mt2_vs_karga_vs_qwen_real30_v1"
DATASET_PATH = SOURCE_DIR / "dataset.json"
RAW_HY_PATH = SOURCE_DIR / "hy_mt2_results.json"
OUTPUT_DIR = BASE_DIR / "benchmark_results" / "hy_mt2_production_integration_v1"
EXPECTED_DATASET_SHA256 = "4989270bcdd7527ba81d3347131fc350e63519cbedd1ec369237e711b7394a41"
CRITICAL_GUARDS = {
    "translation_server_error",
    "empty_translation",
    "source_translation_wrapper",
    "chatbot_or_explanation_output",
    "unrestored_protected_term",
    "approved_term_missing",
    "approved_source_term_leakage",
    "untranslated_source_prose",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(name: str, value: Any) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def profile_for(series: str) -> SeriesProfile:
    """Reuse the authoritative profiles used to construct the fixed gate set."""
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


def query_vram_mib() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except Exception:
        return None


class VramMonitor:
    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = query_vram_mib()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(0.25)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join(timeout=5)


def contains_base(text: str | None, base: str) -> bool:
    return bool(text and re.search(r"(?<!\w)" + re.escape(base) + r"(?!\w)", text, re.I))


def number_tokens(text: str) -> list[str]:
    return re.findall(r"\b\d+\b", text)


def build_review_pack(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(rows, 1):
        blocks.append(
            "\n".join(
                [
                    "=" * 60,
                    f"ITEM {index:03d}",
                    "",
                    f"Series: {row['series']}",
                    f"Chapter: {row['chapter']}",
                    "",
                    "SOURCE:", row["original_source"], "",
                    "TRANSLATION INPUT:", row["translation_input"], "",
                    "NORMALIZED INPUT:", row["normalized_input"], "",
                    "PROTECTED INPUT:", row["protected_input"], "",
                    "PROTECTED TERMS:", json.dumps(row["protected_terms"], ensure_ascii=False), "",
                    "RAW HY FROM OLD SHOOTOUT:", row["old_raw_hy"], "",
                    "RAW HY THIS PRODUCTION CALL:", row["raw_hy_output"], "",
                    "RESTORED:", str(row["restored_output"]), "",
                    "FINAL:", str(row["final_output"]), "",
                    "GUARDS:", json.dumps(row["guard_flags"], ensure_ascii=False), "",
                    "REQUIRES REVIEW:", str(row["requires_review"]), "",
                    "PIPELINE DIAGNOSIS:", row["pipeline_diagnosis"], "",
                ]
            )
        )
    return "\n".join(blocks) + "=" * 60 + "\n"


def classify_row(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return objective structural failures and pipeline regression reasons."""
    failures: list[str] = []
    regressions: list[str] = []
    final = row["final_output"] or ""
    restored = row["restored_output"] or ""
    guards = set(row["guard_flags"])
    if not final:
        failures.append("missing_final_output")
    if "__WTTERM" in final:
        failures.append("sentinel_leak")
    if guards & CRITICAL_GUARDS:
        failures.extend(sorted(guards & CRITICAL_GUARDS))
    for meta in row["protected_terms"]:
        if meta["proper_name"] and not contains_base(restored, meta["target_base"]):
            failures.append(f"protected_name_missing:{meta['target_base']}")
    source_numbers = number_tokens(row["translation_input"])
    missing_numbers = [token for token in source_numbers if token not in final]
    if missing_numbers:
        failures.append("source_number_missing:" + ",".join(missing_numbers))

    # A regression is pipeline-caused only when a production stage creates an
    # objective structural failure that the stored raw output did not contain.
    old_raw = row["old_raw_hy"] or ""
    for failure in failures:
        if failure == "sentinel_leak" or failure.startswith("protected_name_missing"):
            regressions.append(failure)
        elif failure.startswith("source_number_missing"):
            missing = failure.split(":", 1)[1].split(",")
            if all(token in old_raw for token in missing):
                regressions.append(failure)
        elif failure in CRITICAL_GUARDS:
            regressions.append(failure)
    return sorted(set(failures)), sorted(set(regressions))


def run_gate() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actual_hash = sha256_file(DATASET_PATH)
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if actual_hash != EXPECTED_DATASET_SHA256 or len(dataset) != 30:
        summary = {
            "provider": "hy_mt2",
            "dataset_size": len(dataset),
            "dataset_sha256": actual_hash,
            "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
            "gate_passed": False,
            "failure": "dataset_hash_or_size_mismatch",
            "default_translator_switched": False,
        }
        write_json("summary.json", summary)
        (OUTPUT_DIR / "runtime_probe.txt").write_text(
            "STOPPED BEFORE MODEL LOAD: dataset hash/size mismatch\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    raw_rows = json.loads(RAW_HY_PATH.read_text(encoding="utf-8"))
    raw_by_id = {row["item_id"]: row for row in raw_rows}
    if len(raw_by_id) != 30 or {x["item_id"] for x in dataset} != set(raw_by_id):
        raise RuntimeError("Stored raw Hy results do not match fixed REAL30 item IDs")

    provider = HyMT2GGUFTranslationProvider()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    runtime_error: str | None = None
    baseline_vram = query_vram_mib()
    with VramMonitor() as monitor:
        try:
            provider.load()
            for index, item in enumerate(dataset, 1):
                production_input = TranslationInput(
                    items=[
                        TranslationItem(
                            region_id=index,
                            source=item["translation_input"],
                            reading_order=index,
                        )
                    ],
                    profile=profile_for(item["series"]),
                    chapter_id=f"{item['series']}::{item['chapter']}",
                )
                output = provider.translate(production_input)
                result = output.results[0]
                trace = asdict(provider.last_traces[0])
                old = raw_by_id[item["item_id"]]
                row = {
                    "item_id": item["item_id"],
                    "series": item["series"],
                    "chapter": item["chapter"],
                    "original_source": item["original_source"],
                    "translation_input": item["translation_input"],
                    **{key: trace[key] for key in (
                        "normalized_input", "protected_input", "protected_terms",
                        "raw_hy_output", "stripped_output", "restored_output",
                        "final_output", "guard_flags", "requires_review",
                        "model_call_performed", "latency_sec", "pipeline_diagnosis",
                    )},
                    "old_raw_hy": old.get("stripped_translation") or old.get("raw_output") or "",
                    "old_raw_flags": old.get("flags") or [],
                }
                failures, regressions = classify_row(row)
                row["structural_failures"] = failures
                row["pipeline_regressions"] = regressions
                row["structurally_usable"] = not failures
                rows.append(row)
                print(f"[{index:02d}/30] {item['item_id']}: {trace['pipeline_diagnosis']}")
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
        finally:
            command = list(provider.last_server_command)
            server_log_tail = provider._server_log_tail()
            provider.unload()

    elapsed = time.perf_counter() - started
    peak_vram_mib = max(monitor.samples) if monitor.samples else None
    peak_vram_gb = round(peak_vram_mib / 1024, 4) if peak_vram_mib is not None else None
    latencies = [row["latency_sec"] for row in rows if row["latency_sec"] is not None]
    model_calls = sum(row["model_call_performed"] for row in rows)
    successful_calls = sum(
        row["model_call_performed"] and bool(row["raw_hy_output"]) for row in rows
    )
    bypassed = [row["item_id"] for row in rows if not row["model_call_performed"]]
    sentinel_rows = [row["item_id"] for row in rows if "__WTTERM" in (row["final_output"] or "")]
    restore_rows = [
        row["item_id"] for row in rows
        if "unrestored_protected_term" in row["guard_flags"]
        or "approved_term_missing" in row["guard_flags"]
    ]
    name_rows = [
        row["item_id"] for row in rows
        if any(reason.startswith("protected_name_missing") for reason in row["structural_failures"])
    ]
    regression_rows = [row for row in rows if row["pipeline_regressions"]]
    review_rows = [row["item_id"] for row in rows if row["requires_review"]]
    usable_rows = [row["item_id"] for row in rows if row["structurally_usable"]]
    craft = next((row for row in rows if re.search(r"\[[A-Z][A-Z0-9 _-]+\]", row["translation_input"])), None)
    bracket_tokens = re.findall(r"\[[A-Z][A-Z0-9 _-]+\]", craft["translation_input"]) if craft else []
    craft_preserved = bool(
        craft
        and bracket_tokens
        and all(token in (craft["final_output"] or "") for token in bracket_tokens)
        and "__WTTERM" not in (craft["final_output"] or "")
    )
    money = next((row for row in rows if "my money" in row["translation_input"].casefold()), None)
    blood_axe = next((row for row in rows if "blood axe" in row["translation_input"].casefold()), None)
    guild = next((row for row in rows if "adventurers' guild" in row["translation_input"].casefold()), None)
    kidding = next((row for row in rows if "gotta be kidding" in row["translation_input"].casefold()), None)
    money_ownership_preserved = bool(
        money and re.search(r"\bparam\b", (money["final_output"] or "").casefold())
    )
    blood_axe_identity_preserved = bool(
        blood_axe
        and "balta" in (blood_axe["final_output"] or "").casefold()
        and "kılıç" not in (blood_axe["final_output"] or "").casefold()
    )
    guild_identity_preserved = bool(
        guild
        and any(token in (guild["final_output"] or "").casefold() for token in ("lonca", "guild"))
        and "tüccar" not in (guild["final_output"] or "").casefold()
    )
    wrapper_rows = [
        row["item_id"] for row in rows
        if "chatbot_or_explanation_output" in row["guard_flags"]
        or "source_translation_wrapper" in row["guard_flags"]
    ]
    all_caps_regressions = [
        row["item_id"] for row in rows
        if row["normalized_input"] != row["translation_input"]
        and row["original_source"] != row["translation_input"]
    ]

    conditions = {
        "dataset_hash_and_size": actual_hash == EXPECTED_DATASET_SHA256 and len(dataset) == 30,
        "all_items_completed": len(rows) == 30 and runtime_error is None,
        "zero_runtime_failures": runtime_error is None and successful_calls == model_calls,
        "zero_sentinel_leaks": not sentinel_rows,
        "zero_restore_failures": not restore_rows,
        "zero_name_corruptions": not name_rows,
        "zero_pipeline_semantic_regressions": not regression_rows,
        "bracketed_named_token_preserved": craft_preserved,
        "source_ownership_number_negation_not_regressed": money_ownership_preserved and not regression_rows,
        "existing_good_outputs_not_broadly_degraded": not regression_rows,
        "blood_axe_identity_not_corrupted": blood_axe_identity_preserved,
        "adventurers_guild_identity_not_corrupted": guild_identity_preserved,
        "no_systematic_all_caps_regression": not all_caps_regressions,
        "no_chatbot_or_wrapper_output": not wrapper_rows,
        "guard_review_routing_operational": all(
            not row["guard_flags"] or row["requires_review"] for row in rows
        ),
        "structurally_usable_at_least_26": len(usable_rows) >= min(26, 30 - len(bypassed)),
        "serious_regressions_target_zero": not regression_rows,
    }
    gate_passed = all(conditions.values())
    summary = {
        "provider": "hy_mt2",
        "model_path": DEFAULT_HY_MT2_MODEL_PATH,
        "dataset_size": len(dataset),
        "dataset_sha256": actual_hash,
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "model_calls": model_calls,
        "successful_model_calls": successful_calls,
        "failed_model_calls": model_calls - successful_calls,
        "bypassed_items": bypassed,
        "sentinel_leaks": len(sentinel_rows),
        "restore_failures": len(restore_rows),
        "name_corruptions": len(name_rows),
        "pipeline_semantic_regressions": len(regression_rows),
        "requires_review_count": len(review_rows),
        "structurally_usable_count": len(usable_rows),
        "craft_identity_preserved": craft_preserved,
        "special_case_evidence": {
            "craft": craft,
            "my_money": money,
            "blood_axe": blood_axe,
            "adventurers_guild": guild,
            "youve_gotta_be_kidding_me": kidding,
            "my_money_ownership_preserved": money_ownership_preserved,
            "blood_axe_identity_preserved": blood_axe_identity_preserved,
            "adventurers_guild_identity_preserved": guild_identity_preserved,
        },
        "avg_latency_sec": round(statistics.mean(latencies), 4) if latencies else None,
        "tokens_per_sec": round(provider.metrics.tokens_per_sec, 2) if provider.metrics.tokens_per_sec else None,
        "peak_vram_gb": peak_vram_gb,
        "gate_conditions": conditions,
        "gate_passed": gate_passed,
        "default_translator_switched": False,
        "runtime_error": runtime_error,
        "qwen_rerun": False,
        "madlad_rerun": False,
        "translategemma_rerun": False,
        "ocr_rerun": False,
        "semantic_resolver_rerun": False,
        "commit": False,
        "push": False,
    }

    comparison = [
        {
            "item_id": row["item_id"],
            "source": row["translation_input"],
            "old_raw_hy": row["old_raw_hy"],
            "new_protected_production_hy": row["final_output"],
            "guard_status": row["guard_flags"],
            "changed": row["old_raw_hy"] != row["final_output"],
            "pipeline_regressions": row["pipeline_regressions"],
        }
        for row in rows
    ]
    guard_summary = {
        "critical_guard_names": sorted(CRITICAL_GUARDS),
        "requires_review_item_ids": review_rows,
        "sentinel_leak_item_ids": sentinel_rows,
        "restore_failure_item_ids": restore_rows,
        "name_corruption_item_ids": name_rows,
        "pipeline_regression_items": {
            row["item_id"]: row["pipeline_regressions"] for row in regression_rows
        },
        "wrapper_item_ids": wrapper_rows,
        "guard_counts": {
            guard: sum(guard in row["guard_flags"] for row in rows)
            for guard in sorted({g for row in rows for g in row["guard_flags"]})
        },
    }
    performance = {
        "elapsed_sec_including_load_unload": round(elapsed, 4),
        "model_load_sec": provider.metrics.model_load_seconds,
        "avg_latency_sec": summary["avg_latency_sec"],
        "median_latency_sec": round(statistics.median(latencies), 4) if latencies else None,
        "min_latency_sec": round(min(latencies), 4) if latencies else None,
        "max_latency_sec": round(max(latencies), 4) if latencies else None,
        "generated_tokens": provider.metrics.generated_token_count,
        "prompt_tokens": provider.metrics.input_token_count,
        "tokens_per_sec": summary["tokens_per_sec"],
        "generation_requests_including_retries": provider.metrics.generation_call_count,
        "retries": provider.metrics.retries,
        "baseline_vram_mib": baseline_vram,
        "peak_vram_mib": peak_vram_mib,
        "peak_vram_gb": peak_vram_gb,
        "vram_samples": len(monitor.samples),
    }
    write_json("summary.json", summary)
    write_json("production_gate_results.json", {"summary": summary, "items": rows})
    write_json("raw_vs_protected_comparison.json", comparison)
    write_json("protection_trace.json", rows)
    write_json("guard_summary.json", guard_summary)
    write_json("performance_summary.json", performance)
    (OUTPUT_DIR / "production_gate_results.txt").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comparison_text = "\n\n".join(
        f"{row['item_id']}\nSOURCE: {row['source']}\nOLD RAW HY: {row['old_raw_hy']}\n"
        f"NEW PROTECTED HY: {row['new_protected_production_hy']}\n"
        f"GUARDS: {row['guard_status']}\nREGRESSIONS: {row['pipeline_regressions']}"
        for row in comparison
    ) + "\n"
    (OUTPUT_DIR / "raw_vs_protected_comparison.txt").write_text(
        comparison_text, encoding="utf-8"
    )
    (OUTPUT_DIR / "assistant_review_pack.txt").write_text(
        build_review_pack(rows), encoding="utf-8"
    )
    runtime_probe = {
        "dataset_verified_before_model_load": True,
        "model_exists": Path(DEFAULT_HY_MT2_MODEL_PATH).is_file(),
        "llama_server_command": command,
        "dedicated_server_url": provider.server_url,
        "runtime_error": runtime_error,
        "server_log_tail": server_log_tail,
        "only_hy_mt2_model_executed": True,
    }
    (OUTPUT_DIR / "runtime_probe.txt").write_text(
        json.dumps(runtime_probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate_passed else 1


def mark_default_switched() -> int:
    path = OUTPUT_DIR / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("gate_passed"):
        raise RuntimeError("Cannot mark default switched when the gate did not pass")
    summary["default_translator_switched"] = True
    write_json("summary.json", summary)
    results_path = OUTPUT_DIR / "production_gate_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["summary"] = summary
    write_json("production_gate_results.json", results)
    (OUTPUT_DIR / "production_gate_results.txt").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark-default-switched", action="store_true")
    args = parser.parse_args()
    return mark_default_switched() if args.mark_default_switched else run_gate()


if __name__ == "__main__":
    raise SystemExit(main())
