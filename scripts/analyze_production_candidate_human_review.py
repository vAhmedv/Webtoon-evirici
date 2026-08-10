"""Analyzer for Exported Human Review Answers.

Decodes blind human answers (A/B) against blind_answer_key.json and produces
decoded statistics, category breakdowns, error loss analysis, and recommendation classification.

Usage:
    python scripts/analyze_production_candidate_human_review.py [path_to_human_review_answers.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_ANSWERS_PATH = Path("benchmark_results/qwen_production_candidate_gate_v1/human_review_answers.json")
GATE_DIR = Path("benchmark_results/qwen_production_candidate_gate_v1")


def analyze_human_answers(answers_path: Path) -> dict:
    if not answers_path.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_path}")

    key_path = GATE_DIR / "blind_answer_key.json"
    manifest_path = GATE_DIR / "sample_manifest.json"
    items_path = GATE_DIR / "blind_review_items.json"

    if not key_path.exists() or not manifest_path.exists() or not items_path.exists():
        raise FileNotFoundError(f"Gate directory {GATE_DIR} missing required key/manifest/items files.")

    with open(answers_path, encoding="utf-8") as f:
        answers_data = json.load(f)

    # Allow either dict with "reviews" key or raw dict
    reviews = answers_data.get("reviews", answers_data)

    with open(key_path, encoding="utf-8") as f:
        answer_key = json.load(f)

    with open(manifest_path, encoding="utf-8") as f:
        manifest_list = json.load(f)
        manifest_by_id = {m["review_id"]: m for m in manifest_list}

    with open(items_path, encoding="utf-8") as f:
        items_list = json.load(f)
        items_by_id = {i["review_id"]: i for i in items_list}

    qwen_wins = 0
    tg_wins = 0
    tie_good = 0
    tie_bad = 0
    unscorable_ocr = 0

    qwen_critical_losses = 0
    tg_critical_losses = 0
    qwen_major_critical_losses = 0
    tg_major_critical_losses = 0

    by_series: dict[str, dict[str, int]] = {}
    by_chapter: dict[str, dict[str, int]] = {}
    tags_count: dict[str, int] = {}

    decoded_items = []
    qwen_win_examples = []
    tg_win_examples = []
    critical_error_examples = []
    tie_bad_examples = []

    for review_id, ans in reviews.items():
        if review_id not in answer_key:
            continue

        key_info = answer_key[review_id]
        manifest_info = manifest_by_id.get(review_id, {})
        item_info = items_by_id.get(review_id, {})

        winner_choice = ans.get("winner")
        severity = ans.get("severity", "NONE")
        tags = ans.get("tags", [])
        notes = ans.get("notes", "")

        for t in tags:
            tags_count[t] = tags_count.get(t, 0) + 1

        series = manifest_info.get("series", "unknown")
        chapter = manifest_info.get("chapter", "unknown")

        by_series.setdefault(series, {"qwen_wins": 0, "tg_wins": 0, "ties": 0, "unscorable": 0})
        by_chapter.setdefault(chapter, {"qwen_wins": 0, "tg_wins": 0, "ties": 0, "unscorable": 0})

        model_a = key_info["translation_a_model"]
        model_b = key_info["translation_b_model"]

        winning_model = None
        losing_model = None

        if winner_choice == "A":
            winning_model = model_a
            losing_model = model_b
        elif winner_choice == "B":
            winning_model = model_b
            losing_model = model_a
        elif winner_choice in ["TIE_GOOD", "TIE_BAD"]:
            winning_model = "TIE"
        elif winner_choice == "UNSCORABLE_OCR":
            winning_model = "UNSCORABLE_OCR"

        if winning_model == "qwen35":
            qwen_wins += 1
            by_series[series]["qwen_wins"] += 1
            by_chapter[chapter]["qwen_wins"] += 1
            if severity == "CRITICAL":
                tg_critical_losses += 1
            if severity in ["MAJOR", "CRITICAL"]:
                tg_major_critical_losses += 1

            qwen_win_examples.append({
                "review_id": review_id,
                "benchmark_id": key_info["benchmark_id"],
                "source": manifest_info.get("source"),
                "qwen_translation": item_info.get("translation_a") if model_a == "qwen35" else item_info.get("translation_b"),
                "tg_translation": item_info.get("translation_b") if model_a == "qwen35" else item_info.get("translation_a"),
                "severity": severity,
                "notes": notes,
            })

        elif winning_model == "translategemma":
            tg_wins += 1
            by_series[series]["tg_wins"] += 1
            by_chapter[chapter]["tg_wins"] += 1
            if severity == "CRITICAL":
                qwen_critical_losses += 1
            if severity in ["MAJOR", "CRITICAL"]:
                qwen_major_critical_losses += 1

            tg_win_examples.append({
                "review_id": review_id,
                "benchmark_id": key_info["benchmark_id"],
                "source": manifest_info.get("source"),
                "tg_translation": item_info.get("translation_a") if model_a == "translategemma" else item_info.get("translation_b"),
                "qwen_translation": item_info.get("translation_b") if model_a == "translategemma" else item_info.get("translation_a"),
                "severity": severity,
                "notes": notes,
            })

        elif winning_model == "TIE":
            if winner_choice == "TIE_GOOD":
                tie_good += 1
            else:
                tie_bad += 1
                tie_bad_examples.append({
                    "review_id": review_id,
                    "benchmark_id": key_info["benchmark_id"],
                    "source": manifest_info.get("source"),
                    "translation_a": item_info.get("translation_a"),
                    "translation_b": item_info.get("translation_b"),
                    "notes": notes,
                })
            by_series[series]["ties"] += 1
            by_chapter[chapter]["ties"] += 1

        elif winning_model == "UNSCORABLE_OCR":
            unscorable_ocr += 1
            by_series[series]["unscorable"] += 1
            by_chapter[chapter]["unscorable"] += 1

        if severity == "CRITICAL":
            critical_error_examples.append({
                "review_id": review_id,
                "benchmark_id": key_info["benchmark_id"],
                "losing_model": losing_model,
                "source": manifest_info.get("source"),
                "notes": notes,
            })

        decoded_items.append({
            "review_id": review_id,
            "benchmark_id": key_info["benchmark_id"],
            "series": series,
            "chapter": chapter,
            "winner_choice": winner_choice,
            "winning_model": winning_model,
            "losing_model": losing_model,
            "severity": severity,
            "tags": tags,
            "notes": notes,
        })

    total_reviewed = len(reviews)
    valid_reviewed = total_reviewed - unscorable_ocr

    decisive_count = qwen_wins + tg_wins
    qwen_win_rate_no_ties = (qwen_wins / decisive_count) if decisive_count > 0 else 0.0
    tg_win_rate_no_ties = (tg_wins / decisive_count) if decisive_count > 0 else 0.0

    # Recommendation rule evaluation
    recommendation = "INCONCLUSIVE"
    recommendation_reason = ""

    # Check series level regression
    qwen_series_regressed = any(
        stats["qwen_wins"] < stats["tg_wins"] for stats in by_series.values()
    )

    if (
        tg_wins == 0 and qwen_wins > 0
    ) or (
        tg_wins > 0 and qwen_wins >= 2.0 * tg_wins
        and qwen_major_critical_losses <= tg_major_critical_losses
        and not qwen_series_regressed
    ):
        recommendation = "STRONG_QWEN_CANDIDATE"
        recommendation_reason = (
            f"Qwen wins ({qwen_wins}) >= 2x TranslateGemma wins ({tg_wins}) with "
            f"fewer or equal major/critical losses ({qwen_major_critical_losses} vs {tg_major_critical_losses}) "
            f"and no series regression."
        )
    elif (
        tg_wins > 0 and qwen_wins >= 1.25 * tg_wins
        and qwen_major_critical_losses <= tg_major_critical_losses
    ):
        recommendation = "QWEN_CANDIDATE"
        recommendation_reason = (
            f"Qwen wins ({qwen_wins}) > TranslateGemma wins ({tg_wins}) by >=25% margin "
            f"with equal or fewer major/critical losses ({qwen_major_critical_losses} vs {tg_major_critical_losses})."
        )
    elif (
        qwen_wins > 0 and tg_wins >= 1.25 * qwen_wins
        and tg_major_critical_losses <= qwen_major_critical_losses
    ):
        recommendation = "KEEP_TRANSLATEGEMMA"
        recommendation_reason = (
            f"TranslateGemma wins ({tg_wins}) > Qwen wins ({qwen_wins}) by >=25% margin."
        )
    else:
        recommendation = "INCONCLUSIVE"
        recommendation_reason = (
            f"Results close or trade-offs vary (Qwen wins: {qwen_wins}, TG wins: {tg_wins}, "
            f"Qwen major/crit losses: {qwen_major_critical_losses}, TG major/crit losses: {tg_major_critical_losses})."
        )

    summary_results = {
        "total_reviewed": total_reviewed,
        "valid_reviewed": valid_reviewed,
        "unscorable_ocr": unscorable_ocr,
        "qwen_wins": qwen_wins,
        "translategemma_wins": tg_wins,
        "tie_good": tie_good,
        "tie_bad": tie_bad,
        "qwen_win_rate_no_ties": round(qwen_win_rate_no_ties, 4),
        "translategemma_win_rate_no_ties": round(tg_win_rate_no_ties, 4),
        "net_win_difference": qwen_wins - tg_wins,
        "qwen_critical_losses": qwen_critical_losses,
        "translategemma_critical_losses": tg_critical_losses,
        "qwen_major_critical_losses": qwen_major_critical_losses,
        "translategemma_major_critical_losses": tg_major_critical_losses,
        "by_series": by_series,
        "by_chapter": by_chapter,
        "category_tags": tags_count,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
    }

    # Write output files
    with open(GATE_DIR / "human_review_decoded.json", "w", encoding="utf-8") as f:
        json.dump(decoded_items, f, indent=2, ensure_ascii=False)

    with open(GATE_DIR / "human_review_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    # Format text report
    report_text = f"""Qwen Production Candidate Gate V1 — Human Review Decoded Analysis Report
================================================================================

SUMMARY METRICS:
----------------
Total Reviewed Items:              {total_reviewed}
Valid Scorable Items:              {valid_reviewed}
Unscorable OCR Excluded:           {unscorable_ocr}

Qwen3.5-9B Wins:                   {qwen_wins} ({round(qwen_win_rate_no_ties*100, 1)}% of decisive)
TranslateGemma-12B Wins:           {tg_wins} ({round(tg_win_rate_no_ties*100, 1)}% of decisive)
Tie Good (Both Acceptable):        {tie_good}
Tie Bad (Both Defective):          {tie_bad}
Net Win Difference (Qwen - TG):    {qwen_wins - tg_wins}

ERROR & LOSS ANALYSIS:
---------------------
Qwen Critical Losses:              {qwen_critical_losses}
TranslateGemma Critical Losses:    {tg_critical_losses}
Qwen Major+Critical Losses:        {qwen_major_critical_losses}
TranslateGemma Major+Critical:     {tg_major_critical_losses}

RECOMMENDATION RULE CLASSIFICATION:
----------------------------------
CLASSIFICATION: {recommendation}
REASON:         {recommendation_reason}

PER-SERIES BREAKDOWN:
--------------------
"""
    for s_name, s_stats in by_series.items():
        report_text += f"  - {s_name}: Qwen Wins={s_stats['qwen_wins']}, TG Wins={s_stats['tg_wins']}, Ties={s_stats['ties']}, Unscorable={s_stats['unscorable']}\n"

    report_text += "\nTOP EXAMPLES (HUMAN SCORED):\n----------------------------\n"
    report_text += "\n[ Top 5 Qwen Wins ]\n"
    for ex in qwen_win_examples[:5]:
        report_text += f"  - ID: {ex['review_id']} ({ex['benchmark_id']})\n    Source: {ex['source']}\n    Qwen:   {ex['qwen_translation']}\n    TG:     {ex['tg_translation']}\n    Notes:  {ex['notes']}\n"

    report_text += "\n[ Top 5 TranslateGemma Wins ]\n"
    for ex in tg_win_examples[:5]:
        report_text += f"  - ID: {ex['review_id']} ({ex['benchmark_id']})\n    Source: {ex['source']}\n    TG:     {ex['tg_translation']}\n    Qwen:   {ex['qwen_translation']}\n    Notes:  {ex['notes']}\n"

    if critical_error_examples:
        report_text += "\n[ All Critical Error Items ]\n"
        for ex in critical_error_examples:
            report_text += f"  - ID: {ex['review_id']} ({ex['benchmark_id']}), Loser={ex['losing_model']}\n    Source: {ex['source']}\n    Notes: {ex['notes']}\n"

    if tie_bad_examples:
        report_text += "\n[ All Tie Bad Items ]\n"
        for ex in tie_bad_examples:
            report_text += f"  - ID: {ex['review_id']} ({ex['benchmark_id']})\n    Source: {ex['source']}\n    Notes: {ex['notes']}\n"

    with open(GATE_DIR / "human_review_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Decoded report written to {GATE_DIR / 'human_review_report.txt'}")
    return summary_results


def main() -> None:
    answers_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ANSWERS_PATH
    if not answers_path.exists():
        print(f"Note: Human review answers file '{answers_path}' does not exist yet.")
        print("To run analyzer after human review, execute:")
        print(f"  python scripts/analyze_production_candidate_human_review.py {answers_path}")
        return

    results = analyze_human_answers(answers_path)
    print("Analyzer finished successfully!")
    print(f"Recommendation: {results['recommendation']}")


if __name__ == "__main__":
    main()
