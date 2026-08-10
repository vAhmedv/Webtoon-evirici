"""Script to repair Real Chapter Translation Gate Dataset (v1_clean).

Categorizes all 300 items from benchmark_results/real_chapter_translation_gate_v1/comparison.json
into VALID_STORY_TEXT or EXCLUDE_* categories using objective generic rules.
Outputs clean dataset artifacts and recomputes clean automated metrics without model reruns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SOURCE_DIR = Path("benchmark_results/real_chapter_translation_gate_v1")
CLEAN_DIR = Path("benchmark_results/real_chapter_translation_gate_v1_clean")

SFX_WORDS = {
    "AHHH", "GAH", "HMPH", "GRO OOM", "GROOOM", "THUMP", "WHOOSH", "BOOM", "BANG",
    "RUMBLE", "CLANG", "SWOOSH", "SIGH", "URGH", "GULP", "GRRR", "CRASH", "SNAP",
    "TAP", "SLAM", "GASP", "HEH", "HAHA", "PANT", "GIGGLE", "YAWN", "COUGH", "SHHH",
    "CLAP", "DING", "DONG", "BEEP", "BUZZ", "SHOCK", "ROAR", "SCREAM", "SOB", "WOAH",
    "WOW", "HUH", "EH", "UM", "UH", "AH", "OH", "AWE"
}

VALID_SHORT_WORDS = {
    "I", "A", "AN", "AM", "IS", "ARE", "IN", "ON", "AT", "TO", "GO", "NO", "YES",
    "OH", "AH", "OK", "MY", "ME", "HE", "WE", "IT", "UP", "BY", "OF", "OR", "SO",
    "YU", "GAO", "LUO", "HU", "SAN", "YUAN", "TIAN", "WANG", "LEE", "PARK", "KIM",
    "AXE", "GOD", "LV", "HP", "MP", "EXP"
}


def is_domain_or_watermark(text: str) -> tuple[bool, str]:
    t_clean = text.strip()
    if re.search(r'\b[a-z0-9\-_]+\.(com|net|org|site|io|xyz|info)\b', t_clean, re.I):
        return True, "domain_pattern"
    if re.search(r'\b(aryascans|asmotoon|manhwa|scanlation|discord\.gg)\b', t_clean, re.I):
        return True, "watermark_brand"
    return False, ""


def is_sfx_vocalization(text: str) -> tuple[bool, str]:
    t_clean = text.strip()
    core_sfx = re.sub(r'[\s!~.?,:\-\'\"]+', '', t_clean).upper()
    if core_sfx in SFX_WORDS or any(sfx in core_sfx for sfx in ["AHHH", "GAH", "GROOOM", "THUMP", "WHOOSH"]):
        return True, f"sfx_core_{core_sfx}"
    return False, ""


def is_ocr_garbage(text: str) -> tuple[bool, str]:
    t_clean = text.strip()
    if '\ufffd' in t_clean or re.search(r'[\u4e00-\u9fff]', t_clean):
        return True, "corrupted_encoding_or_cjk"

    core_alnum = re.sub(r'[^a-zA-Z0-9]', '', t_clean)
    if 1 <= len(core_alnum) <= 4 and core_alnum.upper() not in VALID_SHORT_WORDS and not core_alnum.isdigit():
        return True, f"short_non_word_{core_alnum}"

    if len(core_alnum) == 0:
        return True, "punctuation_only"

    return False, ""


def classify_item(item: dict, fragment_map: dict[str, str]) -> tuple[str, str, list[str]]:
    """Classifies a benchmark item into category, reason, and evidence."""
    src = item["original_accepted_english"].strip()
    item_id = item["id"]

    # 1. Obvious Watermark Filter
    is_wm, wm_reason = is_domain_or_watermark(src)
    if is_wm:
        return "EXCLUDE_WATERMARK", wm_reason, ["domain_like_string"]

    # 2. SFX / Vocalization Filter
    is_sfx, sfx_reason = is_sfx_vocalization(src)
    if is_sfx:
        return "EXCLUDE_SFX", sfx_reason, [sfx_reason]

    # 3. OCR Garbage Filter
    is_garb, garb_reason = is_ocr_garbage(src)
    if is_garb:
        return "EXCLUDE_OCR_GARBAGE", garb_reason, [garb_reason]

    # 4. Duplicate / Fragment Filter
    if item_id in fragment_map:
        return "EXCLUDE_FRAGMENT", "Subsequence or duplicate fragment of nearby region", [fragment_map[item_id]]

    # 5. Default Valid Story Text
    return "VALID_STORY_TEXT", "Valid story dialogue or narration", ["valid_english_prose"]


def find_subsequence_fragments(items: list[dict]) -> dict[str, str]:
    """Finds items that are prefixes/suffixes/subsequences of nearby items in same chapter."""
    fragments = {}
    by_chapter: dict[tuple[str, str], list[dict]] = {}
    for it in items:
        key = (it["series"], it["chapter"])
        by_chapter.setdefault(key, []).append(it)

    for key, ch_items in by_chapter.items():
        for i, it1 in enumerate(ch_items):
            t1 = it1["original_accepted_english"].strip().upper()
            if len(t1) < 10:
                continue
            for j, it2 in enumerate(ch_items):
                if i == j:
                    continue
                t2 = it2["original_accepted_english"].strip().upper()
                if len(t2) > len(t1) and (t1 in t2 or (len(t1) > 15 and t1[:20] == t2[:20])):
                    fragments[it1["id"]] = f"subsequence_of_{it2['id']}"
                    break
    return fragments


def main() -> None:
    print("=== Cleaning Real Chapter Translation Gate V1 Dataset ===")

    comp_file = SOURCE_DIR / "comparison.json"
    if not comp_file.exists():
        raise FileNotFoundError(f"Source file {comp_file} not found.")

    with open(comp_file, encoding="utf-8") as f:
        items = json.load(f)

    print(f"Loaded {len(items)} original items.")

    fragment_map = find_subsequence_fragments(items)

    valid_story_items = []
    excluded_items = []
    clean_manifest = []

    reason_counts: dict[str, int] = {}
    by_series_clean: dict[str, int] = {}
    by_chapter_clean: dict[str, int] = {}

    for item in items:
        cat, reason, evidence = classify_item(item, fragment_map)
        item_id = item["id"]
        series = item["series"]
        chapter = item["chapter"]
        src = item["original_accepted_english"]

        reason_counts[cat] = reason_counts.get(cat, 0) + 1

        if cat == "VALID_STORY_TEXT":
            valid_story_items.append(item)
            by_series_clean[series] = by_series_clean.get(series, 0) + 1
            by_chapter_clean[chapter] = by_chapter_clean.get(chapter, 0) + 1
            clean_manifest.append({
                "id": item_id,
                "series": series,
                "chapter": chapter,
                "source": src,
                "status": "VALID_STORY_TEXT",
            })
        else:
            excluded_items.append({
                "id": item_id,
                "series": series,
                "chapter": chapter,
                "source": src,
                "reason": cat,
                "evidence": evidence,
            })
            clean_manifest.append({
                "id": item_id,
                "series": series,
                "chapter": chapter,
                "source": src,
                "status": cat,
            })

    # Verify no lost IDs
    total_processed = len(valid_story_items) + len(excluded_items)
    assert total_processed == len(items) == 300, f"ID accounting mismatch! {total_processed} vs {len(items)}"

    # Recompute automated metrics on CLEAN subset
    tg_rev_clean = sum(1 for it in valid_story_items if it["translategemma"].get("requires_review"))
    qw_rev_clean = sum(1 for it in valid_story_items if it["qwen35"].get("requires_review"))

    tg_empty_clean = sum(1 for it in valid_story_items if not (it["translategemma"].get("translation") or "").strip())
    qw_empty_clean = sum(1 for it in valid_story_items if not (it["qwen35"].get("translation") or "").strip())

    num_clean = len(valid_story_items)

    clean_summary = {
        "original_total_items": 300,
        "valid_story_items_count": num_clean,
        "total_excluded_items": len(excluded_items),
        "exclusion_breakdown": reason_counts,
        "clean_by_series": by_series_clean,
        "clean_by_chapter": by_chapter_clean,
        "clean_subset_metrics": {
            "translategemma": {
                "valid_items": num_clean,
                "requires_review": tg_rev_clean,
                "review_rate_percent": round((tg_rev_clean / num_clean) * 100, 2),
                "empty_outputs": tg_empty_clean,
            },
            "qwen35": {
                "valid_items": num_clean,
                "requires_review": qw_rev_clean,
                "review_rate_percent": round((qw_rev_clean / num_clean) * 100, 2),
                "empty_outputs": qw_empty_clean,
            },
        },
        "contaminated_300_metrics_reference": {
            "translategemma_review_rate_percent": 37.33,
            "qwen35_review_rate_percent": 20.33,
        },
    }

    # Write output files
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLEAN_DIR / "clean_manifest.json", "w", encoding="utf-8") as f:
        json.dump(clean_manifest, f, indent=2, ensure_ascii=False)

    with open(CLEAN_DIR / "valid_story_items.json", "w", encoding="utf-8") as f:
        json.dump(valid_story_items, f, indent=2, ensure_ascii=False)

    with open(CLEAN_DIR / "excluded_items.json", "w", encoding="utf-8") as f:
        json.dump(excluded_items, f, indent=2, ensure_ascii=False)

    with open(CLEAN_DIR / "clean_summary.json", "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2, ensure_ascii=False)

    with open(CLEAN_DIR / "clean_comparison.json", "w", encoding="utf-8") as f:
        json.dump(valid_story_items, f, indent=2, ensure_ascii=False)

    # Format text clean comparison
    txt_lines = [
        "REAL CHAPTER TRANSLATION GATE V1 — CLEAN SUBSET COMPARISON",
        "=" * 70,
        f"Valid Story Items: {num_clean} / 300 (Excluded {len(excluded_items)} noise items)\n",
    ]
    for idx, it in enumerate(valid_story_items, 1):
        txt_lines.append(f"Item #{idx:03d} [{it['id']}] ({it['series']} / {it['chapter']})")
        txt_lines.append(f"  Source: {it['original_accepted_english']}")
        txt_lines.append(f"  TG:     {it['translategemma'].get('translation') or ''}")
        txt_lines.append(f"  Qwen:   {it['qwen35'].get('translation') or ''}\n")

    with open(CLEAN_DIR / "clean_comparison.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    print(f"Clean dataset saved to {CLEAN_DIR}/")
    print(f"Valid story items: {num_clean}, Excluded: {len(excluded_items)}")
    print(f"Clean TG review rate: {round((tg_rev_clean/num_clean)*100, 1)}% ({tg_rev_clean}/{num_clean})")
    print(f"Clean Qwen review rate: {round((qw_rev_clean/num_clean)*100, 1)}% ({qw_rev_clean}/{num_clean})")


if __name__ == "__main__":
    main()
