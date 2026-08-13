"""Qwen-specific candidate extraction adapter for SeriesProfile discovery.

Extracts raw candidate suggestions using Qwen3.5-9B and passes them to
core.translation.profile_discovery for deterministic validation and storage.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from loguru import logger

from core.translation.profile_discovery import (
    CandidateStore,
    DiscoveryResult,
    process_discovered_suggestions,
)
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationItem
from providers.translation.qwen_translation import (
    MAX_NEW_TOKENS,
    QwenTranslationProvider,
    _reset_peak_vram,
    _strip_thinking,
    _try_extract_json,
)

_DISCOVERY_SYSTEM_PROMPT = """You are an expert webtoon/manhwa terminology and entity extractor.

Your task is to analyze the provided English speech bubbles from a webtoon chapter and identify candidate proper entities and domain terms.

Extract candidates into 4 kinds ONLY:
1. "character_name": Character names (e.g. Jin Woo, Alice, Luo Tian)
2. "place_name": Location and place names (e.g. Blackwind Ravine, Northern Continent)
3. "title_or_rank": Titles, roles, or ranks (e.g. Guild Master, Captain, Sect Leader)
4. "term": Recurring domain terms or jargon (e.g. Mana Core, Awakener, Secret Realm, Dantian)

Rules:
- Extract ONLY candidates that explicitly appear in the dialogue text.
- Do NOT include common general English words (e.g. fast, world, careful, go, kid, yes, no).
- Do NOT translate terms or titles into Turkish during discovery. Set "suggested_target" to null for "term" and "title_or_rank".
- Do NOT invent Turkish equivalents during discovery.
- For character_name and place_name, you may suggest canonical capitalization in "suggested_target" (e.g. KANG MINHO -> Kang Minho).
- Output ONLY the JSON object. No preamble, no reasoning, no markdown.

Output format:
{
  "candidates": [
    {
      "source": "<EXACT_TEXT_FROM_SOURCE>",
      "kind": "character_name|place_name|title_or_rank|term",
      "suggested_target": "<SUGGESTION_OR_NULL>",
      "evidence_ids": [<region_id>]
    }
  ]
}"""


def _build_discovery_prompt(items: list[TranslationItem]) -> str:
    parts = [_DISCOVERY_SYSTEM_PROMPT, "", "Chapter dialogues:"]
    for item in items:
        if item.source:
            parts.append(f"[{item.reading_order}] id={item.region_id} | {item.source}")
    parts.append("")
    parts.append("Output JSON with candidate entities:")
    return "\n".join(parts)


def discover_candidates_with_qwen(
    provider: QwenTranslationProvider,
    series_id: str,
    chapter_id: str,
    items: list[TranslationItem],
    existing_profile: SeriesProfile | None = None,
    candidate_store: CandidateStore | None = None,
) -> DiscoveryResult:
    import torch

    """Extract candidates using Qwen and process them through core validation and CandidateStore."""
    if not provider.is_loaded or provider._model is None or provider._processor is None:
        raise RuntimeError("QwenTranslationProvider not loaded; call load() first")

    # Filter items: skip empty or unresolved OCR error regions
    valid_items = [item for item in items if item.source and item.source.strip()]
    if not valid_items:
        logger.info("No valid dialogue items provided for discovery")
        return DiscoveryResult(candidates=[], filtered_count=0, warnings=["no_valid_items"])

    if candidate_store is None:
        candidate_store = CandidateStore(series_id=series_id)

    prompt = _build_discovery_prompt(valid_items)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    inputs = provider._processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        tokenize=True,
        enable_thinking=False,
    )
    if torch.cuda.is_available():
        inputs = {
            k: v.to("cuda:0") for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }

    _reset_peak_vram()
    t0 = time.perf_counter()
    with torch.no_grad():
        gen = provider._model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    gen_time = time.perf_counter() - t0

    ilen = inputs["input_ids"].shape[-1]
    raw_output = provider._processor.decode(
        gen[0, ilen:], skip_special_tokens=True
    ).strip()
    raw_output = _strip_thinking(raw_output)

    # Parse JSON
    json_obj = _try_extract_json(raw_output)
    raw_suggestions: list[dict[str, Any]] = []
    if json_obj and isinstance(json_obj, dict) and "candidates" in json_obj:
        cands = json_obj.get("candidates", [])
        if isinstance(cands, list):
            raw_suggestions = cands
    else:
        # Fallback: find any candidate json array in text
        match = re.search(r'"candidates"\s*:\s*(\[\s*\{.*?\}\s*\])', raw_output, re.DOTALL)
        if match:
            try:
                raw_suggestions = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    logger.info(
        f"Qwen discovery extraction completed in {gen_time:.2f}s ({len(raw_suggestions)} raw suggestions returned)"
    )

    # Pass to generic core validation and merging
    return process_discovered_suggestions(
        raw_suggestions=raw_suggestions,
        items=valid_items,
        chapter_id=chapter_id,
        candidate_store=candidate_store,
        existing_profile=existing_profile,
    )
