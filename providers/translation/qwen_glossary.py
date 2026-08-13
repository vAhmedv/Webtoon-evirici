"""Evidence-Grounded Term Resolver adapter using Qwen3.5-9B.

Selects preferred target terms ONLY from accumulated translation evidence observations.
CRITICAL SAFETY RULE: Cannot invent targets that were not observed in translation outputs.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.translation.profile_discovery import CandidateStore, ProfileCandidate
from core.translation.series_profile import SeriesProfile
from providers.translation.qwen_translation import (
    MAX_NEW_TOKENS,
    QwenTranslationProvider,
    _reset_peak_vram,
    _strip_thinking,
    _try_extract_json,
)


@dataclass
class TermResolutionResult:
    source: str
    kind: str
    status: str  # "resolved" or "unresolved"
    preferred_observed_target: str | None
    reason: str


_RESOLVER_SYSTEM_PROMPT = """You are an expert webtoon terminology resolver.

Your task is to select the single best observed Turkish target form for a candidate term based ONLY on the provided translation observations.

CRITICAL RULES:
1. You MUST choose "preferred_observed_target" from the provided "Observed target forms" list ONLY.
2. Do NOT invent new words or targets not listed in "Observed target forms".
3. If no clear choice exists or forms are contradictory, set "status" to "unresolved" and "preferred_observed_target" to null.
4. Output ONLY valid JSON. No preamble, no thinking blocks, no markdown.

Output format:
{
  "resolutions": [
    {
      "source": "<SOURCE_TERM>",
      "status": "resolved|unresolved",
      "preferred_observed_target": "<OBSERVED_FORM_OR_NULL>",
      "reason": "<CONCISE_EVIDENCE_BASED_REASON>"
    }
  ]
}"""


def resolve_candidate_targets_with_qwen(
    provider: QwenTranslationProvider,
    candidates: list[ProfileCandidate],
    existing_profile: SeriesProfile | None = None,
) -> list[TermResolutionResult]:
    import torch

    """Resolve candidate targets strictly grounded in translation evidence."""
    if not provider.is_loaded or provider._model is None or provider._processor is None:
        raise RuntimeError("QwenTranslationProvider not loaded; call load() first")

    # Filter: Resolver ONLY runs on candidates with 2+ observations
    eligible_candidates = [c for c in candidates if len(c.observations) >= 2 and c.observed_target_counts]
    if not eligible_candidates:
        logger.info("No candidates eligible for evidence-grounded resolution (requires 2+ observations)")
        return []

    # Build prompt
    parts = [_RESOLVER_SYSTEM_PROMPT, "", "Candidate terms to resolve:"]
    for cand in eligible_candidates:
        obs_forms = ", ".join(f"'{k}' ({v}x)" for k, v in cand.observed_target_counts.items())
        sample_obs = cand.observations[0] if cand.observations else None
        evidence_str = f"\"{sample_obs.translated_text}\"" if sample_obs else "No snippet"
        parts.append(f"- Source: {cand.source} (kind: {cand.kind})")
        parts.append(f"  Observed target forms: [{obs_forms}]")
        parts.append(f"  Sample translation evidence: {evidence_str}")

    parts.append("")
    parts.append("Output JSON with term resolutions:")
    prompt = "\n".join(parts)

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

    json_obj = _try_extract_json(raw_output)
    results: list[TermResolutionResult] = []

    raw_resolutions = []
    if json_obj and isinstance(json_obj, dict):
        raw_resolutions = json_obj.get("resolutions", json_obj.get("proposals", []))

    cand_map = {c.source.strip().upper(): c for c in eligible_candidates}

    for item in raw_resolutions:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "")).strip().upper()
        cand = cand_map.get(src)
        if not cand:
            continue

        raw_target = item.get("preferred_observed_target")
        target_str = str(raw_target).strip() if raw_target else None
        status = str(item.get("status", "unresolved")).strip().lower()
        reason = str(item.get("reason", "")).strip()

        # DETERMINISTIC GROUNDING CHECK:
        # Target MUST be one of the observed target forms!
        valid_observed_forms = {k.lower() for k in cand.observed_target_counts.keys()}
        if target_str and target_str.lower() not in valid_observed_forms:
            logger.warning(f"Resolver output '{target_str}' for '{src}' was NOT in observed target forms {valid_observed_forms}. Marking UNRESOLVED.")
            status = "unresolved"
            target_str = None

        if status == "resolved" and target_str:
            cand.suggested_target = target_str

        results.append(
            TermResolutionResult(
                source=cand.source,
                kind=cand.kind,
                status=status if target_str else "unresolved",
                preferred_observed_target=target_str if status == "resolved" else None,
                reason=reason,
            )
        )

    logger.info(f"Evidence-grounded term resolver finished in {gen_time:.2f}s ({len(results)} resolved)")
    return results
