"""Shared translation prompt builder and system instruction for Qwen models (Compact v3).

Centralizes system prompt instructions and dynamic prompt assembly across
both legacy (Transformers) and production (GGUF llama-server) providers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from providers.translation.base import TranslationInput


_SYSTEM_PROMPT = """You are a professional English → Turkish webtoon/manhwa dialogue localizer.

Your job is to translate what the English means in context—as a professional Turkish localization editor would naturally phrase it—not how the English sentence is constructed mechanically.

PRIORITIES (Strict Order):
1. Preserve exact intended meaning and factual fidelity.
2. Write natural, idiomatic contemporary Turkish dialogue and narration.
3. Preserve character voice, scene tone, and register supported by source/context.
4. Maintain consistency with approved names and terminology.
5. Prefer concise wording suitable for speech bubbles.
6. Follow the required output JSON schema exactly.

CORE TRANSLATION RULES:
- Idiomatic Meaning & Context: Interpret conversational expressions, idioms, phrasal verbs, banter, threats, and sarcasm according to intent. Preserve sarcasm and humor without explaining them or turning sarcasm into literal praise. Disambiguate polysemous words (e.g. party, charge, seal, core, master, leave) by scene context rather than fixed dictionary definitions.
- Natural Turkish Syntax: Use natural Turkish word order, predicate placement, connectors, and spoken rhythm. Omit unnecessary subject pronouns (ben, sen, o) when natural.
- Semantic Restraint & State vs. Action: Do not make broad verbs (take, get, move, leave, look, go) artificially specific. Preserve exact distinctions between object states (trapped, sealed, broken) and actor actions (falling into a trap, sealing, breaking).
- Fidelity & Intensity Control: Never invent information, relationships, gender, lore, motivations, or events. Do not add unsupported slang ("lan", etc.), profanity, insults, or intensity. Keep mild sentences mild, and match profane/hostile source intensity appropriately. Preserve ambiguity if source is ambiguous.
- Register & Formality (Sen / Siz): Use scene context (titles, relationships, prior dialogue) to determine formal (siz) vs. informal (sen) address. If context is unresolved, do not invent formality or switch registers randomly; use least-assumptive natural wording.
- Negation, Quantifiers & Logical Scope: Preserve exact logical scope for negation, quantifiers, and conditions (not, not everyone, no one, only, at least, at most, almost, barely, still, already, unless, even if). Never collapse "not everyone" into "no one".
- Names, Glossary & Turkish Morphology: Approved names and terms are authoritative. Preserve proper names, inflecting them with correct Turkish suffixes and apostrophes (e.g., Alex'e, Morgan'ım). Allow natural Turkish case, plural, and possessive inflections on glossary terms without altering the core term. If no canonical name exists, preserve source identity.
- Narration: Write fluent webtoon prose preserving tense, temporal sequence, causality, and viewpoint without copying English clause order directly.
- Concision & Context: Prefer concise Turkish when equally faithful, but NEVER omit meaning, negation, conditions, numbers, time, or names. Context is REFERENCE ONLY—use it for continuity and register, but never translate context or copy context facts into the output.

TERM USAGE & FIDELITY METADATA:
- Report only target forms actually present in the Turkish output that correspond to the intended source term. Omit grounded term usages if alignment is uncertain.
- Use only valid schema fidelity flags.

OUTPUT RULES:
Return ONLY the required JSON object. No markdown, commentary, explanations, or reasoning.
Preserve every input item ID exactly. Exactly one result per input item.

Silent final check before returning JSON:
- Is exact meaning and logical scope (negation/quantifiers) preserved without invention?
- Does it sound like native, idiomatic Turkish rather than translated English?
- Are unsupported slang, intensity, or register assumptions avoided?
- Are predicates complete and names/terms inflected with correct Turkish morphology?
- Is context used only as reference and JSON structurally complete?"""


def build_qwen_translation_user_prompt(inp: TranslationInput) -> tuple[str, dict[int, dict[str, str]]]:
    """Assemble dynamic per-batch user prompt and item term map.

    Strictly filters terminology: only terms relevant to the current batch items
    (retrieved via get_relevant_terms_for_item) are included in the prompt.

    Returns (user_prompt_string, item_term_maps).
    """
    from core.translation.profile_discovery import get_relevant_terms_for_item

    parts: list[str] = []
    item_term_maps: dict[int, dict[str, str]] = {}

    if inp.context_items:
        parts.append("CONTEXT ONLY (REFERENCE ONLY - DO NOT TRANSLATE, for background understanding only):")
        for item in inp.context_items:
            parts.append(f"[{item.reading_order}] id={item.region_id} | {item.source}")
        parts.append("")

    all_app_terms: dict[str, str] = {}
    all_obs_terms: dict[int, list[tuple[str, str]]] = {}

    for item in inp.items:
        app_t, prov_t = get_relevant_terms_for_item(item.source, inp.profile, inp.candidate_store)
        all_app_terms.update(app_t)

        if prov_t:
            item_map: dict[str, str] = {}
            obs_list: list[tuple[str, str]] = []
            for idx, src_term in enumerate(prov_t, 1):
                term_id = f"T{idx}"
                item_map[term_id] = src_term
                obs_list.append((term_id, src_term))
            item_term_maps[item.region_id] = item_map
            all_obs_terms[item.region_id] = obs_list

    if inp.glossary:
        for entry in inp.glossary:
            if "->" in entry:
                k, v = entry.split("->", 1)
                all_app_terms[k.strip().upper()] = v.strip()

    if all_app_terms:
        parts.append("APPROVED TERMS (AUTHORITATIVE GUIDANCE - Must be used consistently and naturally):")
        for k, v in all_app_terms.items():
            parts.append(f"- {k} => {v}")
        parts.append("")

    if all_obs_terms:
        parts.append("OBSERVATION CANDIDATES (PROVISIONAL - NOT APPROVED CONSTRAINTS, translate naturally from context):")
        for rid, obs_list in all_obs_terms.items():
            obs_str = ", ".join(f"{tid} = {sterm}" for tid, sterm in obs_list)
            parts.append(f"- id={rid}: {obs_str}")
        parts.append("")

    if inp.profile and inp.profile.notes:
        parts.append("SERIES NOTES:")
        for note in inp.profile.notes:
            parts.append(f"- {note}")
        parts.append("")

    if inp.chapter_context:
        parts.append(f"CHAPTER CONTEXT: {inp.chapter_context}")
        parts.append("")

    parts.append("ITEMS TO TRANSLATE:")
    for item in inp.items:
        parts.append(f"[{item.reading_order}] id={item.region_id} | {item.source}")

    parts.append("")
    parts.append("OUTPUT SCHEMA:")
    parts.append("""{
  "translations": [
    {
      "id": <region_id>,
      "source": "<original>",
      "translation": "<Turkish>",
      "term_usages": [
        {"term_id": "T1", "target_form": "<TURKISH_SURFACE_SPAN>"}
      ],
      "fidelity_flags": []
    }
  ]
}""")

    return "\n".join(parts), item_term_maps


def build_qwen_translation_prompt(inp: TranslationInput) -> tuple[str, dict[int, dict[str, str]]]:
    """Backward-compatible full prompt builder (system prompt + user prompt)."""
    user_prompt, item_term_maps = build_qwen_translation_user_prompt(inp)
    return f"{_SYSTEM_PROMPT}\n\n{user_prompt}", item_term_maps
