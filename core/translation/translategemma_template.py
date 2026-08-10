"""TranslateGemma Raw Prompt Template Renderer.

Renders raw TranslateGemma model prompts for llama.cpp /completion endpoint.
Strictly contains only turn tokens, language translation instruction, and prepared source text.
Does NOT contain Qwen prompt, system instructions, or glossary prose.
"""
from __future__ import annotations


def render_translategemma_prompt(
    prepared_source_text: str,
    source_lang_code: str = "en",
    target_lang_code: str = "tr",
    variant: str = "legacy",
) -> str:
    """Render raw TranslateGemma prompt string for llama.cpp /completion endpoint."""
    lang_map = {
        "en": "English",
        "tr": "Turkish",
    }
    src_name = lang_map.get(source_lang_code.lower(), source_lang_code)
    tgt_name = lang_map.get(target_lang_code.lower(), target_lang_code)

    normalized_variant = variant.strip().lower()
    if normalized_variant == "legacy":
        return (
            f"<bos><start_of_turn>user\n"
            f"Translate from {src_name} to {tgt_name}:\n"
            f"{prepared_source_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    if normalized_variant == "canonical":
        return (
            f"<bos><start_of_turn>user\n"
            f"You are a professional {src_name} ({source_lang_code}) to "
            f"{tgt_name} ({target_lang_code}) translator. Your goal is to "
            f"accurately convey the meaning and nuances of the original {src_name} "
            f"text while adhering to {tgt_name} grammar, vocabulary, and cultural "
            f"sensitivities.\n"
            f"Produce only the {tgt_name} translation, without any additional "
            f"explanations or commentary. Please translate the following {src_name} "
            f"text into {tgt_name}:\n\n\n"
            f"{prepared_source_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    if normalized_variant == "minimal_faithful":
        return (
            f"<bos><start_of_turn>user\n"
            f"Translate the following text from {src_name} ({source_lang_code}) to "
            f"{tgt_name} ({target_lang_code}).\n"
            f"Preserve the original meaning exactly. Do not add, infer, explain, "
            f"omit, or invent any information.\n"
            f"Output only the {tgt_name} translation.\n\n"
            f"{prepared_source_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    raise ValueError(
        "TranslateGemma prompt variant must be 'legacy', 'canonical', or "
        "'minimal_faithful'"
    )
