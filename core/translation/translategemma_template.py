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
) -> str:
    """Render raw TranslateGemma prompt string for llama.cpp /completion endpoint."""
    lang_map = {
        "en": "English",
        "tr": "Turkish",
    }
    src_name = lang_map.get(source_lang_code.lower(), source_lang_code)
    tgt_name = lang_map.get(target_lang_code.lower(), target_lang_code)

    return (
        f"<bos><start_of_turn>user\n"
        f"Translate from {src_name} to {tgt_name}:\n"
        f"{prepared_source_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
