"""Translation provider paketi."""
from __future__ import annotations

from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationProvider,
)
from providers.translation.qwen_translation import QwenTranslationProvider
from providers.translation.qwen_gguf_translation import QwenGGUFTranslationProvider


def get_translation_provider(
    backend: str = "gguf",
    **kwargs,
) -> TranslationProvider:
    """Factory function returning configured translation provider.

    backend can be 'gguf' (default, llama.cpp CUDA) or 'transformers' (legacy).
    """
    if backend.lower() in ("gguf", "llama.cpp", "llamacpp"):
        return QwenGGUFTranslationProvider(**kwargs)
    elif backend.lower() in ("transformers", "bitsandbytes", "8bit"):
        return QwenTranslationProvider(**kwargs)
    else:
        raise ValueError(f"Unknown translation backend: {backend}. Use 'gguf' or 'transformers'.")


__all__ = [
    "TranslationProvider",
    "TranslationItem",
    "TranslationInput",
    "TranslationOutput",
    "QwenTranslationProvider",
    "QwenGGUFTranslationProvider",
    "get_translation_provider",
]
