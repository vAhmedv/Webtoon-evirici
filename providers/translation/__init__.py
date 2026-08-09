"""Translation provider package.

TranslateGemma-12B GGUF via llama.cpp CUDA is the DEFAULT production translator.
Qwen3.5-9B GGUF and Transformers remain available as legacy/fallback backends.
"""
from __future__ import annotations

from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)
from providers.translation.qwen_translation import QwenTranslationProvider
from providers.translation.qwen_gguf_translation import QwenGGUFTranslationProvider
from providers.translation.translategemma_gguf_translation import (
    TranslateGemmaGGUFTranslationProvider,
)


def get_translation_provider(
    backend: str = "translategemma_gguf",
    **kwargs,
) -> TranslationProvider:
    """Factory function returning configured translation provider.

    Default backend is 'translategemma_gguf' (TranslateGemma-12B-IT Q5_K_M GGUF via llama.cpp CUDA).
    Legacy backends:
      - 'qwen_gguf' / 'gguf' (Qwen3.5-9B GGUF via llama.cpp CUDA)
      - 'transformers' (Qwen3.5-9B 8-bit/4-bit PyTorch)
    """
    b = backend.lower().strip()
    if b in ("translategemma_gguf", "translategemma", "gemma", "default"):
        return TranslateGemmaGGUFTranslationProvider(**kwargs)
    elif b in ("qwen_gguf", "qwen", "gguf", "llama.cpp", "llamacpp"):
        return QwenGGUFTranslationProvider(**kwargs)
    elif b in ("transformers", "bitsandbytes", "8bit"):
        return QwenTranslationProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown translation backend: '{backend}'. Use 'translategemma_gguf', 'qwen_gguf', or 'transformers'."
        )


__all__ = [
    "TranslationProvider",
    "TranslationItem",
    "TranslationInput",
    "TranslationOutput",
    "TranslationOutputItem",
    "TranslateGemmaGGUFTranslationProvider",
    "QwenGGUFTranslationProvider",
    "QwenTranslationProvider",
    "get_translation_provider",
]
