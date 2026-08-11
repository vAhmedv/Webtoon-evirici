"""Translation provider package.

Hy-MT2-7B Q8_0 is the default protected English→Turkish text translator.
TranslateGemma, Qwen3.5-9B GGUF, and Transformers remain explicitly selectable.
"""
from __future__ import annotations

from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)
def get_translation_provider(
    backend: str = "hy_mt2_gguf",
    **kwargs,
) -> TranslationProvider:
    """Factory function returning configured translation provider.

    Default backend is 'hy_mt2_gguf' (Hy-MT2-7B Q8_0 GGUF via llama.cpp CUDA).
    Legacy backends:
      - 'qwen_gguf' / 'gguf' (Qwen3.5-9B GGUF via llama.cpp CUDA)
      - 'transformers' (Qwen3.5-9B 8-bit/4-bit PyTorch)
    """
    b = backend.lower().strip()
    if b in ("hy_mt2_gguf", "hy_mt2", "hy-mt2", "hymt2", "default"):
        from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider

        return HyMT2GGUFTranslationProvider(**kwargs)
    elif b in ("translategemma_gguf", "translategemma", "gemma"):
        from providers.translation.translategemma_gguf_translation import (
            TranslateGemmaGGUFTranslationProvider,
        )

        return TranslateGemmaGGUFTranslationProvider(**kwargs)
    elif b in ("qwen_gguf", "qwen", "gguf", "llama.cpp", "llamacpp"):
        from providers.translation.qwen_gguf_translation import QwenGGUFTranslationProvider

        return QwenGGUFTranslationProvider(**kwargs)
    elif b in ("transformers", "bitsandbytes", "8bit"):
        # Keep the llama.cpp production path independent from the optional
        # PyTorch/CUDA runtime until the legacy backend is explicitly selected.
        from providers.translation.qwen_translation import QwenTranslationProvider

        return QwenTranslationProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown translation backend: '{backend}'. Use 'hy_mt2_gguf', "
            "'translategemma_gguf', 'qwen_gguf', or 'transformers'."
        )


def get_configured_translation_provider(config) -> TranslationProvider:
    """Build a provider from ``TranslatorConfig`` without enabling fallback."""
    backend = config.provider or "hy_mt2_gguf"
    kwargs = {}
    if config.model_path:
        kwargs["model_path"] = config.model_path
    if config.llama_executable:
        kwargs["executable_path"] = config.llama_executable
    if config.server_url:
        kwargs["server_url"] = config.server_url
    return get_translation_provider(backend, **kwargs)


def __getattr__(name: str):
    """Preserve the public legacy class export without eagerly importing torch."""
    if name == "QwenTranslationProvider":
        from providers.translation.qwen_translation import QwenTranslationProvider

        return QwenTranslationProvider
    if name == "QwenGGUFTranslationProvider":
        from providers.translation.qwen_gguf_translation import QwenGGUFTranslationProvider

        return QwenGGUFTranslationProvider
    if name == "HyMT2GGUFTranslationProvider":
        from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider

        return HyMT2GGUFTranslationProvider
    if name == "TranslateGemmaGGUFTranslationProvider":
        from providers.translation.translategemma_gguf_translation import (
            TranslateGemmaGGUFTranslationProvider,
        )

        return TranslateGemmaGGUFTranslationProvider
    raise AttributeError(name)


__all__ = [
    "TranslationProvider",
    "TranslationItem",
    "TranslationInput",
    "TranslationOutput",
    "TranslationOutputItem",
    "TranslateGemmaGGUFTranslationProvider",
    "HyMT2GGUFTranslationProvider",
    "QwenGGUFTranslationProvider",
    "QwenTranslationProvider",
    "get_translation_provider",
    "get_configured_translation_provider",
]
