"""Minimal translation provider base interface.

Traceable, structured output: raw model response, parsed text, validation
warnings, and requires_review flag are all preserved per item.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.translation.series_profile import SeriesProfile


@dataclass(frozen=True)
class TranslationItem:
    """One dialogue bubble to translate."""

    region_id: int
    source: str
    reading_order: int = 0
    speaker: str | None = None
    known_names: list[str] = field(default_factory=list)
    nearby_context: str | None = None


@dataclass(frozen=True)
class TranslationInput:
    """A batch of dialogue bubbles for a single translation call."""

    items: list[TranslationItem]
    glossary: list[str] = field(default_factory=list)
    chapter_context: str | None = None
    profile: SeriesProfile | None = None
    context_items: list[TranslationItem] = field(default_factory=list)
    candidate_store: Any | None = None
    chapter_id: str = "ch001"


@dataclass(frozen=True)
class TranslationOutputItem:
    """Result for one translated bubble."""

    region_id: int
    source: str
    translation: str | None
    raw_model_response: str
    validation_warnings: list[str] = field(default_factory=list)
    requires_review: bool = False
    term_usages: list[dict[str, Any]] = field(default_factory=list)
    fidelity_flags: list[str] = field(default_factory=list)
    term_id_map: dict[str, str] = field(default_factory=dict)
    micro_batch_id: str | None = None
    micro_batch_region_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class TranslationOutput:
    """Full translation output."""

    inputs: TranslationInput
    results: list[TranslationOutputItem]
    raw_response: str
    repair_model: str


class TranslationProvider:
    """Interface for translation providers."""

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "translation"

    @property
    def version(self) -> str:
        return "unknown"

    def translate(self, inp: TranslationInput) -> TranslationOutput:
        raise NotImplementedError

    def translate_batch(
        self,
        texts: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Translates a list of strings with automatic item mapping."""
        if not texts:
            return []
        items = [TranslationItem(region_id=idx + 1, source=t) for idx, t in enumerate(texts)]
        inp = TranslationInput(items=items)
        out = self.translate(inp)
        out_map = {item.region_id: item.translation for item in out.results}
        return [out_map.get(idx + 1, "") or "" for idx in range(len(texts))]
