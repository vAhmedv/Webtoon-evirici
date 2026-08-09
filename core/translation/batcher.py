"""Token-aware batching and context propagation for translation providers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from core.translation.series_profile import SeriesProfile
from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)

DEFAULT_MAX_INPUT_TOKENS = 1200
DEFAULT_CONTEXT_WINDOW_SIZE = 3


def estimate_token_count(text: str) -> int:
    """Estimate token count for a text string (~4 chars per token + padding)."""
    if not text:
        return 0
    return math.ceil(len(text) / 3.8) + 4


@dataclass
class BatcherConfig:
    """Configuration for token-aware translation batching."""

    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    context_window_size: int = DEFAULT_CONTEXT_WINDOW_SIZE


class TranslationBatcher:
    """Batches dialogue items according to token budget while preserving context."""

    def __init__(self, config: BatcherConfig | None = None) -> None:
        self.config = config or BatcherConfig()

    def create_batches(
        self,
        inp: TranslationInput,
        tokenizer: Any | None = None,
    ) -> list[TranslationInput]:
        """Split a single TranslationInput into token-budgeted sub-inputs.

        Items are kept in strict reading order. Subsequent batches include
        previous clean English dialogues as context_items.
        """
        if not inp.items:
            return [inp]

        # Calculate base prompt overhead tokens (system prompt + profile/glossary/context)
        overhead = self._estimate_prompt_overhead(inp)
        token_budget = max(30, self.config.max_input_tokens - overhead)

        sub_batches: list[list[TranslationItem]] = []
        current_batch: list[TranslationItem] = []
        current_tokens = 0

        for item in inp.items:
            item_tokens = self._item_tokens(item, tokenizer)
            if current_batch and (current_tokens + item_tokens > token_budget):
                sub_batches.append(current_batch)
                current_batch = [item]
                current_tokens = item_tokens
            else:
                current_batch.append(item)
                current_tokens += item_tokens

        if current_batch:
            sub_batches.append(current_batch)

        if len(sub_batches) <= 1:
            return [inp]

        logger.info(
            f"Split {len(inp.items)} items into {len(sub_batches)} sub-batches (budget: {self.config.max_input_tokens} tokens)"
        )

        inputs: list[TranslationInput] = []
        for i, batch_items in enumerate(sub_batches):
            context_items: list[TranslationItem] = []
            if i > 0 and self.config.context_window_size > 0:
                prev_items = sub_batches[i - 1]
                ctx_count = min(len(prev_items), self.config.context_window_size)
                context_items = prev_items[-ctx_count:]

            inputs.append(
                TranslationInput(
                    items=batch_items,
                    glossary=inp.glossary,
                    chapter_context=inp.chapter_context,
                    profile=inp.profile,
                    context_items=context_items,
                    candidate_store=inp.candidate_store,
                    chapter_id=inp.chapter_id,
                )
            )


        return inputs

    def merge_outputs(
        self,
        original_input: TranslationInput,
        sub_outputs: list[TranslationOutput],
    ) -> TranslationOutput:
        """Merge outputs from multiple sub-batches into a single TranslationOutput.

        Maintains original item order and filters out any context-only items.
        """
        if not sub_outputs:
            return TranslationOutput(
                inputs=original_input,
                results=[],
                raw_response="",
                repair_model="unknown",
            )

        if len(sub_outputs) == 1:
            return sub_outputs[0]

        merged_results: list[TranslationOutputItem] = []
        combined_raw_parts: list[str] = []
        model_name = sub_outputs[0].repair_model

        # Map results by region_id
        results_by_id: dict[int, TranslationOutputItem] = {}
        for out in sub_outputs:
            combined_raw_parts.append(out.raw_response)
            for item_res in out.results:
                results_by_id[item_res.region_id] = item_res

        # Reconstruct in original item order
        for orig_item in original_input.items:
            res = results_by_id.get(orig_item.region_id)
            if res:
                merged_results.append(res)
            else:
                merged_results.append(
                    TranslationOutputItem(
                        region_id=orig_item.region_id,
                        source=orig_item.source,
                        translation=None,
                        raw_model_response="",
                        validation_warnings=["missing_id_after_batch_merge"],
                        requires_review=True,
                    )
                )

        return TranslationOutput(
            inputs=original_input,
            results=merged_results,
            raw_response="\n---\n".join(combined_raw_parts),
            repair_model=model_name,
        )

    def _item_tokens(self, item: TranslationItem, tokenizer: Any | None = None) -> int:
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(item.source)) + 15
            except Exception:
                pass
        return estimate_token_count(item.source) + 15

    def _estimate_prompt_overhead(self, inp: TranslationInput) -> int:
        from core.translation.profile_discovery import get_relevant_terms_for_item
        overhead = 50  # Base prompt overhead
        rel_app: dict[str, str] = {}
        for item in inp.items:
            app_t, _ = get_relevant_terms_for_item(item.source, inp.profile, inp.candidate_store)
            rel_app.update(app_t)

        if rel_app:
            overhead += estimate_token_count(str(rel_app))
        if inp.profile and inp.profile.notes:
            overhead += estimate_token_count(" ".join(inp.profile.notes))
        if inp.chapter_context:
            overhead += estimate_token_count(inp.chapter_context)
        return overhead
