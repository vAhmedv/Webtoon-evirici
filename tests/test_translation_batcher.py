"""Unit tests for TranslationBatcher token-aware batching and context propagation."""
import pytest
from core.translation.batcher import BatcherConfig, TranslationBatcher
from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
)


def test_small_input_single_batch():
    batcher = TranslationBatcher(BatcherConfig(max_input_tokens=1000))
    items = [
        TranslationItem(region_id=1, source="Short dialogue 1", reading_order=0),
        TranslationItem(region_id=2, source="Short dialogue 2", reading_order=1),
    ]
    inp = TranslationInput(items=items)
    batches = batcher.create_batches(inp)
    assert len(batches) == 1
    assert len(batches[0].items) == 2
    assert len(batches[0].context_items) == 0


def test_budget_exceeded_multi_batch():
    # Low budget forces splitting
    batcher = TranslationBatcher(BatcherConfig(max_input_tokens=100, context_window_size=2))
    items = [
        TranslationItem(region_id=1, source="Long sentence number one for bubble", reading_order=0),
        TranslationItem(region_id=2, source="Long sentence number two for bubble", reading_order=1),
        TranslationItem(region_id=3, source="Long sentence number three for bubble", reading_order=2),
        TranslationItem(region_id=4, source="Long sentence number four for bubble", reading_order=3),
    ]
    inp = TranslationInput(items=items)
    batches = batcher.create_batches(inp)
    assert len(batches) >= 2

    # Reading order preserved
    all_batch_items = [item for b in batches for item in b.items]
    assert [i.region_id for i in all_batch_items] == [1, 2, 3, 4]

    # Context items passed to subsequent batch
    second_batch = batches[1]
    assert len(second_batch.context_items) > 0
    # Context items are from previous batch
    assert second_batch.context_items[0].region_id in [1, 2]


def test_merge_outputs_preserves_order_and_ids():
    batcher = TranslationBatcher()
    inp = TranslationInput(
        items=[
            TranslationItem(region_id=1, source="Hello", reading_order=0),
            TranslationItem(region_id=2, source="World", reading_order=1),
        ]
    )

    out1 = TranslationOutput(
        inputs=inp,
        results=[
            TranslationOutputItem(region_id=1, source="Hello", translation="Merhaba", raw_model_response="")
        ],
        raw_response="raw1",
        repair_model="mock",
    )
    out2 = TranslationOutput(
        inputs=inp,
        results=[
            TranslationOutputItem(region_id=2, source="World", translation="Dünya", raw_model_response="")
        ],
        raw_response="raw2",
        repair_model="mock",
    )

    merged = batcher.merge_outputs(inp, [out1, out2])
    assert len(merged.results) == 2
    assert merged.results[0].region_id == 1
    assert merged.results[0].translation == "Merhaba"
    assert merged.results[1].region_id == 2
    assert merged.results[1].translation == "Dünya"
