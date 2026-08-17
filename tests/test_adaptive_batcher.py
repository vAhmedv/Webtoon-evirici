"""Unit tests for ElasticAdaptiveBatcher with 95% VRAM Ceiling and Step-by-Step Decay."""

from unittest.mock import MagicMock, patch
import pytest

from core.system.adaptive_batcher import (
    ElasticAdaptiveBatcher,
    execute_with_elastic_batch,
)


def test_elastic_batcher_normal_execution():
    items = list(range(20))
    batcher = ElasticAdaptiveBatcher(default_batch_size=8, min_batch_size=1, vram_ceiling=0.95)

    def process_fn(chunk):
        return [x * 2 for x in chunk]

    results = batcher.execute(items, process_fn)
    assert results == [x * 2 for x in items]
    assert batcher.current_optimal_batch == 8


def test_elastic_batcher_step_by_step_decay_on_oom():
    """Simulates OOM when chunk_size > 26, verifying step-by-step decay: 28 -> 27 -> 26."""
    items = list(range(100))
    batcher = ElasticAdaptiveBatcher(default_batch_size=28, min_batch_size=1, vram_ceiling=0.95)

    decay_history = []

    def mock_process(chunk):
        decay_history.append(len(chunk))
        if len(chunk) > 26:
            # Simulate CUDA OOM
            raise RuntimeError("CUDA out of memory. Tried to allocate 512MB")
        return [x + 1 for x in chunk]

    with patch("gc.collect") as mock_gc:
        results = batcher.execute(items, mock_process)

    assert results == [x + 1 for x in items]
    # First attempted 28 (failed) -> then 27 (failed) -> then 26 (succeeded)
    assert 28 in decay_history
    assert 27 in decay_history
    assert 26 in decay_history
    assert batcher.current_optimal_batch == 26
    assert mock_gc.call_count >= 2


def test_elastic_batcher_proactive_vram_ceiling():
    """Verifies that when VRAM usage >= 95%, batch size is proactively decremented."""
    batcher = ElasticAdaptiveBatcher(default_batch_size=16, min_batch_size=1, vram_ceiling=0.95)

    mock_props = MagicMock()
    mock_props.total_memory = 10_000_000_000  # 10 GB

    # Mock 9.6 GB allocated (96% >= 95%)
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=mock_props), \
         patch("torch.cuda.memory_allocated", return_value=9_600_000_000), \
         patch("torch.cuda.memory_reserved", return_value=9_600_000_000), \
         patch("torch.cuda.empty_cache") as mock_empty:
        batcher.check_vram_and_adjust()
        assert batcher.current_optimal_batch == 15
        mock_empty.assert_called_once()


def test_elastic_batcher_sticky_optimal_batch():
    """Verifies that the decayed optimal batch persists across calls."""
    batcher = ElasticAdaptiveBatcher(default_batch_size=10, min_batch_size=1)

    call_count = 0
    def mock_process(chunk):
        nonlocal call_count
        call_count += 1
        if len(chunk) > 7:
            raise RuntimeError("CUDA out of memory")
        return chunk

    # First call decays 10 -> 9 -> 8 -> 7
    batcher.execute(list(range(20)), mock_process)
    assert batcher.current_optimal_batch == 7

    # Second call starts directly at 7 without OOM
    call_count_before = call_count
    batcher.execute(list(range(14)), mock_process)
    # 14 items / 7 = 2 successful chunks
    assert call_count - call_count_before == 2


def test_execute_with_elastic_batch_helper():
    items = ["a", "b", "c", "d"]
    res = execute_with_elastic_batch(items, lambda c: [x.upper() for x in c], initial_batch=2)
    assert res == ["A", "B", "C", "D"]
