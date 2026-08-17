"""Unit tests for CUDA DLL initialization and ONNX Runtime GPU loading."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.system.cuda_init import (
    find_cuda_dll_directories,
    get_cuda_dll_directories,
    init_cuda_runtime,
    is_cuda_runtime_initialized,
)


def test_init_cuda_runtime_is_idempotent():
    """Verifies that init_cuda_runtime can be called multiple times safely."""
    res1 = init_cuda_runtime()
    res2 = init_cuda_runtime()
    assert res1 == res2
    assert is_cuda_runtime_initialized() is True


def test_find_cuda_dll_directories():
    """Verifies finding CUDA / Torch DLL directories."""
    dirs = find_cuda_dll_directories()
    assert isinstance(dirs, list)
    if sys.platform == "win32":
        # Should find at least cublas or torch/lib in .venv
        assert any("nvidia" in d.lower() or "torch" in d.lower() for d in dirs)


def test_get_cuda_dll_directories():
    """Verifies get_cuda_dll_directories returns registered paths."""
    init_cuda_runtime()
    dirs = get_cuda_dll_directories()
    assert isinstance(dirs, list)
    assert len(dirs) > 0


def test_ctd_loads_cuda_execution_provider():
    """Verifies that ComicTextDetector initializes with CUDAExecutionProvider when available."""
    from providers.detector.ctd import ComicTextDetector

    det = ComicTextDetector()
    det.load()
    assert det.is_loaded is True
    assert det.session is not None

    providers = det.session.get_providers()
    assert isinstance(providers, list)
    if "CUDAExecutionProvider" in providers:
        assert det.device == "cuda"
    else:
        assert det.device == "cpu"
