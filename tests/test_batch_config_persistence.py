"""Unit tests for BatchConfig persistence (save/load), serialization, and dialog integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from core.system.adaptive_batcher import (
    BatchConfig,
    get_batch_config,
    load_batch_config,
    save_batch_config,
    set_batch_config,
)
from gui.dialogs.batch_settings_dialog import BatchSettingsDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_batch_config_serialization_roundtrip():
    """BatchConfig to_dict and from_dict produce consistent objects with alias support."""
    orig = BatchConfig(
        mode="manual",
        vram_ceiling=0.88,
        lama_batch=48,
        ocr_vl_batch=128,
        llm_chunk=48,
        cpu_ocr_workers=14,
        detector_tile_batch=24,
        sticky_optimal_batch={"lama_batch": 48, "benchmark_vram_pct": 85.0},
    )

    d = orig.to_dict()
    assert d["mode"] == "manual"
    assert d["vram_ceiling"] == 0.88
    assert d["vram_ceiling_pct"] == 0.88
    assert d["inpainting_batch"] == 48
    assert d["cpu_workers"] == 14
    assert d["sticky_optimal_batch"]["lama_batch"] == 48

    restored = BatchConfig.from_dict(d)
    assert restored.mode == "manual"
    assert restored.vram_ceiling == 0.88
    assert restored.lama_batch == 48
    assert restored.ocr_vl_batch == 128
    assert restored.llm_chunk == 48
    assert restored.cpu_ocr_workers == 14
    assert restored.detector_tile_batch == 24
    assert restored.sticky_optimal_batch == {"lama_batch": 48, "benchmark_vram_pct": 85.0}


def test_batch_config_from_dict_aliases_and_clamping():
    """from_dict handles alias keys and clamps invalid values safely."""
    raw = {
        "mode": "INVALID_MODE",
        "vram_ceiling_pct": 1.5,  # Out of bounds -> clamp to 0.99
        "inpainting_batch": 500,  # Out of bounds -> clamp to 256
        "ocr_vl_batch": -5,       # Out of bounds -> clamp to 1
        "cpu_workers": 99,        # Out of bounds -> clamp to 16
    }
    cfg = BatchConfig.from_dict(raw)
    assert cfg.mode == "auto"
    assert cfg.vram_ceiling == 0.99
    assert cfg.lama_batch == 256
    assert cfg.ocr_vl_batch == 1
    assert cfg.cpu_ocr_workers == 16


def test_save_and_load_batch_config(tmp_path: Path):
    """Saves batch configuration to custom path and reloads accurately."""
    config_file = tmp_path / "test_batch_config.json"

    custom_cfg = BatchConfig(
        mode="manual",
        vram_ceiling=0.92,
        lama_batch=64,
        ocr_vl_batch=96,
        llm_chunk=48,
        cpu_ocr_workers=12,
        detector_tile_batch=16,
    )

    save_batch_config(custom_cfg, path=config_file)
    assert config_file.exists()

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "batch_settings" in data
    assert data["batch_settings"]["mode"] == "manual"
    assert data["batch_settings"]["lama_batch"] == 64

    # Reset global and reload from disk
    set_batch_config(BatchConfig())
    loaded = load_batch_config(path=config_file)

    assert loaded.mode == "manual"
    assert loaded.vram_ceiling == 0.92
    assert loaded.lama_batch == 64
    assert loaded.ocr_vl_batch == 96
    assert get_batch_config().lama_batch == 64


def test_load_batch_config_missing_file_fallback(tmp_path: Path):
    """load_batch_config returns current global config if file is missing."""
    missing_file = tmp_path / "non_existent.json"
    init_cfg = BatchConfig(lama_batch=42)
    set_batch_config(init_cfg)

    loaded = load_batch_config(path=missing_file)
    assert loaded.lama_batch == 42


def test_dialog_save_persists_to_disk(qapp, tmp_path: Path, monkeypatch):
    """BatchSettingsDialog _on_save_clicked persists settings to disk."""
    config_file = tmp_path / "dialog_batch_config.json"
    monkeypatch.setattr("core.system.adaptive_batcher.DEFAULT_BATCH_CONFIG_PATH", config_file)

    dialog = BatchSettingsDialog()
    dialog.radio_manual.setChecked(True)
    dialog.slider_vram.setValue(91)
    dialog.slider_lama.setValue(128)
    dialog.slider_ocr.setValue(192)
    dialog.slider_det.setValue(24)
    dialog.slider_llm.setValue(48)
    dialog.slider_cpu.setValue(12)

    dialog._on_save_clicked()

    assert config_file.exists()
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    settings = data["batch_settings"]
    assert settings["mode"] == "manual"
    assert settings["vram_ceiling"] == 0.91
    assert settings["lama_batch"] == 128
    assert settings["ocr_vl_batch"] == 192
    assert settings["detector_tile_batch"] == 24
    assert settings["llm_chunk"] == 48
    assert settings["cpu_ocr_workers"] == 12

    dialog.close()


def test_dialog_bench_completed_persists_sticky_optimal(qapp, tmp_path: Path, monkeypatch):
    """Benchmark completion updates sticky optimal values and persists to disk."""
    config_file = tmp_path / "bench_batch_config.json"
    monkeypatch.setattr("core.system.adaptive_batcher.DEFAULT_BATCH_CONFIG_PATH", config_file)

    dialog = BatchSettingsDialog()
    dialog._on_bench_completed(optimal_lama=128, optimal_ocr=192, optimal_llm=48, max_vram_pct=89.0)

    assert dialog.slider_lama.value() == 128
    assert dialog.slider_ocr.value() == 192
    assert dialog.slider_llm.value() == 48

    assert config_file.exists()
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sticky = data["batch_settings"]["sticky_optimal_batch"]
    assert sticky["lama_batch"] == 128
    assert sticky["ocr_vl_batch"] == 192
    assert sticky["llm_chunk"] == 48
    assert sticky["benchmark_vram_pct"] == 89.0

    dialog.close()
