"""Unit tests for BatchSettingsDialog and TelemetryStatusBar badge updates."""

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from core.system.adaptive_batcher import (
    BatchConfig,
    get_batch_config,
    set_batch_config,
)
from gui.components.telemetry_bar import TelemetryStatusBar
from gui.dialogs.batch_settings_dialog import BatchSettingsDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_batch_settings_dialog_load_and_mode_toggle(qapp):
    # Set initial config
    init_cfg = BatchConfig(mode="auto", vram_ceiling=0.95, lama_batch=24, ocr_vl_batch=32, detector_tile_batch=4)
    set_batch_config(init_cfg)

    dialog = BatchSettingsDialog()
    assert dialog.radio_auto.isChecked() is True
    assert dialog.slider_lama.isEnabled() is False
    assert dialog.slider_ocr.isEnabled() is False
    assert dialog.slider_det.isEnabled() is False

    # Switch to manual mode
    dialog.radio_manual.setChecked(True)
    assert dialog.slider_lama.isEnabled() is True
    assert dialog.slider_ocr.isEnabled() is True
    assert dialog.slider_det.isEnabled() is True
    assert dialog.slider_llm.isEnabled() is True

    dialog.close()


def test_batch_settings_dialog_save_and_apply(qapp):
    dialog = BatchSettingsDialog()
    dialog.radio_manual.setChecked(True)
    dialog.slider_vram.setValue(90)
    dialog.slider_lama.setValue(128)
    dialog.slider_ocr.setValue(192)
    dialog.slider_det.setValue(8)
    dialog.slider_llm.setValue(18)
    dialog.slider_cpu.setValue(12)

    applied_configs = []
    dialog.config_applied.connect(applied_configs.append)

    dialog._on_save_clicked()

    current_cfg = get_batch_config()
    assert current_cfg.mode == "manual"
    assert current_cfg.vram_ceiling == 0.90
    assert current_cfg.lama_batch == 128
    assert current_cfg.ocr_vl_batch == 192
    assert current_cfg.detector_tile_batch == 8
    assert current_cfg.llm_chunk == 18
    assert current_cfg.cpu_ocr_workers == 12
    assert len(applied_configs) == 1

    dialog.close()


def test_batch_settings_dialog_reset_defaults(qapp):
    dialog = BatchSettingsDialog()
    dialog.radio_manual.setChecked(True)
    dialog.slider_lama.setValue(10)
    dialog.slider_det.setValue(12)

    dialog._on_reset_clicked()

    assert dialog.radio_auto.isChecked() is True
    assert dialog.slider_vram.value() == 95
    assert dialog.slider_lama.value() == 24
    assert dialog.slider_ocr.value() == 64
    assert dialog.slider_det.value() == 16
    assert dialog.slider_llm.value() == 32
    assert dialog.slider_cpu.value() == 10

    dialog.close()


def test_telemetry_bar_batch_badge_update(qapp):
    bar = TelemetryStatusBar()
    assert bar.btn_batch_settings is not None

    # Test auto mode badge
    cfg_auto = BatchConfig(mode="auto", vram_ceiling=0.95)
    bar.update_batch_badge(cfg_auto)
    assert "OTO %95" in bar.btn_batch_settings.text()

    # Test manual mode badge
    cfg_man = BatchConfig(mode="manual", lama_batch=28)
    bar.update_batch_badge(cfg_man)
    assert "MANUEL (28)" in bar.btn_batch_settings.text()

    bar.timer.stop()
