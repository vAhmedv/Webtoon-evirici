from __future__ import annotations

from dataclasses import dataclass, asdict, field
import gc
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar
from loguru import logger

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_BATCH_CONFIG_PATH = Path.home() / ".webtoon_translator_config.json"
FALLBACK_BATCH_CONFIG_PATH = Path.home() / ".webtoon_translator_batch_config.json"


@dataclass
class BatchConfig:
    """Global configuration for elastic/manual batch execution."""

    mode: str = "auto"  # "auto" or "manual"
    vram_ceiling: float = 0.95  # 0.70 to 0.98
    lama_batch: int = 24  # 1 to 256
    ocr_vl_batch: int = 64  # 1 to 256
    llm_chunk: int = 32  # 1 to 64
    cpu_ocr_workers: int = 10  # 1 to 16
    detector_tile_batch: int = 16  # 1 to 32
    sticky_optimal_batch: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes configuration to dictionary with standardized and alias keys."""
        return {
            "mode": self.mode,
            "vram_ceiling": self.vram_ceiling,
            "vram_ceiling_pct": self.vram_ceiling,
            "lama_batch": self.lama_batch,
            "inpainting_batch": self.lama_batch,
            "ocr_vl_batch": self.ocr_vl_batch,
            "detector_tile_batch": self.detector_tile_batch,
            "llm_chunk": self.llm_chunk,
            "cpu_ocr_workers": self.cpu_ocr_workers,
            "cpu_workers": self.cpu_ocr_workers,
            "sticky_optimal_batch": dict(self.sticky_optimal_batch),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchConfig:
        """Deserializes configuration from dictionary with backwards-compatibility and clamp checks."""
        if not isinstance(d, dict):
            return cls()

        mode = str(d.get("mode", "auto")).lower()
        if mode not in ("auto", "manual"):
            mode = "auto"

        vram = float(d.get("vram_ceiling", d.get("vram_ceiling_pct", 0.95)))
        vram = max(0.50, min(0.99, vram))

        lama = int(d.get("lama_batch", d.get("inpainting_batch", 24)))
        ocr = int(d.get("ocr_vl_batch", 64))
        det = int(d.get("detector_tile_batch", 16))
        llm = int(d.get("llm_chunk", 32))
        cpu = int(d.get("cpu_ocr_workers", d.get("cpu_workers", 10)))
        sticky = dict(d.get("sticky_optimal_batch", {}))

        return cls(
            mode=mode,
            vram_ceiling=vram,
            lama_batch=max(1, min(256, lama)),
            ocr_vl_batch=max(1, min(256, ocr)),
            detector_tile_batch=max(1, min(32, det)),
            llm_chunk=max(1, min(64, llm)),
            cpu_ocr_workers=max(1, min(16, cpu)),
            sticky_optimal_batch=sticky,
        )


_GLOBAL_BATCH_CONFIG = BatchConfig()


def get_batch_config() -> BatchConfig:
    return _GLOBAL_BATCH_CONFIG


def set_batch_config(config: BatchConfig) -> None:
    global _GLOBAL_BATCH_CONFIG
    _GLOBAL_BATCH_CONFIG = config


def save_batch_config(config: BatchConfig | None = None, path: Path | str | None = None) -> None:
    """Diske 'batch_settings' anahtarı altında kalıcı JSON olarak kaydeder."""
    cfg = config or get_batch_config()
    target_path = Path(path) if path else DEFAULT_BATCH_CONFIG_PATH
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if target_path.exists():
            try:
                data = json.loads(target_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}
        data["batch_settings"] = cfg.to_dict()
        target_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("Batch configuration saved to {}", target_path)
    except Exception as e:
        logger.warning("Failed to save batch configuration to {}: {}", target_path, e)


def load_batch_config(path: Path | str | None = None) -> BatchConfig:
    """Diskten 'batch_settings' yapılandırmasını okur ve global yapılandırmayı günceller."""
    if path:
        target_path = Path(path)
    elif DEFAULT_BATCH_CONFIG_PATH.exists():
        target_path = DEFAULT_BATCH_CONFIG_PATH
    elif FALLBACK_BATCH_CONFIG_PATH.exists():
        target_path = FALLBACK_BATCH_CONFIG_PATH
    else:
        target_path = DEFAULT_BATCH_CONFIG_PATH

    if not target_path.exists():
        return get_batch_config()

    try:
        raw_text = target_path.read_text(encoding="utf-8")
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "batch_settings" in parsed:
            cfg = BatchConfig.from_dict(parsed["batch_settings"])
        elif isinstance(parsed, dict):
            cfg = BatchConfig.from_dict(parsed)
        else:
            cfg = BatchConfig()
        set_batch_config(cfg)
        logger.debug("Batch configuration loaded from {}: {}", target_path, cfg)
        return cfg
    except Exception as e:
        logger.warning("Failed to load batch configuration from {}: {}, using current default", target_path, e)
        return get_batch_config()


# Initialize on import if any config file exists
if DEFAULT_BATCH_CONFIG_PATH.exists() or FALLBACK_BATCH_CONFIG_PATH.exists():
    try:
        load_batch_config()
    except Exception:
        pass


class ElasticAdaptiveBatcher:
    """Manages elastic batch execution with proactive 95% VRAM checks and N -> N-1 reactive OOM recovery."""

    def __init__(
        self,
        default_batch_size: int = 32,
        min_batch_size: int = 1,
        vram_ceiling: float = 0.95,
    ) -> None:
        cfg = get_batch_config()
        self.default_batch_size = default_batch_size
        self.min_batch_size = min_batch_size
        self.vram_ceiling = cfg.vram_ceiling if cfg.mode == "auto" else vram_ceiling
        self.current_optimal_batch = default_batch_size

    def check_vram_and_adjust(self) -> None:
        """Proactively checks VRAM usage and decrements batch if exceeding ceiling."""
        try:
            import torch

            if not torch.cuda.is_available():
                return

            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram <= 0:
                return

            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            ratio = max(allocated, reserved) / total_vram

            if ratio >= self.vram_ceiling:
                prev = self.current_optimal_batch
                self.current_optimal_batch = max(self.min_batch_size, self.current_optimal_batch - 1)
                logger.warning(
                    "[ELASTIC BATCH] VRAM ceiling reached ({:.1f}% >= {:.1f}%). Reducing batch: {} -> {}",
                    ratio * 100,
                    self.vram_ceiling * 100,
                    prev,
                    self.current_optimal_batch,
                )
                torch.cuda.empty_cache()
                gc.collect()
        except Exception:
            pass

    def execute(
        self,
        items: Sequence[T],
        process_fn: Callable[[Sequence[T]], Sequence[R]],
        batch_size: int | None = None,
    ) -> list[R]:
        """Executes items in chunks with step-by-step (N -> N-1) decay on OOM."""
        if not items:
            return []

        if batch_size is not None:
            self.current_optimal_batch = min(self.current_optimal_batch, batch_size)

        results: list[R] = []
        idx = 0
        total = len(items)

        while idx < total:
            self.check_vram_and_adjust()
            chunk_size = max(self.min_batch_size, self.current_optimal_batch)
            chunk = items[idx : idx + chunk_size]

            try:
                chunk_results = process_fn(chunk)
                results.extend(chunk_results)
                idx += len(chunk)
            except Exception as e:
                err_str = str(e).lower()
                is_oom = (
                    "out of memory" in err_str
                    or "cuda error: out of memory" in err_str
                    or "cuda out of memory" in err_str
                    or e.__class__.__name__ == "OutOfMemoryError"
                )

                if is_oom:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()

                    prev_batch = self.current_optimal_batch
                    self.current_optimal_batch = max(self.min_batch_size, self.current_optimal_batch - 1)

                    logger.warning(
                        "[ELASTIC BATCH OOM] Memory pressure caught! Decayed batch: {} -> {}. Retrying chunk.",
                        prev_batch,
                        self.current_optimal_batch,
                    )

                    if prev_batch <= self.min_batch_size:
                        logger.error("[ELASTIC BATCH] OOM at minimum batch size {}", self.min_batch_size)
                        raise
                else:
                    raise

        return results


def execute_with_elastic_batch(
    items: Sequence[T],
    process_func: Callable[[Sequence[T]], Sequence[R]],
    initial_batch: int = 32,
    vram_threshold: float = 0.95,
    min_batch: int = 1,
) -> list[R]:
    """Helper function executing items with elastic batching."""
    batcher = ElasticAdaptiveBatcher(
        default_batch_size=initial_batch,
        min_batch_size=min_batch,
        vram_ceiling=vram_threshold,
    )
    return batcher.execute(items, process_func)
