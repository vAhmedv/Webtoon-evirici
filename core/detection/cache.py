"""Detection cache — YOLO sonuçlarını YAML'de saklar ve tekrar kullanır.

Cache key: sha256(page_image_bytes) + model_id + model_mtime
Model değişirse (id veya mtime) cache MISS olur, YOLO yeniden çalışır.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from core.config import PROJECT_ROOT
from core.detection.bbox import BBox
from core.detection.detection import Detection, RegionType


CACHE_FILENAME = "detections.yaml"
CACHE_DIRNAME = ".cache"
CACHE_PATH = PROJECT_ROOT / CACHE_DIRNAME / CACHE_FILENAME
DEFAULT_MAX_ENTRIES = 512


def compute_image_hash(image_bytes: bytes) -> str:
    """Görüntü byte'larının sha256 hash'ini hesaplar."""
    return hashlib.sha256(image_bytes).hexdigest()


def _serialize_mask(mask: Any) -> Any:
    """Mask/polygon verisini YAML-serializable biçime çevirir."""
    if mask is None:
        return None
    if hasattr(mask, "tolist"):
        return mask.tolist()
    if isinstance(mask, (list, tuple)):
        return list(mask)
    return None


def _serialize_detection(det: Detection) -> dict[str, Any]:
    """Detection'ı YAML-serializable dict'e çevirir."""
    return {
        "bbox": {"x1": det.bbox.x1, "y1": det.bbox.y1, "x2": det.bbox.x2, "y2": det.bbox.y2},
        "confidence": float(det.confidence),
        "type": det.type.value,
        "source_window_id": int(det.source_window_id),
        "mask": _serialize_mask(det.mask),
        "metadata": dict(det.metadata),
    }


def _deserialize_detection(data: dict[str, Any]) -> Detection:
    """Dict'ten Detection oluşturur."""
    bbox_data = data["bbox"]
    bbox = BBox(
        x1=int(bbox_data["x1"]),
        y1=int(bbox_data["y1"]),
        x2=int(bbox_data["x2"]),
        y2=int(bbox_data["y2"]),
    )
    return Detection(
        bbox=bbox,
        confidence=float(data["confidence"]),
        type=RegionType(data["type"]),
        source_window_id=int(data["source_window_id"]),
        mask=data.get("mask"),
        metadata=dict(data.get("metadata", {})),
    )


class DetectionCache:
    """YAML-backed detection cache.

    Attributes:
        cache_path: YAML dosyasının yolu.
        max_entries: Maksimum giriş sayısı (LRU eviction için).
        enabled: Cache aktif mi?
    """

    def __init__(
        self,
        cache_path: str | Path | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        enabled: bool = True,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else CACHE_PATH
        self.max_entries = max_entries
        self.enabled = enabled
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Cache belleğe yüklü mü?"""
        return self._loaded

    def _cache_key(self, page_hash: str, model_id: str, model_mtime: str | float) -> str:
        """Cache anahtarı oluşturur: page_hash|model_id|model_mtime."""
        return f"{page_hash}|{model_id}|{model_mtime}"

    def _ensure_cache_dir(self) -> None:
        """Cache dizinini oluşturur."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """YAML dosyasından cache'i belleğe yükler.

        Dosya yoksa boş cache oluşturur. Hata durumunda loglar.
        """
        if not self.enabled:
            self._loaded = True
            return

        if not self.cache_path.exists():
            logger.debug(f"Cache dosyası yok, boş başlatılıyor: {self.cache_path}")
            self._loaded = True
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "entries" in data:
                raw_entries = data["entries"]
            else:
                raw_entries = data if isinstance(data, dict) else {}

            self._entries = OrderedDict()
            for key, value in raw_entries.items():
                if isinstance(value, dict):
                    self._entries[key] = value
            self._loaded = True
            logger.debug(f"Cache yüklendi: {len(self._entries)} giriş")
        except Exception as e:
            logger.warning(f"Cache yüklenemedi ({self.cache_path}): {e}")
            self._entries = OrderedDict()
            self._loaded = True

    def _evict_lru(self) -> None:
        """LRU eviction: en eski girişleri çıkarır."""
        while len(self._entries) > self.max_entries:
            evicted_key, _ = self._entries.popitem(last=False)
            logger.debug(f"Cache LRU eviction: {evicted_key}")

    def get(self, page_hash: str, model_id: str, model_mtime: str | float) -> list[Detection] | None:
        """Cache'den Detection listesini getirir.

        Args:
            page_hash: sha256(page_image_bytes).hexdigest().
            model_id: Model kimliği (dosya yolu veya hash).
            model_mtime: Model dosyasının mtime.

        Returns:
            Cache hit → Detection listesi; miss → None.
        """
        if not self.enabled:
            return None

        if not self._loaded:
            self.load()

        key = self._cache_key(page_hash, model_id, model_mtime)
        entry = self._entries.get(key)
        if entry is None:
            return None

        # LRU: erişilen girişi en yeni konuma taşı
        self._entries.move_to_end(key)

        raw_dets = entry.get("detections")
        if not raw_dets:
            return []

        try:
            detections = [_deserialize_detection(d) for d in raw_dets]
            logger.debug(f"Cache HIT: {key}")
            return detections
        except Exception as e:
            logger.warning(f"Cache deserialize hatası ({key}): {e}")
            return None

    def put(
        self,
        page_hash: str,
        model_id: str,
        model_mtime: str | float,
        detections: list[Detection],
    ) -> None:
        """Detection listesini cache'e ekler.

        Args:
            page_hash: sha256(page_image_bytes).hexdigest().
            model_id: Model kimliği.
            model_mtime: Model dosyasının mtime.
            detections: Saklanacak Detection listesi.
        """
        if not self.enabled or not detections:
            return

        if not self._loaded:
            self.load()

        key = self._cache_key(page_hash, model_id, model_mtime)
        serialized = [_serialize_detection(d) for d in detections]
        entry: dict[str, Any] = {
            "detections": serialized,
        }

        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._evict_lru()

        logger.debug(f"Cache PUT: {key} ({len(detections)} detections)")

    def flush(self) -> None:
        """Cache'i temizler ve YAML dosyasını siler."""
        self._entries.clear()
        self._loaded = False
        if self.cache_path.exists():
            try:
                self.cache_path.unlink()
                logger.info(f"Cache dosyası silindi: {self.cache_path}")
            except Exception as e:
                logger.warning(f"Cache dosyası silinemedi: {e}")

    def save(self) -> None:
        """Cache'i YAML dosyasına atomik yazma.

        Temp dosyaya yazar, ardından os.replace ile atomik taşır.
        """
        if not self.enabled:
            return

        self._ensure_cache_dir()

        data: dict[str, Any] = {
            "entries": dict(self._entries),
        }

        try:
            # Temp dosya oluştur (same directory for os.replace)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.cache_path.parent),
                prefix=".tmp_cache_",
                suffix=".yaml",
            )
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            # Atomic replace
            os.replace(tmp_path, self.cache_path)
            logger.debug(f"Cache kaydedildi: {self.cache_path} ({len(self._entries)} giriş)")
        except Exception as e:
            logger.warning(f"Cache kaydedilemedi: {e}")
            # Temp dosyayı temizle
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    @staticmethod
    def compute_hash(image_bytes: bytes) -> str:
        """Page image byte'larının sha256 hash'ini hesaplar.

        Args:
            image_bytes: PIL görüntüsünün raw byte'ları (PNG encode edilmiş).

        Returns:
            sha256 hexdigest string.
        """
        return hashlib.sha256(image_bytes).hexdigest()
