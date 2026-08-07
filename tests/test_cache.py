"""Detection cache testleri."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from core.detection import BBox, Detection, DetectionCache, RegionType
from core.detection.cache import CACHE_PATH, DEFAULT_MAX_ENTRIES, compute_image_hash


def _make_detection(
    bbox: tuple[int, int, int, int],
    window_id: int,
    confidence: float = 0.8,
    rtype: RegionType = RegionType.DIALOGUE,
) -> Detection:
    return Detection(
        bbox=BBox.from_tuple(bbox),
        confidence=confidence,
        type=rtype,
        source_window_id=window_id,
    )


def _make_image_bytes(width: int = 100, height: int = 100, color=(255, 0, 0)) -> bytes:
    """Sentetik görüntüden byte üretir."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDetectionCache:
    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=10)
        result = cache.get("hash1", "model_v1", "12345")
        assert result is None

    def test_cache_hit_returns_detections(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=10)
        dets = [_make_detection((10, 10, 50, 50), window_id=0)]
        cache.put("hash1", "model_v1", "12345", dets)

        result = cache.get("hash1", "model_v1", "12345")
        assert result is not None
        assert len(result) == 1
        assert result[0].bbox.x1 == 10
        assert result[0].confidence == pytest.approx(0.8)

    def test_model_change_invalidates_cache(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=10)
        dets = [_make_detection((10, 10, 50, 50), window_id=0)]

        cache.put("hash1", "model_v1", "12345", dets)

        # Same page hash, different model_id → miss
        result = cache.get("hash1", "model_v2", "12345")
        assert result is None

    def test_model_mtime_change_invalidates_cache(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=10)
        dets = [_make_detection((10, 10, 50, 50), window_id=0)]

        cache.put("hash1", "model_v1", "12345", dets)

        # Same page hash + model_id, different mtime → miss
        result = cache.get("hash1", "model_v1", "67890")
        assert result is None

    def test_page_change_invalidates_cache(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=10)
        dets = [_make_detection((10, 10, 50, 50), window_id=0)]

        cache.put("hash1", "model_v1", "12345", dets)

        # Different page hash → miss
        result = cache.get("hash2", "model_v1", "12345")
        assert result is None

    def test_lru_eviction_removes_oldest(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=3)

        cache.put("hash1", "m", "1", [_make_detection((1, 1, 10, 10), 0)])
        cache.put("hash2", "m", "1", [_make_detection((2, 2, 10, 10), 0)])
        cache.put("hash3", "m", "1", [_make_detection((3, 3, 10, 10), 0)])

        # Access hash1 so it becomes most-recently-used
        cache.get("hash1", "m", "1")

        # Add 4th entry → should evict hash2 (oldest after hash1 was accessed)
        cache.put("hash4", "m", "1", [_make_detection((4, 4, 10, 10), 0)])

        assert cache.get("hash1", "m", "1") is not None
        assert cache.get("hash2", "m", "1") is None
        assert cache.get("hash3", "m", "1") is not None
        assert cache.get("hash4", "m", "1") is not None

    def test_save_and_load_persists_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.yaml"
        cache = DetectionCache(cache_path=cache_path, max_entries=10)
        dets = [
            _make_detection((10, 10, 50, 50), window_id=0, confidence=0.9),
            _make_detection((60, 60, 100, 100), window_id=1, confidence=0.7),
        ]
        cache.put("hash1", "model_v1", "12345", dets)
        cache.save()

        # New cache instance, same file
        cache2 = DetectionCache(cache_path=cache_path, max_entries=10)
        cache2.load()
        result = cache2.get("hash1", "model_v1", "12345")
        assert result is not None
        assert len(result) == 2
        assert result[0].bbox.x1 == 10
        assert result[1].confidence == pytest.approx(0.7)

    def test_flush_deletes_cache_file(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.yaml"
        cache = DetectionCache(cache_path=cache_path, max_entries=10)
        cache.put("hash1", "model_v1", "12345", [_make_detection((10, 10, 50, 50), 0)])
        cache.save()
        assert cache_path.exists()

        cache.flush()
        assert not cache_path.exists()
        assert cache.get("hash1", "model_v1", "12345") is None

    def test_disabled_cache_always_misses(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=10, enabled=False)
        cache.put("hash1", "m", "1", [_make_detection((10, 10, 50, 50), 0)])
        result = cache.get("hash1", "m", "1")
        assert result is None

    def test_compute_image_hash_deterministic(self) -> None:
        bytes1 = _make_image_bytes(100, 100, (255, 0, 0))
        bytes2 = _make_image_bytes(100, 100, (255, 0, 0))
        bytes3 = _make_image_bytes(100, 100, (0, 255, 0))

        h1 = compute_image_hash(bytes1)
        h2 = compute_image_hash(bytes2)
        h3 = compute_image_hash(bytes3)

        assert h1 == h2
        assert h1 != h3

    def test_empty_detections_not_cached(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=10)
        cache.put("hash1", "m", "1", [])
        result = cache.get("hash1", "m", "1")
        assert result is None

    def test_cache_with_polygon_metadata(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=10)
        det = Detection(
            bbox=BBox.from_tuple((10, 10, 50, 50)),
            confidence=0.9,
            type=RegionType.DIALOGUE,
            source_window_id=0,
            metadata={"polygon": [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]]},
        )
        cache.put("hash1", "m", "1", [det])
        result = cache.get("hash1", "m", "1")
        assert result is not None
        assert len(result) == 1
        poly = result[0].metadata.get("polygon")
        assert poly is not None
        assert len(poly) == 4
        assert poly[0] == [10.0, 10.0]

    def test_default_cache_path_constant(self) -> None:
        assert CACHE_PATH is not None
        assert CACHE_PATH.name == "detections.yaml"
        assert CACHE_PATH.parent.name == ".cache"

    def test_default_max_entries(self) -> None:
        assert DEFAULT_MAX_ENTRIES == 512

    def test_get_after_eviction_then_re_add(self, tmp_path: Path) -> None:
        cache = DetectionCache(cache_path=tmp_path / "cache.yaml", max_entries=2)

        cache.put("h1", "m", "1", [_make_detection((1, 1, 5, 5), 0)])
        cache.put("h2", "m", "1", [_make_detection((2, 2, 5, 5), 0)])
        cache.put("h3", "m", "1", [_make_detection((3, 3, 5, 5), 0)])  # evicts h1

        assert cache.get("h1", "m", "1") is None
        assert cache.get("h2", "m", "1") is not None
        assert cache.get("h3", "m", "1") is not None
