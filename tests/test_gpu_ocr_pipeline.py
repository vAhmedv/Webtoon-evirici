"""Unit tests for In-GPU Zero-Copy Tensor Cropping and OCR Pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, Region, RegionType
from core.imaging.region_cropper import RegionCrop, RegionCropper
from core.models import Page
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider


def test_in_gpu_cropping_and_tensor_creation(tmp_path: Path):
    """Verifies that In-GPU cropping produces valid torch.Tensor and RegionCrop."""
    img = Image.new("RGB", (800, 14500), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 500, 300, 600], fill=(10, 20, 30))
    page_path = tmp_path / "long_page.webp"
    img.save(page_path, "WEBP", quality=90)

    page = Page(index=0, path=page_path, width=800, height=14500, y_offset=0)
    coords = GlobalCoordinateSystem([page])
    cropper = RegionCropper([page], coords, padding=10)

    region = Region(
        id=1,
        global_bbox=BBox(x1=100, y1=500, x2=300, y2=600),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(0,),
    )

    crop = cropper.crop_region_gpu(region, adaptive_padding=False, create_pil=True)

    assert crop.tensor is not None
    assert isinstance(crop.tensor, torch.Tensor)
    # Tensor should be [C, H, W] -> [3, 100 + 20, 200 + 20]
    assert crop.tensor.shape == (3, 120, 220)
    assert crop.image is not None
    assert crop.image.size == (220, 120)
    assert crop.global_origin == (90, 490)

    # Test to_tensor helper
    t_cuda = crop.to_tensor("cuda" if torch.cuda.is_available() else "cpu")
    assert isinstance(t_cuda, torch.Tensor)
    assert t_cuda.shape == (3, 120, 220)

    cropper.clear_gpu_cache()
    assert len(cropper._gpu_page_cache) == 0


def test_in_gpu_vs_cpu_cropping_pixel_consistency(tmp_path: Path):
    """Verifies that GPU tensor slicing matches CPU PIL cropping pixel-for-pixel."""
    arr = np.random.randint(0, 256, (1200, 800, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    page_path = tmp_path / "test_consistency.png"
    img.save(page_path, "PNG")

    page = Page(index=0, path=page_path, width=800, height=1200, y_offset=0)
    coords = GlobalCoordinateSystem([page])
    cropper = RegionCropper([page], coords, padding=15)

    region = Region(
        id=2,
        global_bbox=BBox(x1=150, y1=300, x2=450, y2=500),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(0,),
    )

    crop_gpu = cropper.crop_region_gpu(region, adaptive_padding=False, create_pil=True)
    crop_cpu = cropper._crop_region_cpu(region, adaptive_padding=False)

    gpu_pil_arr = np.array(crop_gpu.to_pil())
    cpu_pil_arr = np.array(crop_cpu.image)

    assert gpu_pil_arr.shape == cpu_pil_arr.shape
    np.testing.assert_array_equal(gpu_pil_arr, cpu_pil_arr)

    cropper.clear_gpu_cache()


def test_in_gpu_page_crossing_boundary_crop(tmp_path: Path):
    """Verifies that In-GPU crop seamlessly stitches regions crossing page boundaries."""
    img0 = Image.new("RGB", (800, 2000), (100, 100, 100))
    img1 = Image.new("RGB", (800, 2000), (200, 200, 200))
    p0_path = tmp_path / "page_0.png"
    p1_path = tmp_path / "page_1.png"
    img0.save(p0_path)
    img1.save(p1_path)

    pages = [
        Page(index=0, path=p0_path, width=800, height=2000, y_offset=0),
        Page(index=1, path=p1_path, width=800, height=2000, y_offset=2000),
    ]
    coords = GlobalCoordinateSystem(pages)
    cropper = RegionCropper(pages, coords, padding=10)

    # Region spans from Y=1950 (Page 0) to Y=2050 (Page 1)
    region = Region(
        id=3,
        global_bbox=BBox(x1=100, y1=1950, x2=300, y2=2050),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(0,),
    )

    crop = cropper.crop_region_gpu(region, adaptive_padding=False, create_pil=True)
    assert crop.tensor is not None
    # Height = (2050 - 1950) + 20 pad = 120. Width = 200 + 20 = 220
    assert crop.tensor.shape == (3, 120, 220)
    assert crop.page_indices == (0, 1)

    cropper.clear_gpu_cache()


def test_in_gpu_upscaling_small_text(tmp_path: Path):
    """Verifies that text with height < 36px is scaled directly via GPU interpolation."""
    img = Image.new("RGB", (800, 1000), (255, 255, 255))
    p_path = tmp_path / "page_small.png"
    img.save(p_path)

    page = Page(index=0, path=p_path, width=800, height=1000, y_offset=0)
    coords = GlobalCoordinateSystem([page])
    cropper = RegionCropper([page], coords, padding=0)

    # Height = 10px (< 36px)
    region = Region(
        id=4,
        global_bbox=BBox(x1=50, y1=100, x2=150, y2=110),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(0,),
    )

    crop = cropper.crop_region_gpu(region, adaptive_padding=False, create_pil=True)
    assert crop.tensor is not None
    assert crop.tensor.shape[1] == 36
    # Original width 100 -> scale 36/10 = 3.6 -> width = 360
    assert crop.tensor.shape[2] == 360

    cropper.clear_gpu_cache()


def test_paddleocr_vl_batch_accepts_region_crop_and_tensors():
    """Verifies that PaddleOCRVLOcrProvider handles RegionCrop and torch.Tensor inputs."""
    provider = PaddleOCRVLOcrProvider()
    provider._loaded = True
    provider._device = "cpu"

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    # Simulate generate output tensor for batch of 3
    mock_model.generate.return_value = torch.tensor([
        [101, 102, 103, 104],
        [101, 102, 103, 104],
        [101, 102, 103, 104],
    ])
    provider._model = mock_model

    mock_processor = MagicMock()
    mock_processor.tokenizer.padding_side = "left"
    mock_processor.apply_chat_template.return_value = "OCR: prompt"
    mock_processor.return_value = {
        "input_ids": torch.tensor([[101, 102], [101, 102], [101, 102]]),
        "pixel_values": torch.zeros((3, 3, 224, 224)),
    }
    mock_processor.decode.return_value = "Detected Webtoon Text"
    provider._processor = mock_processor

    # Create dummy RegionCrop and Tensor
    crop_tensor = torch.zeros((3, 50, 100), dtype=torch.uint8)
    crop_obj = RegionCrop(
        image=None,
        region_id=1,
        global_origin=(50, 50),
        tensor=crop_tensor,
    )
    pil_img = Image.new("RGB", (100, 50), (255, 255, 255))

    results = provider.recognize_batch(
        images=[crop_obj, crop_tensor, pil_img],
        region_bboxes=[BBox(x1=0, y1=0, x2=10, y2=10)] * 3,
        batch_size=4,
    )

    assert len(results) == 3
    for res in results:
        assert res.text == "Detected Webtoon Text"
        assert res.raw_text == "Detected Webtoon Text"
