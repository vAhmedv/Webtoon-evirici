"""Tests for Flat Bubble Fast-Path inpainting optimization."""

import numpy as np
import pytest
from core.imaging.inpainter import Inpainter
from core.imaging.text_mask import TextMask


def test_can_use_flat_fill_solid_white_bubble():
    # 100x100 solid white image with a small central text mask
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    can_fill, color = Inpainter._can_use_flat_fill(image, mask)
    assert can_fill is True
    assert color == (255, 255, 255)


def test_can_use_flat_fill_solid_tint_bubble():
    # 100x100 solid light-blue tint
    image = np.full((100, 100, 3), (240, 245, 250), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255

    can_fill, color = Inpainter._can_use_flat_fill(image, mask)
    assert can_fill is True
    assert color == (240, 245, 250)


def test_can_use_flat_fill_complex_artwork_gradient():
    # 100x100 complex gradient/texture image
    gradient_x = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
    gradient_y = np.tile(np.linspace(0, 255, 100, dtype=np.uint8).reshape(100, 1), (1, 100))
    image = np.stack([gradient_x, gradient_y, np.full((100, 100), 128, dtype=np.uint8)], axis=-1)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    can_fill, _ = Inpainter._can_use_flat_fill(image, mask)
    assert can_fill is False


def test_apply_flat_fill_with_soft_blend():
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    # Add black text in the center
    image[20:30, 20:30] = [0, 0, 0]
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 255

    blended = Inpainter._apply_flat_fill_with_soft_blend(image, mask, (255, 255, 255))
    assert blended.shape == (50, 50, 3)
    # The center should be replaced by white
    assert np.all(blended[22:28, 22:28] == [255, 255, 255])
    # The corners outside the mask should remain exact source white
    assert np.all(blended[0:10, 0:10] == [255, 255, 255])
