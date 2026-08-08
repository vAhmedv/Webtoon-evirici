#!/usr/bin/env python3
"""Reproduce Region 0 crop with padding=20 from validated bbox.

Uses the project's RegionCropper + chapter loader + global coordinate system
instead of randomly picking the first PNG that fits the bbox.
"""
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, Region, RegionStatus, RegionType
from core.imaging.region_cropper import RegionCropper
from core.io.input_loader import load_chapter

# Source chapter (verified path)
SRC = r"C:\Users\Ahmed\Desktop\Yeni klasör\koharu test"
OUT = r"test_data\output\region_0_crop.png"

# Validated Region 0 bbox (global coordinates)
X1, Y1, X2, Y2 = 235, 849, 747, 1122
PAD = 20


def main():
    if not os.path.isdir(SRC):
        print("CROP FAIL: Source chapter dir not found")
        print(f"Tried: {SRC}")
        sys.exit(1)

    # Load chapter via project loader (assigns y_offsets, validates widths)
    try:
        pages = load_chapter(SRC)
    except Exception as e:
        print(f"CROP FAIL: Chapter load error: {e}")
        sys.exit(1)

    coords = GlobalCoordinateSystem(tuple(pages))

    # Build a Region with the validated global bbox
    region = Region(
        id=0,
        global_bbox=BBox(x1=X1, y1=Y1, x2=X2, y2=Y2),
        type=RegionType.DIALOGUE,
        detection_confidence=1.0,
        source_window_ids=(0,),
        status=RegionStatus.AUTO,
    )

    # Crop via RegionCropper (correct source chunk / coordinate mapping)
    cropper = RegionCropper(pages, coords, padding=PAD)
    try:
        crop = cropper.crop_region(region)
    except Exception as e:
        print(f"CROP FAIL: RegionCropper error: {e}")
        sys.exit(1)

    img = crop.image
    w, h = img.size
    print(f"Source chunk(s): {crop.page_indices}")
    print(f"Crop box: ({crop.global_origin[0]},{crop.global_origin[1]}) "
          f"size=({w},{h})")

    # Validation: expected ~552x313
    if not (530 <= w <= 570 and 290 <= h <= 330):
        print(f"CROP FAIL: Unexpected crop size {w}x{h} "
              f"(expected ~552x313, range 530-570 x 290-330)")
        sys.exit(2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT)
    print(f"CROP PASS: {w}x{h}")
    print(f"Saved crop to {OUT}")


if __name__ == "__main__":
    main()