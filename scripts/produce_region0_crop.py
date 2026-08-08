#!/usr/bin/env python3
"""Reproduce Region 0 crop with padding=20 from validated bbox."""
import os, sys
from PIL import Image

# try common spellings of "klasör"
CANDIDATES = [
    r"C:\Users\Ahmed\Desktop\Yeni klasor\koharu test",
    r"C:\Users\Ahmed\Desktop\Yeni klasür\koharu test",
    r"C:\Users\Ahmed\Desktop\Yeni klasör\koharu test",
]
SRC = None
for c in CANDIDATES:
    if os.path.isdir(c):
        SRC = c
        break
if SRC is None:
    print("CROP FAIL: Source chapter dir not found")
    print("Tried:", CANDIDATES)
    sys.exit(1)

OUT = r"test_data\output\region_0_crop.png"

X1, Y1, X2, Y2 = 235, 849, 747, 1122
PAD = 20

# Find source chunk that covers y1..y2
chosen = None
for fname in sorted(os.listdir(SRC)):
    if not fname.endswith(".png"):
        continue
    f = os.path.join(SRC, fname)
    img = Image.open(f).convert("RGB")
    w, h = img.size
    cx1 = X1 - PAD
    cy1 = Y1 - PAD
    cx2 = X2 + PAD
    cy2 = Y2 + PAD
    if cx1 >= 0 and cy1 >= 0 and cx2 <= w and cy2 <= h:
        chosen = (fname, img, w, h, cx1, cy1, cx2, cy2)
        break

if chosen is None:
    print("Validated bbox does not fit any chunk with padding. Trying direct crop without padding...")
    for fname in sorted(os.listdir(SRC)):
        if not fname.endswith(".png"):
            continue
        f = os.path.join(SRC, fname)
        img = Image.open(f).convert("RGB")
        w, h = img.size
        if Y2 <= h and X2 <= w and Y1 >= 0 and X1 >= 0:
            chosen = (fname, img, w, h, X1, Y1, X2, Y2)
            print(f"  Found: {fname} size={w,h} (cropped WITHOUT padding)")
            break
    if chosen is None:
        print("CROP FAIL: bbox doesn't fit any chunk")
        sys.exit(1)

fname, img, w, h, cx1, cy1, cx2, cy2 = chosen
print(f"Using source: {fname} size=({w},{h})")
crop = img.crop((cx1, cy1, cx2, cy2))
print(f"Crop box: ({cx1},{cy1},{cx2},{cy2}) size=({cx2-cx1},{cy2-cy1})")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
crop.save(OUT)
print(f"Saved crop to {OUT}")
