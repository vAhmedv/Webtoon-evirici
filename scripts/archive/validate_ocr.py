import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import time
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
from dataclasses import replace

from core.config import load_config
from application.chapter_analyzer import ChapterAnalyzer
from providers.detector.registry import get_registry
from providers.ocr.registry import get_ocr_registry

# Create synthetic 8-page chapter
tmp = tempfile.mkdtemp()
chapter = Path(tmp) / 'chapter'
chapter.mkdir()

page_height = 4546
for i in range(8):
    img = Image.new('RGB', (800, page_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y_positions = [200, 800, 1300, 1800, 2300, 2800, 3300, 3800, 4200]
    for j, y in enumerate(y_positions):
        if y < page_height - 100:
            draw.text((60, y), f'Page {i} dialogue line {j}', fill=(20, 20, 20))
    draw.rectangle((500, 400, 750, 600), outline=(0, 0, 0), width=4)
    draw.text((520, 480), 'SFX', fill=(0, 0, 0))
    img.save(chapter / f'{i:03d}.webp', 'WEBP', quality=95)

output = Path(tmp) / 'output'
output.mkdir()

config = replace(load_config(), window_height=5000, window_overlap=1000)
analyzer = ChapterAnalyzer(config)
detector = get_registry().create('YOLOv8 Comic Text Segmenter')
ocr = get_ocr_registry().create('RapidOCR-ONNX')

print('=== Phase 3D + 4A Real Chapter Validation ===')
print(f'Chapter: 8 pages, {8 * page_height} px total')
print(f'Windows: window_height={config.window_height}, overlap={config.window_overlap}')

start = time.time()
result = analyzer.analyze(
    chapter_path=chapter,
    output_path=output,
    detector=detector,
    progress_callback=lambda e: None,
    ocr_provider=ocr,
)
elapsed = time.time() - start

print(f'\n=== Results ===')
print(f'Pages: {len(result.pages)}')
print(f'Windows: {len(result.windows)}')
print(f'Regions: {len(result.regions)}')
print(f'AUTO: {result.auto_count}')
print(f'REVIEW: {result.review_count}')
print(f'SKIP: {result.skip_count}')
print(f'OCR elapsed: {result.ocr_elapsed_time:.2f}s')
print(f'Total runtime: {elapsed:.2f}s')

# Check polygon global coordinates
print('\n=== Polygon Global Coordinate Check ===')
for i, reg in enumerate(result.regions[:3]):
    poly = reg.metadata.get('polygon')
    if poly:
        ys = [p[1] for p in poly]
        match = all(reg.global_bbox.y1 <= y <= reg.global_bbox.y2 for y in ys)
        print(f'Region {reg.id}: global_bbox.y={reg.global_bbox.y1}-{reg.global_bbox.y2}, polygon y={min(ys):.0f}-{max(ys):.0f}, match={match}')

# OCR results
print('\n=== OCR Results ===')
ocr_success = [r for r in result.regions if r.text]
ocr_blank = [r for r in result.regions if not r.text]
print(f'OCR success: {len(ocr_success)}')
print(f'OCR blank: {len(ocr_blank)}')

if ocr_success:
    avg_conf = sum(r.ocr_confidence for r in ocr_success) / len(ocr_success)
    print(f'Average OCR confidence: {avg_conf:.2f}')

print('\n=== Sample OCR Results ===')
for reg in result.regions[:5]:
    if reg.text:
        print(f'Region {reg.id}: "{reg.text[:60]}" (conf={reg.ocr_confidence:.2f})')
    else:
        print(f'Region {reg.id}: <blank>')

print('\n=== Done ===')
