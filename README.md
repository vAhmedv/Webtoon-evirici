# Webtoon Çevirici

Windows üzerinde İngilizce webtoon bölümlerini tamamen yerel olarak Türkçeye çeviren production pipeline ve PySide6 masaüstü uygulaması.

## Production akışı

`ChapterAnalyzer.process_chapter()` tek orchestration kaynağıdır:

1. ComicTextDetector (YOLOv5 backbone + UNet text-mask + DBNet line head, tek provider) global koordinatlarda tespit üretir; window-çakışmaları `merge_duplicates` ile birleştirilir.
2. CTD validity gate açıkça non-text olan crop'ları pahalı OCR repair aşamalarından önce eler.
3. PP-OCRv6 primary OCR'dır; `min_confidence<0.85`/şüpheli sonuçlarda PaddleOCR-VL verifier çalışır (registry: `PaddleOCR-PP-OCRv6` + `PaddleOCR-VL-1.6`).
4. Qwen3.5-9B yalnız uygun, çözülmemiş OCR uyuşmazlıklarında görsel fallback'tir.
5. Uygun üyelerden oluşan TextBlock'lar Hy-MT2-7B-Q8_0 ile EN→TR çevrilir (blok-veto yok: SFX/REVIEW üyeler atlanır, uygun üyeler kısmi çevrilir).
6. LaMa yalnız onaylı text maskesi içindeki İngilizce glyph'leri temizler.
7. Renderer yalnız başarılı inpaint bloklarını çizer ve yatay/dikey overflow'u raporlar.
8. Global canvas kaynak sayfa sayısı korunarak output klasörüne yeniden bölünür.

SKIP bloklar çevrilmez/inpaint edilmez/render edilmez. Kısmi bloklarda yalnız uygun (AUTO + hikaye-tipi) üyeler çevrilir ve render edilir; SFX/REVIEW/WATERMARK üyeler ile inpaint-REVIEW bloklarında orijinal İngilizce pikseller korunur. Kaynak görseller hiçbir aşamada yazılmaz; output klasörü source klasöründen farklı olmak zorundadır (`output_exporter` + `ChapterAnalyzer` guard).

## Yapılandırma

Provider/model yollarının uygulama kaynağı `config.yaml` dosyasıdır. Provider isimleri registry anahtarlarıyla birebir olmalıdır (`DetectorRegistry` / `OCRRegistry.resolve_name()`); bilinmeyen isim worker'da net hatayla listelenir. `validate_config()` pencere/port/isim tutarlılığını kontrol eder. Varsayılan yerel roller:

- Detector: ComicTextDetector
- Primary OCR: `PaddleOCR-PP-OCRv6` (model: `PP-OCRv6_medium_rec`)
- Verifier: PaddleOCR-VL 1.6
- OCR repair: Qwen3.5-9B Q5_K_M GGUF + mmproj
- Translator: Hy-MT2-7B Q8_0 GGUF
- Inpainter: LaMa Large

Server portları merkezi haritadadır (`core/system/ports.py`): Hy-MT2 8085, Qwen-repair 8086; deneysel 8080-8083.

Model yollarını kendi makinenize göre `config.yaml` içinde değiştirin. Externally owned llama-server süreçleri öldürülmez; provider yalnız başlattığı PID'yi yönetir.

## Kullanım

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Detector/OCR ekstraları yalnızca gerekiyorsa (env'i korumak için ayrı):
# pip install -r requirements-detector.txt
# pip install -r requirements-ocr.txt
.venv\Scripts\python.exe -m pytest -q
```

GUI'yi proje entrypoint'iyle başlatın, source chapter ve ayrı output klasörü seçip Translate/Analyze düğmesini kullanın. Ağır modeller worker thread içinde çalışır; progress, cancellation, REVIEW sayısı ve output klasörü UI'da gösterilir.

Gerçek production doğrulaması:

```powershell
.venv\Scripts\python.exe scripts\audit_e2e_real_chapter1.py
```

## Output yapısı

- Çevrilmiş sayfalar: output kökü
- `analysis/regions.json`: region, TextBlock ve lifecycle metrikleri
- `analysis/summary.json`: final AUTO/REVIEW/SKIP ve render özeti
- `analysis/inpainting_debug/`: mask/inpaint diagnostic artefaktları

Pipeline REVIEW durumunu başarı gibi gizlemez. Kullanıcı REVIEW bölgelerini veya inpaint bloklarını incelemelidir.
