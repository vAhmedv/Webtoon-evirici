# Webtoon Çevirici

İngilizce manga/manhwa/webtoon bölümlerini tamamen local olarak Türkçeye çeviren Windows masaüstü uygulaması.

- **Tamamen local:** API key yok, cloud yok, veri dışarı gitmez
- **NVIDIA RTX 5070 desteği:** CUDA 12.8+ (Blackwell mimarisi)
- **Kalite öncelikli:** Hız değil, kalite

## Kurulum (Windows)

### 1. Python 3.11 kur

Terminalde:
```
winget install Python.Python.3.11
```

Kurulumdan sonra terminali kapatıp yeniden aç. Doğrulama:
```
python --version
```
Beklenen: `Python 3.11.x`

### 2. Git deposu başlat

```
git init
```

### 3. Sanal ortam oluştur ve aktifleştir

```
python -m venv .venv
.venv\Scripts\activate
```
Komut satırının başında `(.venv)` görünmeli.

### 4. PyTorch kur (RTX 5070 için CUDA 12.8)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Bu indirme ~2.5 GB'dir. 5-20 dakika sürebilir.

### 5. Diğer paketleri kur

```
pip install -r requirements.txt
```

### 6. GPU doğrulama

```
python scripts/check_gpu.py
```

Beklenen çıktı:
```
PyTorch sürümü: 2.x.x
CUDA available: True
GPU 0: NVIDIA GeForce RTX 5070
  CUDA capability: 12.0
  Toplam VRAM: 12.0 GB
GPU hesaplama testi: BAŞARILI
Sonuç: GPU doğru çalışıyor. Phase 0 kabul testi geçti.
```

### 7. Testleri çalıştır

```
pytest tests/ -v
```

Beklenen: Tüm testler `PASSED`.

## Phase 1 Kabul Testi

### 1. Sentetik test bölümü üret

```
python scripts/generate_test_chapter.py
```

### 2. Pipeline'ı çalıştır

```
python scripts/process_chapter.py --input test_data/chapter_test --output test_data/output
```

### 3. Önizlemeleri kontrol et

`test_data/output/windows/` klasöründe window önizleme görselleri oluşur. Kırmızı çerçeve window sınırını, mavi yatay çizgiler sayfa sınırlarını gösterir.

## Proje Yapısı

```
core/          # Çekirdek motor (UI'dan bağımsız)
providers/     # Model sağlayıcıları (ileride)
ui/            # Windows GUI (Phase 9)
scripts/       # Komut satırı araçları
tests/         # Testler
config.yaml    # Tüm ayarlar
```

## Faz Durumu

- [x] Phase 0: Ortam + GPU doğrulama (kod hazır)
- [x] Phase 1: Input loader + global koordinat + sliding window (kod hazır)
- [ ] Phase 2: Detector interface + görselleştirme
- [ ] Phase 3: Gerçek detector
- [ ] Phase 4: OCR
- [ ] Phase 5: Qwen + çeviri
- [ ] Phase 6: Inpainting
- [ ] Phase 7: Renderer
- [ ] Phase 8: Re-split + output
- [ ] Phase 9: Windows GUI
- [ ] Phase 10: Suwayomi otomasyonu