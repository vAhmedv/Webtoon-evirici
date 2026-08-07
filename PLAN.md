# Webtoon Çevirici — Phase 0 & Phase 1 Planı

## Genel Bakış

Bu doküman, İngilizce webtoon/manhwa bölümlerini tamamen local olarak Türkçeye çeviren Windows masaüstü uygulamasının **Phase 0** ve **Phase 1** planını içerir.

- **Phase 0:** Geliştirme ortamı hazırlığı + GPU doğrulama
- **Phase 1:** Girdi yükleme + global koordinat sistemi + sliding window

Bu iki phase'te **hiçbir AI modeli kurulmaz**. Yalnızca sağlam temel atılır: Python ortamı kurulur, GPU'nun doğru çalıştığı kanıtlanır, bölüm okuma ve pencereleme altyapısı yazılır ve test edilir.

---

## 1. Önerilen Teknoloji Yığını

### Phase 0 & 1'de kurulacaklar

| Bileşen | Sürüm | Neden |
|---|---|---|
| Python | 3.11.x | Tüm kütüphanelerle uyumlu, kararlı sürüm |
| Git | mevcut | Sürüm kontrolü, her kilometre taşında commit |
| PyTorch | 2.7+ (cu128) | RTX 5070 (Blackwell) desteği için CUDA 12.8 build'i. GPU doğrulama + ileride tüm modellerin temeli |
| Pillow | en son | WEBP/PNG/JPG görüntü okuma-yazma |
| numpy | en son | Görüntü ve koordinat hesapları |
| opencv-python | en son | Görüntü işleme, window önizleme görselleri |
| PyYAML | en son | config.yaml okuma |
| loguru | en son | Konsol + dosyaya log |
| tqdm | en son | İlerleme çubukları |
| pytest | en son | Test sistemi |

### İleride eklenecekler (şimdi KURULMAZ)

| Bileşen | Phase | Neden |
|---|---|---|
| Detector modeli | 3 | Benchmark ile seçilecek |
| OCR motoru | 4 | Benchmark ile seçilecek |
| llama-cpp-python (CUDA) | 5 | Qwen GGUF modelini GPU'da çalıştırma |
| Inpainting modeli | 6 | Benchmark ile seçilecek |
| PySide6 | 9 | Windows masaüstü arayüzü |

---

## 2. Proje Klasör Yapısı

```
Webtoon cevirici/
├── config.yaml                  # Tüm ayarlar (window boyutları, overlap vb.)
├── requirements.txt             # Sabitlenmiş paket listesi
├── README.md                    # Kullanım kılavuzu
├── PLAN.md                      # Bu plan
├── .gitignore
├── core/                        # Çekirdek motor (UI'dan bağımsız)
│   ├── __init__.py
│   ├── config.py                # config.yaml yükleyici
│   ├── logging_setup.py         # Log sistemi (loguru)
│   ├── models/                  # Veri modelleri (dataclass)
│   │   ├── __init__.py
│   │   ├── page.py              # Sayfa: index, yol, genişlik, yükseklik, y_offset
│   │   └── window.py            # Pencere: id, y_start, y_end, sayfa aralığı
│   ├── io/                      # Girdi/çıktı
│   │   ├── __init__.py
│   │   └── input_loader.py      # Bölüm klasörünü okur, doğal sıralar
│   └── coordinate/              # Koordinat sistemi
│       ├── __init__.py
│       ├── global_coords.py     # Global koordinat sistemi
│       └── sliding_window.py    # Sliding window üretici
├── providers/                   # Model sağlayıcıları (ileride doldurulacak)
│   ├── __init__.py
│   ├── detector/                # Phase 3
│   ├── ocr/                     # Phase 4
│   ├── translator/              # Phase 5
│   └── inpainter/               # Phase 6
├── ui/                          # Phase 9 (GUI)
├── scripts/
│   ├── check_gpu.py             # GPU doğrulama scripti
│   ├── generate_test_chapter.py # Sentetik test bölümü üretir
│   └── process_chapter.py       # Phase 1 pipeline'ını çalıştırır
├── tests/
│   ├── test_input_loader.py
│   ├── test_global_coords.py
│   └── test_sliding_window.py
├── test_data/                   # Sentetik test bölümü (gitignore)
├── projects/                    # Gerçek bölüm çalışma verileri (gitignore)
└── logs/                        # Loglar (gitignore)
```

---

## 3. Windows Geliştirme Ortamı Hazırlığı

### Adım 1 — Python 3.11 kur
Terminalde şu komutu çalıştır:
```
winget install Python.Python.3.11
```
Kurulum bittikten sonra terminali kapatıp yeniden aç. Doğrulama:
```
python --version
```
Beklenen çıktı: `Python 3.11.x`

### Adım 2 — Git deposu başlat
```
git init
```

### Adım 3 — Sanal ortam oluştur ve aktifleştir
```
python -m venv .venv
.venv\Scripts\activate
```
Komut satırının başında `(.venv)` yazısı görünmeli.

### Adım 4 — PyTorch kur (RTX 5070 için CUDA 12.8)
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Bu indirme yaklaşık 2.5 GB'dir; internet gerekir. RTX 5070 (Blackwell mimarisi) yalnızca CUDA 12.8+ build'leriyle çalışır.

### Adım 5 — Diğer paketleri kur
```
pip install -r requirements.txt
```

### Adım 6 — GPU doğrulama
```
python scripts/check_gpu.py
```

---

## 4. Dependency Gerekçeleri

- **PyTorch (cu128):** RTX 5070, NVIDIA'nın Blackwell mimarisidir (sm_120). Yalnızca CUDA 12.8+ sürümleri bu GPU'yu destekler. PyTorch'un resmi cu128 wheel'leri bu desteği sağlar. Phase 0'da kurulmasının amacı, daha hiçbir model yazmadan GPU'nun doğru çalıştığını kanıtlamaktır. İleride detector, OCR ve inpainting modellerinin tamamı PyTorch üzerinde çalışacak.
- **Pillow:** Python'un standart görüntü kütüphanesi. WEBP dahil tüm formatları okur/yazar. Bölüm sayfalarını yüklemek ve çıktı yazmak için gerekli.
- **numpy:** Görüntüleri sayısal dizilere çevirir. Koordinat hesapları ve tüm görüntü işlemenin temeli.
- **opencv-python:** Gelişmiş görüntü işleme (maskeleme, genişletme, önizleme çizimi). Phase 1'de window önizlemeleri için, ileride inpainting compositing için gerekli.
- **PyYAML:** Tüm ayarlar config.yaml'da tutulacak. Bu paket onu okur.
- **loguru:** Konsola ve logs/latest.log dosyasına temiz log yazar. Hata analizi için kritik.
- **tqdm:** Uzun işlemlerde ilerleme çubuğu gösterir.
- **pytest:** Test sistemi. Her phase'in kabul testleri pytest ile çalıştırılacak.

---

## 5. Phase 0 Kabul Testleri

1. `python --version` → `Python 3.11.x` görünür.
2. `python scripts/check_gpu.py` → şu bilgiler görünür:
   - PyTorch sürümü
   - CUDA available: **True**
   - GPU adı: **NVIDIA GeForce RTX 5070**
   - VRAM: **~12 GB**
   - GPU hesaplama testi: **Başarılı**
3. `pytest tests/ -v` → tüm testler **PASSED**.
4. `logs/latest.log` dosyası oluşur ve başlangıç bilgilerini içerir.

---

## 6. Phase 1 Kabul Testleri

1. `python scripts/generate_test_chapter.py` → `test_data/chapter_test/` içinde 5 WEBP dosyası oluşturur (farklı yüksekliklerde, doğru sırada: 001.webp ... 005.webp).
2. `python scripts/process_chapter.py --input test_data/chapter_test --output test_data/output` → hatasız çalışır.
3. Konsol çıktısında:
   - Sayfa sırası doğru: 001, 002, 003, 004, 005
   - Toplam yükseklik
   - Oluşturulan window sayısı ve aralıkları
4. `test_data/output/windows/` klasöründe window önizleme görselleri oluşur. Kullanıcı bunları açıp overlap bölgelerini görsel olarak doğrular.
5. `pytest tests/ -v` → tüm testler **PASSED**.
6. `logs/latest.log` aşama sürelerini içerir.

---

## 7. Phase 1 Mimari Detayları

### Input Loader (core/io/input_loader.py)
- Klasördeki tüm görüntü dosyalarını bulur (webp, png, jpg, jpeg).
- **Doğal sıralama** yapar: `1, 2, 10` yerine `001, 002, 010` gibi doğru sıra.
- Her sayfanın genişlik/yüksekliğini okur.
- Tüm sayfaların aynı genişlikte olduğunu doğrular (webtoon sayfaları aynı genişlikte olmalı).
- Sonuç: `Page` nesneleri listesi.

### Global Koordinat Sistemi (core/coordinate/global_coords.py)
- Her sayfaya kümülatif `y_offset` atar:
  - Sayfa 0: y 0–1000
  - Sayfa 1: y 1000–1840
  - Sayfa 2: y 1840–2840
  - ...
- İki yönlü dönüşüm sağlar:
  - `page_to_global(sayfa_no, yerel_y)` → global y
  - `global_to_page(global_y)` → (sayfa_no, yerel_y)
- Bu, Suwayomi'nin parçaladığı WEBP'lerin tek uzun webtoon gibi düşünülmesini sağlar.

### Sliding Window (core/coordinate/sliding_window.py)
- `window_height` ve `overlap` config.yaml'dan okunur (varsayılan: 5000 px, 1000 px).
- Örnek:
  - Window 0: y 0–5000
  - Window 1: y 4000–9000
  - Window 2: y 8000–13000
- Her window için hangi sayfaların dahil olduğunu hesaplar.
- Kenar durumları yönetir: son window kısa olabilir, window yüksekliği toplamdan büyükse tek window oluşur.

### Görselleştirme (scripts/process_chapter.py)
- Her window'u küçültülmüş önizleme görseli olarak kaydeder.
- Window sınırlarını, sayfa sınırlarını ve window ID'sini görsele çizer.
- Böylece kullanıcı overlap mantığını gözle doğrulayabilir.

---

## 8. Riskler ve Notlar

- **PyTorch indirmesi büyüktür** (~2.5 GB). İnternet hızına bağlı olarak 5–20 dk sürebilir.
- **GPU doğrulama başarısız olursa** ilerlemeden önce kök nedeni analiz ederiz (sürücü, CUDA, PyTorch sürümü).
- **Hiçbir AI modeli bu phase'lerde kurulmaz.** Yalnızca temel altyapı.
- Kaynak görüntüler asla değiştirilmez; tüm çıktılar ayrı klasöre yazılır.

---

## 9. Onay Sonrası Uygulama Sırası

1. **Phase 0 uygulama:** Tüm dosyaları oluştururum (config, requirements, core iskeleti, check_gpu.py, testler). Sana çalıştırma komutlarını veririm. Sen çalıştırıp sonuçları bana iletirsin.
2. **Phase 0 kabul testleri:** GPU doğrulama + pytest geçer. Git commit atarız.
3. **Phase 1 uygulama:** input_loader, global_coords, sliding_window, test bölümü üretici, process_chapter.py.
4. **Phase 1 kabul testleri:** Sentetik bölüm üretilir, pipeline çalışır, window önizlemeleri görsel olarak doğrulanır. Git commit atarız.
5. Ancak bundan sonra Phase 2 (detector) planlanır.