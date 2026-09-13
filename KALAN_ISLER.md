# KALAN İŞLER — Detaylı İcra Planı

> Tarih: 2026-09-12. Kapsam: Faz 0–4 (büyük ölçüde) tamamlandı ve push'landı.
> Bu dosya SADECE açık kalan işleri ve her birinin NASIL yapılacağını anlatır.
> Her madde: kanıt → hipotez → dokunulacak dosya/fonksiyon → test → kabul kriteri → doğrulama.
>
> DURUM 2026-09-12 akşam (detay ROADMAP "KALAN-İŞLER TURU" bölümünde):
> İŞ 1 = DURDURULDU (Faz 4-kısıt: köprü denendi, tam-audit reddetti, geri alındı).
> İŞ 2, 3, 5 = BİTTİ (testler yeşil). İŞ 4 = 4.1/4.2/4.3a bitti, 4.3b insan oyu + 4.4 karar AÇIK.
> Doğrulama tam-audit'i arka-planda koşuyor (logs/audit_verify.log).

Genel çalışma disiplini (tüm maddeler için geçerli):
- Yeni diagnostic yok: her değişiklikten sonra `opencode debug lsp diagnostics <dosya>` çalıştırılır; değişen hunk'larda yeni hata kabul edilmez (PySide6/cv2/transformers stub gürültüsü hariç).
- Her davranış değişikliğine sentetik test (gerçek bölüm verisi YOK, eşikler göreli).
- Her görsel iddia mini-koşu (`scripts/fast_verify_mini.py`, ~2 dk, arka planda) veya tam audit (~5 dk) ile kanıtlanır.
- Atomic commit + push; working tree temiz bırakılır.

---

## İŞ 1 — DAMMIT artığı (öncelik: YÜKSEK)

### Kanıt
- `mini_out18/pages/005.png` (0,1700,800,2500) crop: balonda orijinal `DAMMIT ...!` duruyor, Türkçe basılmamış → blok REVIEW'a düşmüş.
- Önceki turlar: mini9 (kısmi hayalet + `Kahretsin!` basıldı), mini11–16 (değişen kısmi temizlik), mini18 (REVIEW, basım yok).
- `mask_overlay.png` (mini16): grown maske `AMM`+noktaları kapsıyor, `D` (sol) ve `IT` (sağ) dışarıda.

### Bilinen gerçekler
- Bölge kutusu kısmi: 182×110 px, gerçek glif açıklığı ~300 px.
- `_expand_mask_in_bubble` flood'u çalışıyor ama gliflere ulaşamıyor.
- İç-ortanca dolgu devrede; kara-leke (P003 sınıfı) riski kapalı.

### Hipotezler (olasılık sırasıyla)
1. **Tohum konumu yanlış:** flood tohumu (`dilate(refined,3) & bg_like`) glif çekirdeklerine değmiyor; tohum beyaz boşlukta kalıyor ve bağlı beyaz bileşen gliflerin ARKASINDA kalıyor olabilir. (Glifler siyah = bg dışı = taşma bariyeri!)
2. **12×/40× tavan erken kesiyor:** `refined` küçükse tavan küçük kalır.
3. **Halka bg kestirimi hâlâ kirli:** 7–25px halka glif-ağırlıklıysa medyan gri çıkar, hiçbir renkle eşleşmez.

### Yapılacaklar (sırayla, her adımda mini18 ile karşılaştır)
1. **Adım 1.1 — Tohum teşhisi (kod yok, 10 dk):** `mini_out18/.../block_0017/` debug setinden `source.png` + `refined_text_mask.png` okunur; tohum piksellerinin gliflere bitişik olup olmadığı gözle doğrulanır. Eğer tohum gliflere değmiyorsa hipotez 1 doğrulanır.
2. **Adım 1.2 — Tohum genişletme (kod, küçük):** `core/imaging/inpainter.py::_expand_mask_in_bubble` içinde tohum çekirdeği 3x3 → 7x7 yapılır ve tohum koşulu gevşetilir: `dilate(refined,7) & (bg_like | refined_dilate_küçük)` — yani tohum, maskeye 7px yakın HER piksel (renk koşulsuz) + bg_like kesişimi bölgesel flood için saklanır. Dosya: `core/imaging/inpainter.py` (~10 satır).
3. **Adım 1.3 — Tavan teşhisi:** tohum sayısı ve bölge büyüklüğü loglanır (geçici `logger.debug`, kalıcı değil). Eğer tavan kesiyorsa `SECOND_CHANCE_MAX_AREA_RATIO` 40 → 60 yapılır (gerekçeyle).
4. **Adım 1.4 — Test:** `tests/test_inpaint_guards.py`'ye tohum-bitişiklik testi eklenir (sentetik: glife 2px mesafedeki beyaz tohumdan flood başlar).
5. **Kabul kriteri:** mini koşuda DAMMIT balonu temiz + `Kahretsin!` basılmış + REVIEW yok. Görsel crop kanıtı.
6. **Geri dönüş:** 2 mini turunda çözülmezse İŞ DURDURULUR, bulgu ROADMAP'e `Faz 4-kısıt` olarak işlenir (kısmi-kutu + sivri-uçlu balon kombinasyonu detector-recall işidir, maske katmanının ötesinde) ve Faz 5'e geçilir. Kör inat yok.

### Riskler
- Aşırı büyüme → beyaz leke (P003 sınıfı). Koruma: renk-kısıtı + tavan + `REVIEW` geri-dönüşü yerinde durur; mini görselde leke kontrolü zorunludur.

---

## İŞ 2 — Faz 3 cila: kilit kasa artığı (öncelik: ORTA)

### Kanıt
- Mini koşularda `WORLD`→`DÜNYA` kilidi cümlelerde `DÜNYAda` / `DÜNYAinde` basıyor (ek doğru, kasa ham). Beklenen: `Dünyada` / `Dünya'inde` (kesme + küçük).
- `normalize_lock_target` + `_tr_titlecase` yazıldı, testli (24 test yeşil) AMA üretimde etki görülmedi.

### Hipotezler
1. Normalize edilen taban cümlede küçük yazılıyor ama sentinel-restore çekim üretirken tabanın KENDİ kasasını koruyor (`_inflect_target("Dünya","loc")` → `Dünyada` olmalı — O HALDE neden `DÜNYAda`?). Demek ki kilitlenen hedef normalize EDİLMEMİŞ biçimde sentinel'e giriyor: ya normalize hook'tan SONRA çalışıyor, ya resolve yolu normalize'i atlıyor (vote/harvest yolu!), ya da `_tr_titlecase` İ-karakterinde beklenen çıktıyı vermiyor.
2. Olası kök: `harvest_confirmed_locks` / vote yolu `normalize_lock_target`'tan GEÇMİYOR (doğrudan `mapping[term] = target` yazıyor olabilir).

### Yapılacaklar
1. **Adım 2.1 — Yol denetimi (kod yok, 15 dk):** `core/translation/chapter_glossary.py` içinde `mapping[...] = ...` yazan TÜM satırlar listelenir; hangilerinin `normalize_lock_target`'tan geçtiği işaretlenir. `resolve_chapter_glossary`, `harvest_confirmed_locks`, vote yolu tek tek okunur.
2. **Adım 2.2 — Düzeltme (kod, küçük):** normalize, mapping'e yazan TEK noktada uygulanır (helper: `_store_lock(mapping, methods, term, target, method)` — tüm yollar bunu çağırır). Böylece gelecek yollar da otomatik normalleşir.
3. **Adım 2.3 — Test:** mevcut `test_normalize_lock_target` korunur + yeni test: vote/harvest yoluyla gelen `DÜNYA` hedefi `mapping`'de `Dünya` olarak görünür (uçtan-uca resolve seviyesinde, stub translator ile).
4. **Kabul kriteri:** mini koşu `glossary.json` içinde `target: "Dünya"` (büyük `DÜNYA` yok) + LSP temiz.
5. **Kapsam dışı:** kesme işareti (`Dünya'inde` vs `Dünyainde`) — morfoloji motorunun (`_inflect_target`, `proper_name` bayrağı) alanıdır; ayrı iş olarak Faz 3-cila-2'ye yazılır, bu işte dokunulmaz.

---

## İŞ 3 — Faz 5 artığı: kısa-ünlem coverage kilidi (öncelik: ORTA)

### Kanıt
- Mevcut tam-audit: 122 kısa-EN-unknown bölgenin 111'i AUTO; kalan 11'i OCR çöpü (doğru REVIEW/SKIP). DIALOGUE-tipli kısa balon: 0 adet (dedektör kısa metni UNKNOWN yazar).
- `DAMMIT...!` vakası detector-recall idi (İŞ 1'in alanı değil — ikinci-şans bandıyla kurtarıldı, mini9 kanıtlı).

### Hüküm
Ortada CANLI bug YOK. Kod yazmak overfit olur. Bu iş = KİLİTLEME işidir, kod işi değil.

### Yapılacaklar
1. **Adım 3.1 — Regresyon testi (kod, küçük):** `tests/test_translation_eligibility.py`'ye kısa-ünlem bekçi testleri eklenir: `DAMMIT...!`, `BUT...`, `HEY!`, `NO... NOT NOW...!` metinli AUTO member'lı bloklar `eligible` dönmelidir (şu anki davranışı mühürler). Hepsi sentetik fixture, mevcut `_member`/`_block` kalıplarıyla.
2. **Adım 3.2 — Metrik bekçisi (kod, küçük):** `scripts/audit_e2e_real_chapter1.py`'nin metrik özetine `short_dialogue_untranslated_count` eklenir: tanımsız (uzunluğu ≤20, alfa içeren, status != auto, tip sfx/watermark olmayan) bölge sayısı. Eşik: bu bölümde 0 (OCR çöpü hariç — `word_difference`/`primary_*` reason'lılar sayılmaz).
3. **Kabul kriteri:** `pytest` yeşil + audit metriği 0 + mini koşuda P005 üst balonlarının tamamı Türkçe (görsel crop).
4. **YapılMAYacaklar:** eligibility eşiği gevşetme, yeni ünlem listesi, dedektör eşiğiyle oynama — kanıtlanmış ihtiyaç yok.

---

## İŞ 4 — Faz 6 model shootout (öncelik: ORTA, Faz 1–5 yeşilken)

### Bağlam
- Üretim varsayılanı: Hy-MT2-7B-Q8_0. Diskte hazır alternatif: TranslateGemma-12B-Q5_K_M (provider tamam, `:8081`). Aday havuzu (12GB tavan ~9-10GB ağırlık): Qwen3-8B-Q8_0 (~8.5GB), EuroLLM-9B-Q5_K_M (~7GB), Gemma3-12B-Q4_K_M (~7.5GB). Yasak: 27B+, 12B-Q8, çift-server, büyük ctx.
- Altyapı mevcut: `scripts/translation_model_shootout_v1.py`, `hy_mt2_karga_real30_*`, `translategemma_quality_gate_v1..5`, `hy_mt2_production_gate_v1.py`.

### Yapılacaklar
1. **Adım 4.1 — Kör set dondurma (kod yok, 30 dk):** Mevcut audit `regions.json`'dan 30 gerçek balon seçilir (kısa ünlem + uzun anlatı + terim-yoğun + SFX-komşu karışık): `benchmark/translation_ab_v1.json` olarak dondurulur (kaynak + blok id). Bir kez yazılır, bir daha değişmez.
2. **Adım 4.2 — Koşu (yarı-otomatik):** Her aday model aynı 30 balonu çevirir (mevcut shootout scriptleri). Çıktılar `benchmark/results/translation_ab_v1/<model>.json` altına tarihli yazılır.
3. **Adım 4.3 — Kör sıralama (insan + kural):** (a) Otomatik: terim-tutarlılık skoru (aynı kaynak terimin farklı karşılık sayısı — `chapter_glossary.extract_*` yeniden kullanılır), yankı-oranı, boşluk-oranı. (b) İnsan: 30 çift kör okuma (sen yaparsın, ben formu hazırlarım).
4. **Adım 4.4 — Karar kapısı:** Kazanan mevcuttan ANLAMLI üstünse (`translation_model_shootout` farkı + kör oy) `config.yaml` default'u değişir (`hy_mt2_production_gate_v1` deseninden geçer). Değilse mevcut kalır — karar dosyaya gerekçeyle yazılır.
5. **Kabul kriteri:** Karar (değişim veya kalış) veriye dayanır ve `ROADMAP.md` Faz 6 satırına işlenir.
6. **Süre notu:** Çoğu süre indirme + GPU koşu beklemesidir; işler arka-plana atılır, arada başka işe devam edilir.

### Riskler
- Yeni model indirme `.venv`'i etkilemez (GGUF + harici llama-server deseni korunur); yeni provider = klon deseni (30 dk/model): provider dosyası + registry alias + config. `chapter_analyzer.py`'ye DOKUNULMAZ.

---

## İŞ 5 — Faz 7 kalıcı gate'ler (öncelik: ORTA, Faz 6 sonrası)

### Amaç
Bu turda kazanılan her kuralı bir daha kırılamayacak hale getirmek.

### Yapılacaklar
1. **Adım 5.1 — Audit eşikleri (kod, orta):** `scripts/audit_e2e_real_chapter1.py` sonuna gate bloğu eklenir: `logo_pixel_diff == 0` (P002 logo hash), `duplicate_render_count == 0`, `untranslated_story_count == 0` (İŞ 3 metriği), `term_consistency < eşik`. Eşik aşılırsa script non-zero çıkar. Eşikler bu dosyanın kanıt turlarındaki ÖLÇÜLMÜŞ değerlerden alınır (uydurma yok).
2. **Adım 5.2 — Test envanteri (kod yok, 20 dk):** `tests/test_*.py` dosyaları ROADMAP Faz 1–5 maddeleriyle eşleştirilir; eşleşmeyen faz maddesi varsa o fazın testi yazılır (eksik halka kapatılır).
3. **Adım 5.3 — `review_output/` kusur raporu (kod, orta):** Her koşuda şüpheli blok crop'ları + sayfa listesi otomatik üretilir (insan REVIEW turu kör olmaz). Mevcut `analysis/inpainting_debug/` + `regions.json` girdidir; yeni model koşusu YOK.
4. **Kabul kriteri:** `pytest -q` + audit gate yeşil olmadan merge yok kuralı `AGENTS.md`'ye yazılır.

---

## İŞ 6 — Faz 8 performans (öncelik: DÜŞÜK, ERTELENDİ)

- Mevcut: ~261 sn/bölüm (25 sayfa), mini ~2 dk. Kabul edilebilir.
- Şart: Faz 1–7 bitmeden DOKUNULMAZ.
- Adaylar (sırayla, her biri ayrı mini-kanıtlı): çeviri chunk 32 → adaptif; LaMa batch 24 tavanı; OCR gated 398/602 oranı (gereksiz crop analizi).
- Kabul kriteri: aynı görsel çıktı + %20+ süre kazancı (audit `elapsed_seconds` karşılaştırmalı).

---

## YÜRÜTME SIRASI (önerilen)

```
İŞ 1 (DAMMIT artığı) → İŞ 2 (kasa cilası) → İŞ 3 (coverage kilidi)
   → İŞ 4 (shootout) → İŞ 5 (gate'ler) → İŞ 6 (perf, erteli)
```

- İŞ 1 ile İŞ 2 bağımsızdır (farklı dosyalar), istenirse paralel yürütülebilir; ama tek akışta İŞ 1 önce (görsel kusur > kozmetik).
- İŞ 4, İŞ 1–3 yeşilken başlar (temiz zeminde model farkı ölçülür).
- Her işin sonunda: test + LSP + mini/tam kanıt + atomic commit + ROADMAP satırı.
