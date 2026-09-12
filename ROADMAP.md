# WEBTOON ÇEVİRİCİ — ACIMASIZ ROADMAP

> Tarih: 2026-09-12. Kaynak: Chapter 1 eski output (18.08) + güncel kodla üretilmiş taze output (`audit_output/real_chapter1_e2e`, 254 sn) sayfa-sayfa görsel QA, regions/summary metrikleri, BenchLM Eyl 2026 + TranslateGemma raporu, VRAM denetimi (RTX 5070 12GB).
> Kural: Hiçbir faz, acceptance kriteri yeşile dönmeden kapanmaz. Kriteri tutmayan iş "bitti" sayılmaz.

---

## FAZ 0 — GÜVENLİK VE HİJYEN (hemen, ~30 dk, kod yok)

**Neden önce bu:** Repo şu an kirli ve tehlikeli durumda. Bunun üstüne inşa edilmez.

- [ ] 0.1 `config.yaml` içinde **gerçek Gemini API key** var ve dosya tracked. Seçenek: key'i `GEMINI_API_KEY` env'ine taşı + `git update-index --skip-worktree config.yaml`, veya config'i ikiye böl (`config.yaml` + gitignore'lı `config.local.yaml`). Kriter: `git diff` ve `git status` içinde key görünmüyor.
- [ ] 0.2 Working tree'de 25 dosyada ~690 satır uncommitted değişiklik duruyor. Review et, atomic commit'lere böl, push'la. Kriter: `git status` temiz.
- [ ] 0.3 `output/`, `audit_output/`, `e2e_output/`, model ağırlıkları gitignore'lı mı doğrula (kriter: `git status --ignored` içinde yoklar).

---

## FAZ 1 — LOGO/ART KORUMASI (en ağır görsel kusur)

**Kanıt:** P002 "Zero Fantasy" logosu inpaint'le silinip üstüne `BİR/ÇEV-/RİMİÇİ` (eski) → `AN/SY/ONLINE` (yeni) basıldı + siyah lekeler. P024 `STUDIO ZOON` → `STÜDYO ZOON`. Mevcut `art/logo protection` (18.08 commiti) yetersiz.

- [ ] 1.1 Logo bölgesi tespiti: sayfanın üst %15'indeki, büyük puntolu, örtüşen/ bitişik, düşük OCR güvenli yazı kümeleri → otomatik SKIP (çeviri yok, inpaint yok, render yok). Heuristik: yükseklik > medyan satır yüksekliği × 2.5 VEYA genişlik > sayfa genişliği × 0.4 VEYA dedektör kutu-içi-kutu çakışması.
- [ ] 1.2 Stilize yazıredd i: OCR confidence < 0.5 VE harf yüksekliği varyansı yüksek olan region'lar story-text sayılmasın → REVIEW'a düşsün, AUTO'ya çıkmasın.
- [ ] 1.3 Regresyon testi: `tests/test_logo_protection.py` — P002 logo kutuları + P024 STUDIO kutusu fixture, pipeline sonrası piksellerin kaynakla birebir aynı olduğu assert edilir (hash karşılaştırma).
- **Kriter:** Yeni Chapter 1 çıktısında logo pikselleri kaynakla %100 aynı, logo üstünde Türkçe kırıntı yok.

---

## FAZ 2 — PARÇA/ÇİFT RENDER TEMİZLİĞİ (en yaygın yapısal kusur)

**Kanıt:** P002 `DÜNYANIN`×2 → yeni çıktıda `THE` + cümle ayrı basım. P003 `I` + cümle ayrı. P018 `ÜRETEMEM. ÜRETEMEM.` P024 `Bu sefer,` + `başaracağım...!` üst üste. P017 sahipsiz `"` artığı, P003 `iz"bırakmak`.

- [ ] 2.1 Renderer'a kural: çeviriye girmeyen üye (SFX/REVIEW/uygun-değil) **hiçbir koşulda** render edilmez — ne tek başına ne blok parçası olarak. Üye-bazlı filtre (11.09 değişikliği) çeviride var, render'da eksik; oradaki boşluk kapatılacak.
- [ ] 2.2 Duplicate render guard: aynı blok iki kez render edilmeye çalışılırsa ikinci çağrı log + skip (blok-id bazlı render kaydı).
- [ ] 2.3 Noktalama bonding denetimi: `punctuation bonding` (18.08) sahipsiz tırnak/kesme üretiyor — `"` ve `'` yalnızca kelimeye yapışıkken korunur, tek başına kalan noktalama render edilmez. `iz"bırakmak` vakası için test.
- [ ] 2.4 Regresyon testi: P002 üst blok, P018, P024 blokları için "çıktıda her blok metni tam 1 kez geçiyor" assert'i.
- **Kriter:** Yeni çıktıda çift/parça/sahipsiz-noktalama renderı sıfır (regions + piksel OCR ile doğrulanır).

---

## FAZ 3 — TERMİNOLOJİ KİLİDİ (çeviri tutarlılığı)

**Kanıt:** `CRAFTER` → USTA / zanaatkar / ÜRETİCİ / işleyici / "Üretici" aynı bölümde karışık. Altyapı var (`qwen_glossary`, `__WTTERM__` sentinel, candidate store) ama production yoluna bağlı değil.

- [ ] 3.1 `APPROVED TERMS` + sentinel koruması production çeviri yoluna (Hy-MT2 VE TranslateGemma) bağlanacak. Kriter: `CRAFTER` bölümün tamamında tek karşılık.
- [ ] 3.2 Bölüm glossary'si `analysis/` altına `glossary.json` olarak yazılacak (terim, ilk görüldüğü sayfa, kullanım sayısı) — REVIEW turunda insan düzeltebilsin.
- [ ] 3.3 `qwen_glossary` offline resolver eşiği gözden geçirilecek (2+ gözlem kuralı kısa bölümlerde terim üretemiyor olabilir).
- **Kriter:** Top-20 sık terimin bölüm-içi tutarlılığı %100 (otomatik script ile ölçülür).

---

## FAZ 4 — KOYU ZEMİN INPAINT (haleler ve taşmalar)

**Kanıt:** P003 siyah zeminde beyaz yama; P023 anlatı kutusunda bulaşık haleler; P024 mavi-gri yama kutusu.

- [ ] 4.1 Flat-fill maskesi balon sınırına kırpılacak — 7x7 dilation koyu zeminde halo üretiyor; dilation miktarı zemin parlaklığına göre adaptif olacak (koyu zeminde küçük, açık zeminde normal).
- [ ] 4.2 Inpaint sonrası halo dedektörü: blok çevresi 8px bantta kaynakla fark > eşikse bloğu REVIEW'a işaretle (sessizce geçirme).
- [ ] 4.3 `inpainting_debug/` artefaktları REVIEW blokları için saklanacak (şu an sadece AUTO izlenebiliyor).
- **Kriter:** P003/P023/P024 tipi halo/yama vakaları sıfır; kalan şüpheliler REVIEW'da görünür.

---

## FAZ 5 — ÇEVRİLMEYEN BALONLAR VE OCR ARTIKLARI

**Kanıt:** P005 `DAMMIT...!`, P008/P017 kısa ünlemler çevrilmemiş (kısmen düzeldi: `BUT...`→`AMA…` yeni çıktıda OK). `OFMY` kaynaşması, `Sv.1` (düzeldi), `A "Üretici" demek...` İngilizce artığı.

- [ ] 5.1 Kısa ünlem kuralı: tamamı-büyük-harf ≤ 12 karakter hikâye balonları (`DAMMIT...!`, `WHAT?!`) doğrudan çeviriye girer, "SFX olabilir" diye elenmez. SFX ayrımı konuma/tarzaba göre yapılır, uzunluğa göre değil.
- [ ] 5.2 Kaynaşmış kelime onarımı (`OFMY`, `MASS- PRODUCE`): OCR sonrası sözlük-tabanlı split denetimi; onarılamayan REVIEW'a.
- [ ] 5.3 Çeviri çıktısında kaynak-dil artığı denetimi: TR çıktıda EN stop-word (`A`, `THE`) tek başına kaldıysa bloğu REVIEW'a at.
- **Kriter:** Hikâye balonu coverage %100 (SKIP'ler yalnız SFX/watermark/logo olacak, gerekçeli).

---

## FAZ 6 — MODEL SHOOTOUT VE YÜKSELTME (12GB sınırı içinde)

**Bağlam:** Hy-MT2-7B-Q8_0 (~8GB) şu an iyi iş çıkarıyor ama terim tutarlılığı ve deyimler zayıf. BenchLM Eyl 2026: open-weight multilingual'de Qwen ailesi lider; TranslateGemma adanmış çeviri ailesi. Staged tasarımda tavan ~9–10GB ağırlık.

| Aday | Boyut | Not |
|---|---|---|
| TranslateGemma-12B-Q5_K_M | ~8.4GB | diskte hazır, provider tamam |
| Qwen3-8B-Q8_0 | ~8.5GB | indirilecek |
| EuroLLM-9B-Q5_K_M | ~7GB | TR dahil, indirilecek |
| 27B+ / 12B-Q8 / çift server | >12GB | **yasak** |

- [ ] 6.1 Mevcut `translation_model_shootout_v1` + `hy_mt2_karga_real30` ile Hy-MT2 vs TranslateGemma-12B vs Qwen3.5-9B, dondurulmuş 30 gerçek balonda kör sıralama. İndirme yok.
- [ ] 6.2 Fark anlamlıysa Qwen3-8B-Q8_0 ve EuroLLM-9B-Q5 indir + provider klon deseniyle bağla (model başına: provider dosyası + registry alias + config; `chapter_analyzer`'a dokunulmaz).
- [ ] 6.3 Kazanan `hy_mt2_production_gate_v1` deseninden geçer → `config.yaml` default'u değişir. Kaybeden provider silinmez, registry'de alternatif kalır.
- **Kriter:** Kör sıralamada kazanan, mevcut default'tan istatistiksel olarak üstün VEYA mevcut kalır (karar veriye dayanır, hisse değil).

---

## FAZ 7 — QA/REGRESYON ALTYAPISI (tekrar kusur üretmeyi yasakla)

- [ ] 7.1 Her fazın regresyon testi `tests/` altında (logo hash, çift-render sayımı, terim tutarlılığı, halo bandı, coverage).
- [ ] 7.2 `audit_e2e_real_chapter1.py` genişletilecek: metrik setine `logo_pixel_diff`, `duplicate_render_count`, `term_consistency`, `untranslated_story_count` eklenecek; gate eşikleri aşılırsa script non-zero çıkar.
- [ ] 7.3 `review_output/` altına her run'da otomatik "kusur raporu" (şüpheli blok crop'ları + sayfa listesi) üretilecek — insan REVIEW turu kör olmaz.
- **Kriter:** `pytest -q` + audit gate yeşil olmadan merge yok.

---

## FAZ 8 — PERFORMANS (son, şart değil)

- Mevcut: ~254 sn/bölüm (25 sayfa). Kabul edilebilir, öncelik değil. Faz 1–7 bitmeden dokunulmaz.
- Sonra: çeviri chunk 32 → adaptif, LaMa batch 24 tavanı, OCR gated 398/602 oranı (gereksiz crop'lar?) incelenir.

---

## YÜRÜTME SIRASI

```
0 (hijyen) → 2 (render) → 1 (logo) → 3 (terim) → 4 (inpaint) → 5 (coverage)
      → 6 (shootout) → 7 (gate'leri kalıcılaştır) → 8 (perf)
```

Faz 2 ile 1 bağımsızdır, paralel yürütülebilir. Faz 3, Faz 2'nin bitmesini bekler (render değişikliği terim testini etkiler). Faz 6 her an koşabilir (üretim yolunu değiştirmez).

**İlk komut:** Faz 0.1 + 0.2 (bugün), ardından Faz 2 + Faz 1 paralel.

---

## İLERLEME GÜNLÜĞÜ (2026-09-12)

- [x] **Faz 0 tamamı:** secret hijyeni (`GEMINI_API_KEY` env'de, yaml null; `load_config` env-fallback + `update_gemini_api_key` sır yazmıyor), 10 atomic commit + push, output/ağırlık ignore doğrulandı. 17 dosyaya LSP taraması: değişiklerden yeni diagnostic yok.
- [x] **Faz 2 tamamı:** renderer ön-geçiş (blok-id dedup, çakışma grubunda en-uzun-metin kazanır + warning, kelimesiz metin atlanır, sahipsiz çift-tırnak temizlenir — ASCII kesme işareti korunur) + eligibility parça-veto (`THE/A/AN/I`, noktalama-only). 19 yeni/güncel test yeşil (`test_renderer_guards.py`, `test_translation_eligibility.py`). Anti-overfit: sentetik fixture, göreli IoU eşiği (0.6), bölüm verisi yok.
- [x] **Faz 1 tamamı:** geometrik logo kuralı (`_is_logo_like_region`: sayfa-göreli üst-bölge/dev-glif/seyreklik, kalibre eşikler) DIALOGUE/NARRATION + UNKNOWN dallarına bağlandı; ezber kelime listesi (`ZERO/FANTASY/ONLINE/PROLOGUE`) silindi. 7 sentetik test + 55 mevcut test yeşil (regresyon yok). Gerçek veri doğrulaması: P002 logo parçaları 13/14 SKIP, 40 rastgele AUTO'dan 39 untouched; id 83 (`Aw`, dev art-vokalizasyon) da SKIP — logo imzası, false positive değil.
- [ ] Faz 3 (sıradaki): provider glossary tüketiyor (`inp.glossary` → sentinel), ama üretim `TranslationInput(items)` ile boş geçiyor. İş: bölüm-içi tekrar kilidi + `glossary.json` artefaktı.

## DOĞRULAMA TURU (2026-09-12, güncel kurallar, 266 sn)

- Metrik: 230/74/306 → **226 auto / 52 review / 332 skip** (0 fail, 0 overflow). Review yükü −22: çoğu art-kırıntı REVIEW'dan SKIP'e geçti.
- `logo_art_skip` 69 region: tamamı dev-seyrek art-crop (52×179'dan 489×466'ya), hiçbiri çevrilmedi (translation None) — hikâye kaybı yok (40 rastgele AUTO kontrolü: 39 untouched).
- P002 logo: **"Zero FANTASY" artı tamamen sağlam**, üstünde Türkçe kırıntı/leke yok. Çift-render (`DÜNYANIN`×2/`THE`) yok, tek temiz basım. `-ONLINE-` → `ÇEVRİMİÇİ` başlık altına lokalize basıldı (art bozulmadan — kabul).
- P003/P024 kalan izler YENİDEN TEŞHİS: render'daki `I` ve `iz"bırakmak` izleri çeviri değil, **eksik inpaint temizliği** (bölge çevirilerinde o metinler yok). Yani Faz 4'ün kapsamı netleşti: maske kapsamı + inpaint-sonrası glif kalıntı denetimi (temizlenen kutuda OCR harf bulursa REVIEW).
- P024: watermark rozeti (`ASMOTOON.COM`) sağlam; `Bu sefer/Bunu yapıyorum` kısmi çakışması sürüyor (IoU 0.6 eşiğinin altı — eşik/apartman değil, grouping işi).

## FAZ 3 DOĞRULAMA TURU (2026-09-12)

- Mekanizma: `TranslationInput.glossary` → sentinel koruması + Türkçe morfoloji-restore zaten vardı; üretim boş geçiyordu. Şimdi bölüm-terim kilidi besliyor (`chapter_glossary.py` + `glossary.json`).
- Kilit disiplini (anti-overfit): TAM-BÜYÜK + morfolojik filtre (çoğul-S/-ING/-ED/kısaltma yok) + bölüm-sözlüğü (küçük hali geçen sıradan sözcük yok) + NEVER_LOCK çekirdeği + hedef-validasyonu. Alıntılı/özel adlar kilitsiz `observed_terms`.
- Kritik ders: bağımsız çözüm yanlış anlamı kilitleyebilir (`GUILD`→spor `LİG`, oysa bağlam `LONCA`). Çare: **tutarlılık kapanışı** — kilit, en kısa ≤2 geçiş cümlesinde çekimli yüzey olarak geçmiyorsa REDDEDİLİR (modelin kendi bağlamı bekçi). Ayrıca `translate_batch` içinde gizli `TranslationItem` import hatası bulundu ve düzeltildi (o yol üretimde hiç çağrılmamıştı).
- Mini-kanıt (5 sayfa, 97 sn): `ASMOTOON` echo-kilidi (kredi korunur); `CRAFTER` kilidi REDDEDİLDİ çünkü bağımsız `ÜRETİCİ` cümlelerde yoktu — cümleler zaten tutarlı `USTA` idi. Kapanış regresyonu engelledi, mekanizma doğru çalıştı.
- Altyapı: `scripts/fast_verify_mini.py` (5 sayfa ~97 sn vs tam audit ~261 sn) + arka-plan koşu deseni (`Start-Process` + log + poll) — uzun komutlar artık engellemiyor.
