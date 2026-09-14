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

## FAZ 3-DEVAM: ÇOĞUNLUK-OYU KİLİDİ (2026-09-12)

- Sorun: tutarlılık kapanışı `CRAFTER`ı kilitleyemedi (bağımsız `ÜRETİCİ` cümlelerde yoktu) ama cümleler kendi arasında tutarlıydı (`USTA` ailesi) — kapanış fazla katıydı.
- Çözüm: masked-diff oylaması — geçiş cümleleri maskeli/maskesiz çevrilir, fark aralıkları gövde-ailesine kümelenir (≥5 harf ortak önek), ≥%50 + ≥2 uzlaşıda en kısa üye kilitlenir. Maliyet tavanlı (terim başına ≤8 geçiş, ≤5 terim).
- Reddedilen tasarım (kanıtlı): anlam-ipucu promptu Hy-MT2'de çalışmadı (talimatı çevirmeye kalktı) — silindi.

## FAZ 4 (2026-09-12, mini-kanıtlı)

- Kapsam: göreli LaMa çekirdeği (3× sabit 7x7 → maske-boyu işlevi), maske-içi hayalet denetimi (LaMa/ortanca yolu), dolgu/zemin uyuşmazlığı (balonsuz beyaz-leke → REVIEW). Üçü de sentetik testli (8 test), LSP kapısı temiz.
- Mini-kanıt: denetimler 1 blokta REVIEW üretti (sel yok); logo/balonlar temiz.
- Ek bulgu (aynı tur): P009 çift-basma guard'ı atlatmış — kök neden kopya-tespit eşiği (IoU 0.468 < 0.5) + render eşiği (0.6). Çözüm: IoU 0.4 + kaynak-altküme kuralı (IoU>0.2). P009 görsel doğrulandı: tek temiz basım.
- Bilinen küçük izler: yalnız `!` artığı, `var..!` çift noktalama (çevirmen tarafı, Faz 5/6 backlogu).

## FAZ 3-HASAT (2026-09-12, mini-kanıtlı)

- Kök neden bulundu: parti-bağlam deterministik değil — aynı cümle farklı partide farklı çevriliyor. Maskeli yeniden-çeviri o yüzden çöp üretti. Çare: 1. tur TR'lerde yüzey oylaması (ek parti yok) + kilitlenen terim geçen bloklara 2. tur.
- Mini-kanıt: `CRAFTER`→`USTA` (hasat) — 4 blokta tek aile (`USTA/USTA/USTA/USTAlar`); `WORLD`→`DÜNYA`, `LEVEL`→`SEVİYE` hasat kilitli. Yankı-ailesi dondurma yasağı + `kara/karar` ayrımı testli.
- Bilinen küçük izler (Faz 3 cila backlogu): taban-büyük-harf artığı (`DÜNYAda`/`DÜNYAinde` — ek doğru, kasa ham), nadir çift-çoğul (`USTAlar"’ler` — model sentinel sonrasına ek yapıştırmış).

## KALAN-İŞLER TURU (2026-09-12, İŞ 1–5)

- [x] **İŞ 1 (DAMMIT → Faz 4-kısıt, DURDURULDU):** hipotez 1 doğrulandı (glif-bariyer hapsi) ve 7px renksiz köprü denendi — AMA tam-audit kanıtı reddetti: köprü beyazı %97 doldurur (`block_0035` forenziği) ama uzak glif pikselleri maskeye giremez (glifler `bg_like` dışı; kapsama yalnız refined+7px bant). Sonuç: inpaint-REVIEW 0→10 regresyonu (9'u second-chance). KALAN stop-kuralı uygulandı: köprü GERİ ALINDI (3x3 tohum), yerine kısıt kilitlendi. **FAZ 4-KISIT:** kısmi-kutu + sivri-uçlu balon glifleri maske katmanının ötesindedir (detector-recall işi). Kilit testi: `test_expand_mask_stays_in_seed_compartment` (taşma yok, sessiz leke yok).
- Taze tam-audit (İŞ 1+2'li kod, 298 sn): blok 265, çevrilen 212, inpaint 202 ok / 10 review, render 199, overflow 0; bölge 203 auto / 62 review / 324 skip. DAMMIT çevirisi hazır (`Kahretsin!`) ama temizlik REVIEW — kısıt teyidi.
- [x] **İŞ 2 (kasa cilası):** `_store_lock` — mapping'e yazan TEK nokta, normalize garantili (`chapter_glossary.py`); hasat yolu + `chapter_analyzer` hasat-`update` de normalize'den geçiyor. 3 yeni test (store/harvest/resolve `DÜNYA`→`Dünya`); eski `USTA` beklentisi `Usta` olarak düzeltildi. 27/27 yeşil.
- [x] **İŞ 3 (coverage kilidi):** kod değişikliği YOK. Bekçi testi (`test_short_english_exclamations_stay_eligible`: DAMMIT/BUT/HEY/NO-NOT-NOW) + audit metriği `short_dialogue_untranslated_count` — taze kanıtla rafine edildi: yalnız REVIEW-statü (SKIP tasarım-gereği), OCR-çöp hariç (`word_difference`/`ambiguous_unknown_review`/`primary_*`). Ölçülen: **6** (tamamı inpaint-sınır-artığı review; SKIP gürültü/logo sayılmaz). Eşik: ≤6 (artış yasak).
- [x] **İŞ 4 (shootout 4.1+4.2+4.3a):** kör set donduruldu — `benchmark/translation_ab_v1.json` (30 balon: 8 short + 8 longest + term-yoğun; bölüm gerçeği: maks 108 karakter, çok-üyeli çevrilmiş blok yok). İki kol koştu: Hy-MT2 (8.9 sn) ve TranslateGemma-12B (31.2 sn), ikisi de 30/30 boş-sıfır. Otomatik skor: echo 1-0, TR-karakter 21-21, uzunluk-oran 0.99-1.13 (fark ANLAMLI DEĞİL). Pusula hazır: `benchmark/results/translation_ab_v1/ballot_20260912-1707.md` (A/B karışık, anahtar ayrı dosyada). **AÇIK (insan):** 30 çift kör oy + karar kapısı (4.3b/4.4) — pusulayı doldurunca `config.yaml` kararı verilecek.
- Taze tam-audit (İŞ 1+2 kanıtı: DAMMIT görseli + `glossary.json` kasa) arka-planda koşuyor (`logs/audit_fresh.log`, ~5 dk).

## P1+P2 TURU (2026-09-12, build)

- [x] **P1-A (batch kayma sertleştirme):** `hy_mt2_gguf_translation.py` parse artık tekrarlı/uzaylı numara farkında (`dup_numbers` + `stray_numbers` loglanır/atılır); şüpheli item (tekrarlı numara, TR/EN>2.5) tekil retry'a düşer. 2 yeni test.
- [x] **P1-B (çapraz-numaralama doğrulaması):** kırılgan item'lar (kaynak ≤25 karakter, karmaşık chunk'ta ≤8 adet) izole mini-batch ile yeniden sorulur; model deterministiktir (temp 0.0/top_k 1/seed 0) — uyuşmazlık = kanıtlı karışma → tekil tiebreak + `numbering_inconsistent` + requires_review. 2 yeni test.
- [x] **P1-B analyzer bağlama:** yalnız ÖLÜMCÜL kod (`numbering_inconsistent`) bloğu failed sayar + `translation_guard_review` REVIEW'u verir + `translation_guard_blocks_count` metriği (regions/summary). Yumuşak bayraklar (echo/prose) eski davranışı korur — meşru yankılar (`Lv.998`, `BOSS`) basılmaya devam eder. Uçtan-uca sentetik test yeşil.
- [x] **P2 (kilit yazım kapısı):** `spylls` (pip) + `hunspell-tr` vendorda (`assets/hunspell/`, MPL-2.0). Kural: mesafe-1 + GEÇERLİ öneri = typo (red); bilinmeyen (Goblin) + yankı (BOSS) kabul; sözlük yoksafail-open. `resolve`/`harvest`/`glossary.json` (`rejected_targets`) hattına bağlı. 3 yeni test. Kanıt: `Sılah` red, `Silah/Dünyada/Usta/Goblin` kabul.
- Testler: 680 geçti (8 yeni), 2 önceden-var hata aynı. Taze Ch1 audit'i arka-planda (`logs/audit_p1p2_ch1.log`); ardından Ch2 genelleme audit'i.

## STUDIO POST-MORTEM (2026-09-13, KRİTİK BULGU + FİX)

- **Olay:** P1/P2'li Ch1 audit'inde `regions.json`'daki 211 çevirinin TAMAMI `STUDIO` yazıldı; render'lar doğruydu (P002 görsel + PIL kırpıntı kanıtlı).
- **Kök neden:** `chapter_analyzer.py` bölge-döngüsünde `tr_text = out_map[b.id]` — sızmış döngü değişkeni (`b` = `translation_eligible_blocks`'un SON bloğu). Bölgeler hep son bloğun çevirisini aldı. Son uygun blok, yankı-çevirili `STUDIO` logolu kredi bloğuydu (r571 `STUDIO`, AUTO). Forenzik: `git show HEAD` doğrusu `out_map[b_id]` — gerileme HEAD-sonrası çalışma ağacına girmiş, 211/211 sabit-değer imzası + `b.id`/`b_id` diff'i ile kanıtlandı.
- **Neden yakalanmadı:** mevcut e2e testleri tek blokluydu (sızan `b` tesadüfen doğru), `regions.json` çeviri alanı hiçbir testte blok-bazında doğrulanmıyordu; görsel QA render'a baktı (doğruydu).
- **Fix:** `out_map[b_id]` (tek satır) + mutasyon-kanıtlı bekçi `test_region_translations_match_own_block` (iki kutulu fixture, çeviriye blok-id gömülü; bug'lı kodda KIRMIZI, fix'li kodda YEŞİL doğrulandı). Ek ders: tespit-önbelleği stub adlarına duyarlı (`TwoBoxDetector` ayrı ad).
- Testler: 681 geçti, 2 önceden-var hata aynı. Doğrulama Ch1 + Ch2 audit'leri yeniden koşuyor.

## P1-B İKİNCİ AĞ (2026-09-13, 259/260 kör noktası)

- **Ölçüm:** Ch1'de 175/265 blok ≤25 karakter; her 32'li chunk'ta 7-28 kırılgan — mini-batch doğrulaması neredeyse hiç çalışmıyormuş (yalnız batch_0'da 1 kez, o da gerçek bir yakalama: b249). 259/260'ın chunk'ı eşiği aştığı için denetimsiz kalmış.
- **Fix:** `_verify_swap_pairs` — batch-kabul item'larda bitişik-tamamlayıcı oran (biri >1.5, diğeri <0.5) veya tekil hedge (<0.4, kaynak ≥10kr) → tekil izolasyon; uyuşmazlık `numbering_inconsistent` + REVIEW. Chunk başına ≤8 ek çağrı tavanı.
- Testler: 684 geçti (3 yeni: swap-uyuşmazlık/uyum/hedge), 2 önceden-var hata aynı. Kabul kanıtı için Ch1 audit'i yeniden koşuyor (259/260 balonları).
- [x] **İŞ 5 (gate'ler):** audit `--strict` kapısı (`short_untranslated≤6` [ölçülen], `overflow==0`; varsayılan uyarı, strict'te non-zero) + `scripts/write_defect_report.py` + test envanteri (F1–F5'in her birinin özel test dosyası var: logo/test_logo_protection, render/test_renderer_guards, terim/test_chapter_glossary, inpaint/test_inpaint_guards+test_text_mask_inpainting, coverage/test_translation_eligibility).
- Bilinen önceden-var kusurlar (bu turda dokunulmadı): `test_mixed_status_block_safety` (renderer kısmi-render vs eski tam-veto — spesifikasyon çelişkisi, karar bekliyor), `test_residual_expansion_can_follow_a_bounded_multi_pixel_glyph_edge` (review=True).

## F2 TURU (2026-09-13, build)

- [x] **Spike (ÖLDÜRÜLDÜ):** llama-server json_schema batch kilidi denendi — yapı geçerli (30/30 id) AMA içerik birleştirme + sessiz bir-kayma üretti. Yapı doğru numarayı yanlış metinle kilitler.
- [x] **Merge-pair net + ad-düşürme + S5 + S6** (detay commit mesajlarında). Testler: **722 geçti, 0 kırmızı.**
- [x] **Hızlı ölçü hattı** (tek yükleme ~10 sn) + **F2 terazisi:** overflow 4/4 sıfır, blok-sayımları sabit; REVIEW artışı bilinçli (dungeon +3, swordmaster +5, estate +2, prologue 0). `benchmark/golden_f2_v1.json` donduruldu.

## F3 TURU (2026-09-13, build)

- [x] **S4 bant-tarama:** B19 "M'" fotoğrafıyla kanıtlandı (maske-dışı kör nokta). İlk sürüm sel üretti (balon çizgisi/toz) → ölçülen ayıraçlarla daraltıldı (uzaklık≤8px + %50 aydınlık-komşu + alan 100..4000 + oran≤6). B19 + dungeon b21 ("IT'S" artığı) görsel kanıtlı yakalanıyor, temiz balonlar geçiyor.
- [x] **S2 yankı-koruma:** tek-kelimelik yankı blokları inpaint/render görmez, SKIP olur (orijinal piksel + özgün font korunur). Çok-kelimeliler basılmaya devam eder.
- [x] **S7 tire-birleştirme:** "DIFFER- ENT"→"DIFFERENT" hazırlıkta birleşir (kayıtlı kaynak korunur). Siyah-balon OCR'a dokunulmadı (kanıt yok; F2 guard'ları çöp çeviriyi REVIEW'a düşürür).
- [x] **F3 terazisi:** overflow 4/4 sıfır, blok-sayımları sabit; basılan düşüşü bilinçli (yankı-nötr + gerçek artık + yanlış-anlam tutmaları). `benchmark/golden_f3_v1.json` donduruldu.
- Testler: **732 geçti, 0 kırmızı.** Ara-bulgu: tam-süit, tanımsız `echo_skip_ids` UnboundLocalError yakaladı (çevrilecek-bloksuz hat) — düzeltildi.

## F4+F5 TURU (2026-09-13, build)

- [x] **F4-a kaynaşma-ölçümü → ATLANDI:** 670 blokta gerçek kaynaşma ~3 adet (ACHARACTER vb.), gerisi meşru uzun sözcük. Modeller bağlamdan çözüyor (OFMY kanıtlı). Sözlüksüz ayırıcı risk/faydayı karşılamaz.
- [x] **F4-b fren aleti:** `scripts/compare_golden_v1.py` — yankı-nötr çevrilme-oranı + kısa-hikaye uyarısı + overflow alarmı. İlk fren: temiz (overflow 0, oranlar 5 puan içinde; sword/estate kısa-UYARI takibi).
- [x] **F5 Gemini hakem deneyi → ÖLDÜRÜLDÜ:** 29 guard-REVIEW blokta Hy-tekil vs Gemini anlaşması 5/29; Gemini'de caps hastalığı + kırpma (b260 "BU SEFER,") + kişi-karıştırma (b249) var. Hakem çözümsüz belirsizliği çözmüyor, başka tahmin üretiyor. REVIEW+İngilizce-koruma zaten doğru cevap. Kanıt: `benchmark/results/hakem_gemini_v1.json`.
