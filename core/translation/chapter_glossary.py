"""Bölüm-içi terim tutarlılık kilidi (Faz 3).

Aynı bölümde tekrar eden ayırt edici terimler (oyun terimleri, unvanlar,
alıntılanan isimler) her blokta farklı çevrilirse (`CRAFTER` → USTA /
zanaatkar / ÜRETİCİ) bölüm tutarlılığı çöker. Bu modül terimleri KAYNAK
tekrarından bulur (çeviriye bakmaz — hizalama sorunu yok), her benzersiz
terimi bir kez çevirip kilitler; kilit `TranslationInput.glossary` üzerinden
provider sentinel korumasına girer.

Genel ve ezbersiz: hiçbir seri/bölüm/terim adı gömülü değildir. Eşikler
görelidir (tekrar sayısı). Maliyet sınırlıdır (benzersiz terim başına değil,
tek toplu çağrı).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from loguru import logger


# Genel İngilizce işlev sözcükleri (dilbilgisel, terim olamaz).
EN_STOPWORDS = frozenset({
    "THE", "A", "AN", "AND", "OR", "BUT", "IF", "THEN", "ELSE", "WHEN",
    "WHERE", "WHAT", "WHICH", "WHO", "HOW", "WHY", "THIS", "THAT",
    "THESE", "THOSE", "WITH", "WITHOUT", "FROM", "INTO", "OVER",
    "UNDER", "ABOUT", "THROUGH", "DURING", "BEFORE", "AFTER", "BETWEEN",
    "YOU", "YOUR", "YOURS", "HE", "SHE", "THEY", "THEM", "THEIR",
    "WE", "OUR", "OURS", "US", "ME", "MY", "MINE", "HIM", "HER",
    "HIS", "ITS", "IT", "THIS", "WILL", "WOULD", "CAN", "COULD",
    "SHOULD", "MUST", "HAVE", "HAS", "HAD", "DO", "DOES", "DID",
    "ARE", "WAS", "WERE", "BEEN", "BEING", "NOT", "NO", "YES",
    "VERY", "JUST", "LIKE", "THAN", "TOO", "SUCH", "ONLY", "OWN",
    "SAME", "OTHER", "MORE", "MOST", "MANY", "MUCH", "SOME", "ANY",
    "ALL", "EVERY", "EACH", "FEW", "BOTH", "EITHER", "NEITHER",
    "ONE", "TWO", "FIRST", "LAST", "NEW", "OLD", "BIG", "SMALL",
    "GOOD", "GREAT", "LITTLE", "OWN", "SURE",
})

# Aday terim kalıpları: TAM-BÜYÜK (CRAFTER), "alıntılı ifade", Title-Case uzun.
_ALL_CAPS_RE = re.compile(r"[A-Z][A-Z0-9&'\-]{3,}")
_QUOTED_RE = re.compile(r'"([^"]{2,60})"')
_TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]{4,}(?:\s+[A-Z][a-z]{4,}){0,2}\b")


def _is_lockable_caps(token: str) -> bool:
    """TAM-BÜYÜK token kilitlenebilir mi? (hassasiyet öncelikli).

    Çekimlenen/anlamı bağlama göre kayan biçimler kilitlenmez —
    sentinel-restore yalın hal basar ve Türkçe ekleri/nuansı bozar
    (`WEAPONS`→`Silahlar` + iyelik = `Benim Silahlar`,
    `HERE`→`Burada` + yönelme = `Burada gel`).
    Genel morfolojik + sözlüksel kurallar, kelime ezberi yok:
    -ING (eylem), -ED (geçmiş/sıfat), çoğul -S (SS ile bitmeyenler),
    kısaltmalar, küçük hali bölümde geçenler, NEVER_LOCK çekirdeği.
    """
    if len(token) < 4 or token in EN_STOPWORDS or token in NEVER_LOCK:
        return False
    if token.isdigit():
        return False
    if len(token) >= 5 and token.endswith("ING"):
        return False
    if len(token) >= 4 and token.endswith("ED"):
        return False
    if len(token) > 4 and token.endswith("S") and not token.endswith("SS"):
        return False
    return True


# Asla kilitlenmeyen yaygın sözcük çekirdeği (genel İngilizce; çekimlenir
# veya bağlama göre anlam kaydırır — hiçbir seriye özel değildir).
NEVER_LOCK = frozenset({
    "HERE", "THERE", "NOW", "TODAY", "TONIGHT", "ONCE", "TWICE",
    "NEVER", "EVER", "ALWAYS", "STILL", "ALREADY", "YET",
    "BACK", "AWAY", "AROUND", "TOGETHER", "APART", "ASIDE", "AHEAD",
    "ALONG", "ACROSS", "FORWARD",
    "LONG", "SHORT", "HIGH", "LOW", "FAST", "SLOW", "HARD", "SOFT",
    "EARLY", "LATE", "SOON", "FAR", "NEAR", "CLOSE", "WELL", "QUICK",
    "REAL",
    "STRONG", "WEAK", "DARK", "BRIGHT", "CLEAR", "CLEAN", "FRESH",
    "BEST", "WORST", "BETTER",
    "ANYTHING", "SOMETHING", "NOTHING", "EVERYTHING",
    "ANYONE", "SOMEONE", "EVERYONE", "NOBODY",
})

MIN_OCCURRENCES_DEFAULT = 3
MAX_LOCKED_TERMS_DEFAULT = 30


@dataclass
class ChapterTerm:
    """Tekrar eden aday terim."""

    term: str
    count: int
    block_ids: list[int] = field(default_factory=list)


class TermTranslator(Protocol):
    """Çözümleme için gereken en küçük arayüz (gerçek provider veya stub)."""

    def translate_batch(self, texts: list[str]) -> list[str]: ...


def extract_repeated_terms(
    texts: Sequence[str],
    block_ids: Sequence[int] | None = None,
    min_occurrences: int = MIN_OCCURRENCES_DEFAULT,
) -> list[ChapterTerm]:
    """Kilitlenebilir tekrar terimlerini bulur (yalnız TAM-BÜYÜK, çekimsiz).

    Çeviriye bakılmaz; yalnızca kaynak tekrarı sayılır. Döndürülenler
    sıklığa göre azalan sıralıdır. Alıntılı ifadeler/özel adlar
    `extract_observed_terms` ile ayrıca raporlanır (otomatik kilit yok).
    """
    ids = list(block_ids) if block_ids is not None else [0] * len(texts)
    # Bölüm sözlüğü: küçük harfle de geçen kelime sıradan sözcüktür, terim
    # değil ("HERE" haykırılsa da "here" geçiyorsa kilitlenmez). Terimler
    # yalnızca TAM-BÜYÜK yazılan sözcüklerdir (CRAFTER, GOBLIN).
    lower_vocab: set[str] = set()
    for text in texts:
        if not text:
            continue
        for w in re.findall(r"[A-Za-z]{4,}", text):
            if not w.isupper():
                lower_vocab.add(w.lower())
    hits: dict[str, ChapterTerm] = {}
    for text, bid in zip(texts, ids):
        if not text:
            continue
        # Alıntılı ifadeler birim olarak ele alınır (observed); içindeki
        # kelimeler tek tek kilitlenmez (yarım kilit riski).
        bare = _QUOTED_RE.sub(" ", text)
        candidates: set[str] = set()
        for m in _ALL_CAPS_RE.findall(bare):
            token = m.strip("'").strip("-")
            if "'" in token:
                continue  # kısaltmalar çekimlenir (YOU'RE, DON'T)
            if token.lower() in lower_vocab:
                continue  # küçük hali de var → sıradan sözcük
            if _is_lockable_caps(token):
                candidates.add(token)
        for cand in candidates:
            key = cand.upper()
            entry = hits.get(key)
            if entry is None:
                entry = hits[key] = ChapterTerm(term=cand, count=0)
            entry.count += 1
            if bid not in entry.block_ids:
                entry.block_ids.append(bid)
    repeated = [t for t in hits.values() if t.count >= min_occurrences]
    repeated.sort(key=lambda t: (-t.count, t.term))
    return repeated


def extract_observed_terms(
    texts: Sequence[str],
    block_ids: Sequence[int] | None = None,
    min_occurrences: int = MIN_OCCURRENCES_DEFAULT,
) -> list[ChapterTerm]:
    """KilitlenMEYEN gözlemler: alıntılı ifadeler + Title-Case özel adlar.

    Bunlar `glossary.json` içinde `observed_terms` olarak raporlanır
    (insan REVIEW + gelecek profil için); otomatik sentinel kilidi YOK
    (ek-fiil riski: `"Silent Chain"i` gibi).
    """
    ids = list(block_ids) if block_ids is not None else [0] * len(texts)
    hits: dict[str, ChapterTerm] = {}
    for text, bid in zip(texts, ids):
        if not text:
            continue
        candidates: set[str] = set()
        for m in _QUOTED_RE.findall(text):
            phrase = " ".join(m.split())
            if len(phrase) >= 2 and phrase.upper() not in EN_STOPWORDS:
                candidates.add(phrase)
        for m in _TITLE_CASE_RE.findall(" ".join(text.split())):
            words = m.split()
            if all(w.upper() not in EN_STOPWORDS for w in words):
                candidates.add(m)
        for cand in candidates:
            key = cand.upper()
            entry = hits.get(key)
            if entry is None:
                entry = hits[key] = ChapterTerm(term=cand, count=0)
            entry.count += 1
            if bid not in entry.block_ids:
                entry.block_ids.append(bid)
    repeated = [t for t in hits.values() if t.count >= min_occurrences]
    repeated.sort(key=lambda t: (-t.count, t.term))
    return repeated


def _is_usable_target(target: str) -> bool:
    """Bağımsız terim çevirisi kilitlenebilir mi?

    Tekil terim bağlamsız çevrildiği için model bazen budar (`LONCA`→`L`),
    açıklar (cümle döndürür) veya boş bırakır. Bozuk kilit üretimden
    beterdir — şüpheli hedef reddedilir (o terim kilitsiz kalır).
    """
    t = (target or "").strip()
    if len(t) < 2:
        return False
    if not re.search(r"\w", t, re.UNICODE):
        return False
    if len(t.split()) > 4:
        return False
    return True


_TR_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})
_TR_UPPER_MAP = {"i": "İ", "ı": "I"}


def _tr_titlecase(text: str) -> str:
    """Türkçe-duyarlı cümle-kası (Unicode `capitalize` İ/ı bozar)."""
    low = text.translate(_TR_LOWER_MAP).lower()
    if not low:
        return low
    first = _TR_UPPER_MAP.get(low[0], low[0].upper())
    return first + low[1:]


def normalize_lock_target(source: str, target: str) -> str:
    """Kilit hedefi kasa normalizasyonu (Faz 3 cila).

    Model haykırılan kaynağı haykırarak çevirir (`WORLD`→`DÜNYA`); sentinel
    tabanı büyük harfle kilitlenirse çekimler de öyle basılır (`DÜNYAda`).
    Yankı-hedefler (kaynakla aynı) aynen korunur (`ASMOTOON`); gerçekten
    çevrilenler cümle-kasaya indirilir (`DÜNYA`→`Dünya`) — morfoloji motoru
    zaten küçük tabandan doğru çeker (`Dünyada`).
    """
    t = (target or "").strip()
    if t.casefold() == (source or "").strip().casefold():
        return t
    return _tr_titlecase(t)


def _store_lock(
    mapping: dict[str, str],
    methods: dict[str, str],
    term: str,
    target: str,
    method: str,
) -> str:
    """Kilit yazan TEK nokta (İŞ 2): normalize garantili.

    Gelecek kilit yolları (vote/harvest/standalone) buradan geçmek zorunda;
    `mapping[x] = ham_hedef` doğrudan yazımı YASAK (DÜNYA→DÜNYAda artığı).
    """
    norm = normalize_lock_target(term, target)
    mapping[term] = norm
    methods[term.upper()] = method
    return norm


# P2: kilit yazım-denetime kapısı (WEAPON→"Sılah" vakası). hunspell-tr
# sözlüğü vendorda (assets/hunspell); spylls saf-python'dur. Kural hassasiyet
# önceliklidir: yalnız mesafe-1 + GEÇERLİ öneri = yazım yanlışı (red);
# bilinmeyen sözcük (Goblin) ve yankı (BOSS) KABUL edilir. Sözlük yoksa
# kapı açık-geçer (fail-open — üretim asla kırılmaz).
_SPELL_DIR = Path(__file__).resolve().parents[2] / "assets" / "hunspell"
_spell_dict = None
_spell_unavailable_logged = False


def _spell_dictionary():
    """hunspell-tr singleton (yoksa None)."""
    global _spell_dict, _spell_unavailable_logged
    if _spell_dict is not None:
        return _spell_dict
    try:
        from spylls.hunspell import Dictionary

        aff = _SPELL_DIR / "tr_TR.aff"
        dic = _SPELL_DIR / "tr_TR.dic"
        if not (aff.is_file() and dic.is_file()):
            raise FileNotFoundError(f"sözlük dosyası yok: {_SPELL_DIR}")
        _spell_dict = Dictionary.from_files(str(_SPELL_DIR / "tr_TR"))
    except Exception as exc:
        if not _spell_unavailable_logged:
            logger.warning(f"Yazım kapısı devre-dışı (sözlük yüklenemedi): {exc}")
            _spell_unavailable_logged = True
        _spell_dict = False
    return _spell_dict or None


def _edit_distance(a: str, b: str) -> int:
    """Kısa dizgiler için Levenshtein (yanlış-öneri eşiği)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return 2
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = i
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            row_min = min(row_min, cur[j])
        if row_min > 1:
            return 2
        prev = cur
    return prev[lb]


def _is_typo_lock_target(source: str, target: str) -> bool:
    """Kilit hedefi bariz yazım yanlışı mı? (Sılah→red, Goblin→kabul)."""
    t = (target or "").strip()
    if not t or t.casefold() == (source or "").strip().casefold():
        return False  # yankı muaf
    spell = _spell_dictionary()
    if spell is None:
        return False  # fail-open
    for token in re.findall(r"[A-Za-zÇçĞğİıÖöŞşÜü]+", t.lower()):
        if len(token) < 2 or token.isdigit():
            continue
        try:
            if spell.lookup(token):
                continue
            for sug in list(spell.suggest(token))[:3]:
                cand = str(sug).lower()
                if spell.lookup(cand) and _edit_distance(cand, token) == 1:
                    return True
        except Exception:
            continue
    return False


def _accept_lock_target(
    term: str,
    target: str,
    typo_rejected: dict[str, str] | None = None,
) -> str | None:
    """Normalize + yazım kapısı: kabulde normalize hedef, redde None."""
    norm = normalize_lock_target(term, target)
    if _is_typo_lock_target(term, norm):
        logger.warning(f"Terim kilidi reddedildi ({term}): yazım yanlışı {norm!r}")
        if typo_rejected is not None:
            typo_rejected[term] = norm
        return None
    return norm


def _surface_hits_in_texts(target: str, trs: Sequence[str]) -> list[str | None]:
    """Her TR için bağımsız hedefin eşleşen yüzeyini (veya None) döndürür."""
    from core.translation.protection import ProtectedTermMeta, _target_surface_forms

    meta = ProtectedTermMeta(
        sentinel="", source_original="", target_base=target,
        is_approved=True, proper_name=False,
    )
    surfaces = _target_surface_forms(meta)
    return [
        next(
            (
                s
                for s in surfaces
                if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", tr, re.IGNORECASE)
            ),
            None,
        )
        for tr in (trs or [])
    ]


def resolve_chapter_glossary(
    translator: TermTranslator,
    terms: Sequence[ChapterTerm],
    max_terms: int = MAX_LOCKED_TERMS_DEFAULT,
    texts: Sequence[str] | None = None,
    block_ids: Sequence[int] | None = None,
    typo_rejected: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Benzersiz terimleri çözüp tutarlılık kapanışıyla kilitler.

    1. Adım (ucuz): her terim tek başına çevrilir.
    2. Adım (doğrulama): `texts` verildiyse her adayın en kısa ≤2
       geçiş cümlesi çevrilir; bağımsız hedef bu cümlelerde çekimli
       haliyle geçmiyorsa anlam uyuşmazlığı vardır ve kilit REDDEDİLİR
       (`GUILD`→`LİG` cümlelerde `LONCA` ise kilit yok = eski davranış).
       Reddedilenler döndürülür — ana-geçiş sonrası hasat (`harvest_…`)
       için girdi olurlar.
    Modelin kendi bağlamsal tutarlılığı kapı bekçisidir; harici bilgi yok.

    Dönüş: (mapping, methods, rejected). methods kilit yöntemini verir
    (`standalone`) — `glossary.json` denetimi için.
    """
    from core.translation.protection import ProtectedTermMeta, _target_surface_forms

    chosen = list(terms)[:max(0, max_terms)]
    if not chosen:
        return {}, {}, {}
    try:
        rendered = translator.translate_batch([t.term for t in chosen])
    except Exception as exc:
        logger.warning(f"Terim kilidi çözümlemesi başarısız, glossary boş: {exc}")
        return {}, {}, {}
    standalone: dict[str, str] = {}
    for term, target in zip(chosen, rendered):
        target = (target or "").strip()
        if _is_usable_target(target):
            accepted = _accept_lock_target(term.term, target, typo_rejected)
            if accepted is not None:
                standalone[term.term] = accepted
        else:
            logger.warning(
                f"Terim kilidi reddedildi ({term.term}): bozuk hedef {target!r}"
            )
    if texts is None or block_ids is None:
        logger.info(f"Terim kilidi: {len(standalone)}/{len(chosen)} (doğrulamasız).")
        return standalone, {s.upper(): "standalone" for s in standalone}, {}

    # Tutarlılık kapanışı: en kısa ≤2 geçiş cümlesini çevir, yüzey ara.
    samples: list[str] = []
    sample_owner: list[str] = []
    by_id = dict(zip(block_ids, texts))
    for term in chosen:
        if term.term not in standalone:
            continue
        own = sorted(
            ((len(by_id.get(b, "")), b) for b in term.block_ids if by_id.get(b)),
            key=lambda p: p[0],
        )
        for _, bid in own[:2]:
            samples.append(by_id[bid])
            sample_owner.append(term.term)
    sample_tr: dict[str, list[str]] = {}
    if samples:
        try:
            rendered_samples = translator.translate_batch(samples)
        except Exception as exc:
            logger.warning(f"Terim doğrulama çevirisi başarısız: {exc}")
            rendered_samples = []
        for owner, tr in zip(sample_owner, rendered_samples):
            sample_tr.setdefault(owner, []).append(tr or "")

    mapping: dict[str, str] = {}
    methods: dict[str, str] = {}
    rejected: dict[str, str] = {}
    for term, target in standalone.items():
        checks = sample_tr.get(term, [])
        if not checks:
            _store_lock(mapping, methods, term, target, "standalone")
            continue
        hits = _surface_hits_in_texts(target, checks)
        if all(h is not None for h in hits):
            _store_lock(mapping, methods, term, target, "standalone")
        else:
            rejected[term] = target
            logger.warning(
                f"Terim kilidi reddedildi ({term}): bağımsız hedef {target!r} "
                f"geçiş cümlelerinde yok (anlam uyuşmazlığı). "
                f"örnekler={[ (h, (tr or '')[:40]) for h, tr in zip(hits, checks) ]}"
            )
    logger.info(f"Terim kilidi: {len(mapping)}/{len(chosen)} terim kilitlendi.")
    return mapping, methods, rejected


def glossary_entries(mapping: dict[str, str]) -> list[str]:
    """Provider `inp.glossary` formatı: `KAYNAK->HEDEF`."""
    return [f"{src}->{tgt}" for src, tgt in mapping.items() if src and tgt]


HARVEST_MIN_HITS_DEFAULT = 2
HARVEST_MIN_RATIO_DEFAULT = 0.5


def _family_norm(span: str) -> str:
    """Yüzey karşılaştırma anahtarı: küçük harf, alfanümerik."""
    return re.sub(r"[^\w]", "", span.casefold())


def harvest_confirmed_locks(
    rejected: dict[str, str],
    terms: Sequence[ChapterTerm],
    block_translations: dict[int, str],
    min_hits: int = 2,
    min_ratio: float = 0.5,
    typo_rejected: dict[str, str] | None = None,
) -> dict[str, str]:
    """Reddedilen terimler için ana-geçiş çevirilerinde yüzey oylaması.

    Maskeli yeniden-çeviri YOK (parti-bağlam deterministik değil — aynı
    cümle farklı partide farklı çevriliyor). Bunun yerine 1. tur TR'lerde
    bağımsız hedefin çekimli yüzeyleri aranır: model zaten tutarlıysa kilit,
    değilse kilit yok. Ek LLM maliyeti sıfır, deterministik.
    """
    from core.translation.protection import ProtectedTermMeta, _target_surface_forms

    by_id = {t.term: t for t in terms}
    winners: dict[str, str] = {}
    for term, target in rejected.items():
        info = by_id.get(term)
        if info is None:
            continue
        meta = ProtectedTermMeta(
            sentinel="", source_original=term, target_base=target,
            is_approved=True, proper_name=False,
        )
        surfaces = {_family_norm(s) for s in _target_surface_forms(meta)}
        trs = [
            block_translations[bid]
            for bid in info.block_ids
            if block_translations.get(bid)
        ]
        if not trs:
            continue
        hits = sum(
            1
            for tr in trs
            if any(
                re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", tr, re.IGNORECASE)
                for s in surfaces
            )
        )
        if hits >= min_hits and hits / len(trs) >= min_ratio:
            accepted = _accept_lock_target(term, target, typo_rejected)
            if accepted is None:
                continue
            winners[term] = accepted
            logger.info(f"Hasat kilidi: {term} -> {winners[term]} ({hits}/{len(trs)})")
        else:
            logger.info(f"Hasat kilitsiz bıraktı: {term} ({hits}/{len(trs)})")
    return winners


def write_glossary_json(
    path: str | Path,
    mapping: dict[str, str],
    terms: Sequence[ChapterTerm],
    observed: Sequence[ChapterTerm] = (),
    methods: dict[str, str] | None = None,
    rejected_targets: dict[str, str] | None = None,
) -> Path:
    """`analysis/glossary.json` artefaktı: kilit + kanıt (blok id'leri)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {t.term.upper(): t for t in terms}
    method_of = methods or {}
    payload = {
        "locked_terms": [
            {
                "source": src,
                "target": tgt,
                "occurrences": counts.get(src.upper(), ChapterTerm(src, 0)).count,
                "block_ids": counts.get(src.upper(), ChapterTerm(src, 0)).block_ids,
                "method": method_of.get(src.upper(), "standalone"),
            }
            for src, tgt in mapping.items()
        ],
        "observed_terms": [
            {
                "source": t.term,
                "occurrences": t.count,
                "block_ids": t.block_ids,
            }
            for t in observed
        ],
        "rejected_targets": [
            {"source": src, "target": tgt, "reason": "typo"}
            for src, tgt in (rejected_targets or {}).items()
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
