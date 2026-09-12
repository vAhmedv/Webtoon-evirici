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


def resolve_chapter_glossary(
    translator: TermTranslator,
    terms: Sequence[ChapterTerm],
    max_terms: int = MAX_LOCKED_TERMS_DEFAULT,
    texts: Sequence[str] | None = None,
    block_ids: Sequence[int] | None = None,
) -> dict[str, str]:
    """Benzersiz terimleri çözüp tutarlılık kapanışıyla kilitler.

    1. Adım (ucuz): her terim tek başına çevrilir.
    2. Adım (doğrulama): `texts` verildiyse her adayın en kısa ≤2
       geçiş cümlesi çevrilir; bağımsız hedef bu cümlelerde çekimli
       haliyle geçmiyorsa anlam uyuşmazlığı vardır ve kilit REDDEDİLİR
       (`GUILD`→`LİG` cümlelerde `LONCA` ise kilit yok = eski davranış).
    Modelin kendi bağlamsal tutarlılığı kapı bekçisidir; harici bilgi yok.
    """
    from core.translation.protection import ProtectedTermMeta, _target_surface_forms

    chosen = list(terms)[:max(0, max_terms)]
    if not chosen:
        return {}
    try:
        rendered = translator.translate_batch([t.term for t in chosen])
    except Exception as exc:
        logger.warning(f"Terim kilidi çözümlemesi başarısız, glossary boş: {exc}")
        return {}
    standalone: dict[str, str] = {}
    for term, target in zip(chosen, rendered):
        target = (target or "").strip()
        if _is_usable_target(target):
            standalone[term.term] = target
        else:
            logger.warning(
                f"Terim kilidi reddedildi ({term.term}): bozuk hedef {target!r}"
            )
    if texts is None or block_ids is None:
        logger.info(f"Terim kilidi: {len(standalone)}/{len(chosen)} (doğrulamasız).")
        return standalone

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
    for term, target in standalone.items():
        checks = sample_tr.get(term, [])
        if not checks:
            mapping[term] = target
            continue
        meta = ProtectedTermMeta(
            sentinel="", source_original=term, target_base=target,
            is_approved=True, proper_name=False,
        )
        surfaces = _target_surface_forms(meta)
        ok = [
            any(
                re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", tr, re.IGNORECASE)
                for s in surfaces
            )
            for tr in checks
        ]
        if all(ok):
            mapping[term] = target
        else:
            logger.warning(
                f"Terim kilidi reddedildi ({term}): bağımsız hedef {target!r} "
                f"geçiş cümlelerinde yok (anlam uyuşmazlığı)."
            )
    logger.info(f"Terim kilidi: {len(mapping)}/{len(chosen)} terim kilitlendi.")
    return mapping


def glossary_entries(mapping: dict[str, str]) -> list[str]:
    """Provider `inp.glossary` formatı: `KAYNAK->HEDEF`."""
    return [f"{src}->{tgt}" for src, tgt in mapping.items() if src and tgt]


# Oy-birliği oylaması eşikleri (genel; bölüm/terim ezberi yok).
VOTE_MIN_COUNT_DEFAULT = 4
VOTE_MAX_TERMS_DEFAULT = 5
VOTE_MAX_OCCURRENCES_PER_TERM = 8
VOTE_MASK_TOKEN = "___"
VOTE_MIN_PREFIX_LEN = 5
VOTE_MAJORITY_RATIO = 0.5


def _mask_term_occurrences(text: str, term: str) -> str:
    """Terim geçişlerini `___` ile maskeler (büyük/küçük duyarsız, kelime sınırı)."""
    return re.sub(
        r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])",
        VOTE_MASK_TOKEN,
        text,
        flags=re.IGNORECASE,
    )


def _diff_spans(full_tr: str, masked_tr: str) -> list[str]:
    """Maskeli çeviride kaybolan aralıklar = terimin o cümledeki karşılığı.

    Model cümleyi yeniden kurarsa diff gürültü üretir; oylama aşaması
    uzlaşmayan gürültüyü eler (çoğunluk yoksa kilit yok).
    """
    from difflib import SequenceMatcher

    full_toks = re.findall(r"\S+", full_tr or "")
    masked_toks = re.findall(r"\S+", masked_tr or "")
    if not full_toks or not masked_toks:
        return []
    spans: list[str] = []
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(
        None, full_toks, masked_toks, autojunk=False
    ).get_opcodes():
        if tag in ("delete", "replace"):
            chunk = " ".join(full_toks[i1:i2]).strip(" \t\"'“”‘’.,!?;:()")
            if (
                chunk
                and len(chunk.split()) <= 4
                and len(chunk) <= 40
                and re.search(r"\w", chunk, re.UNICODE)
            ):
                spans.append(chunk)
    return spans


def _cluster_key(a: str, b: str) -> bool:
    """İki aday aynı gövde ailesinden mi (≥5 harf ortak önek)?"""
    a_n = re.sub(r"[^\w]", "", a.casefold())
    b_n = re.sub(r"[^\w]", "", b.casefold())
    n = 0
    for ca, cb in zip(a_n, b_n):
        if ca != cb:
            break
        n += 1
    return n >= VOTE_MIN_PREFIX_LEN


def vote_term_rendering(
    translator: TermTranslator,
    term: str,
    occurrence_texts: Sequence[str],
) -> str | None:
    """Geçiş cümlelerini maskeli/maskesiz çevirip çoğunluk karşılığı bulur.

    Dönüş: kanonik taban biçim (ailenin en kısa üyesi) veya None.
    Maliyet: 2 toplu çağrı (dolu + maskeli). Model kararsızsa None.
    """
    sources = [t for t in occurrence_texts if t and t.strip()]
    if not sources:
        return None
    masked = [_mask_term_occurrences(t, term) for t in sources]
    try:
        tr_full = translator.translate_batch(list(sources))
        tr_masked = translator.translate_batch(masked)
    except Exception as exc:
        logger.warning(f"Oylama çevirisi başarısız ({term}): {exc}")
        return None
    candidates: list[str] = []
    for full, mask in zip(tr_full, tr_masked):
        candidates.extend(_diff_spans(full or "", mask or ""))
    if not candidates:
        return None
    clusters: list[list[str]] = []
    for cand in candidates:
        placed = False
        for cluster in clusters:
            if _cluster_key(cand, cluster[0]):
                cluster.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])
    clusters.sort(key=len, reverse=True)
    best = clusters[0]
    if len(best) < 2 or len(best) / len(candidates) < VOTE_MAJORITY_RATIO:
        return None
    return min(best, key=len)


def vote_rejected_terms(
    translator: TermTranslator,
    terms: Sequence[ChapterTerm],
    locked: dict[str, str],
    texts: Sequence[str],
    block_ids: Sequence[int],
    min_count: int = VOTE_MIN_COUNT_DEFAULT,
    max_terms: int = VOTE_MAX_TERMS_DEFAULT,
) -> dict[str, str]:
    """Tutarlılık kapanışında reddedilen terimler için oy-birliği kilidi.

    Yalnız `min_count` üzeri ve kilitsiz terimler (tavanlı). Bulunamazsa
    boş döner — eski davranış korunur.
    """
    by_id = dict(zip(block_ids, texts))
    winners: dict[str, str] = {}
    considered = 0
    for term in terms:
        if term.term in locked or term.count < min_count:
            continue
        if considered >= max_terms:
            break
        considered += 1
        occ = [
            by_id[bid]
            for bid in term.block_ids
            if by_id.get(bid)
            and re.search(
                r"(?<![A-Za-z])" + re.escape(term.term) + r"(?![A-Za-z])",
                by_id[bid],
                re.IGNORECASE,
            )
        ][:VOTE_MAX_OCCURRENCES_PER_TERM]
        if len(occ) < 2:
            continue
        canonical = vote_term_rendering(translator, term.term, occ)
        if canonical:
            winners[term.term] = canonical
            logger.info(f"Oylama kilidi: {term.term} -> {canonical}")
        else:
            logger.info(f"Oylama kilitsiz bıraktı: {term.term} (uzlaşı yok)")
    return winners


def write_glossary_json(
    path: str | Path,
    mapping: dict[str, str],
    terms: Sequence[ChapterTerm],
    observed: Sequence[ChapterTerm] = (),
    methods: dict[str, str] | None = None,
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
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
