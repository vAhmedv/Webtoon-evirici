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
) -> dict[str, str]:
    """Benzersiz terimleri tek toplu çağrıda çevirip kilitler.

    Boş/bozuk/hedef çeviriler atlanır (kilit yok = eski davranış).
    """
    chosen = list(terms)[:max(0, max_terms)]
    if not chosen:
        return {}
    try:
        rendered = translator.translate_batch([t.term for t in chosen])
    except Exception as exc:
        logger.warning(f"Terim kilidi çözümlemesi başarısız, glossary boş: {exc}")
        return {}
    mapping: dict[str, str] = {}
    for term, target in zip(chosen, rendered):
        target = (target or "").strip()
        if _is_usable_target(target):
            mapping[term.term] = target
        else:
            logger.warning(
                f"Terim kilidi reddedildi ({term.term}): bozuk hedef {target!r}"
            )
    logger.info(f"Terim kilidi: {len(mapping)}/{len(chosen)} terim kilitlendi.")
    return mapping


def glossary_entries(mapping: dict[str, str]) -> list[str]:
    """Provider `inp.glossary` formatı: `KAYNAK->HEDEF`."""
    return [f"{src}->{tgt}" for src, tgt in mapping.items() if src and tgt]


def write_glossary_json(
    path: str | Path,
    mapping: dict[str, str],
    terms: Sequence[ChapterTerm],
    observed: Sequence[ChapterTerm] = (),
) -> Path:
    """`analysis/glossary.json` artefaktı: kilit + kanıt (blok id'leri)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {t.term.upper(): t for t in terms}
    payload = {
        "locked_terms": [
            {
                "source": src,
                "target": tgt,
                "occurrences": counts.get(src.upper(), ChapterTerm(src, 0)).count,
                "block_ids": counts.get(src.upper(), ChapterTerm(src, 0)).block_ids,
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
