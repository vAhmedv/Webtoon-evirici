"""Application-Level Terminology Protection, Opaque Sentinels, and Named-Term Detection.

Protects approved glossary terms and unapproved named abilities/skills/titles
using opaque sentinels (__WTTERM0001__) before calling TranslateGemma, and restores
canonical forms and Turkish morphology afterwards.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from providers.translation.base import TranslationItem


NAMED_TERM_PATTERNS = [
    # Bracketed ALL-CAPS tokens are UI-style named abilities/terms.  Protect the
    # identity inside the brackets while leaving the structural markers intact.
    re.compile(r"\[(?P<term>[A-Z][A-Z0-9 _-]{1,38}[A-Z0-9])\]"),
    re.compile(r"\b(?i:it's|is)\s+(?i:called)\s+(?P<term>[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)"),
    re.compile(r"\b(?i:activate)\s+(?P<term>[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)"),
    re.compile(r"\b(?i:learned)\s+(?P<term>[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)"),
    re.compile(r"^(?P<term>[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\s+(?:is|allows|keeps|requires|remains|failed|cooldown|breaks)\b", re.MULTILINE),
    re.compile(
        r"^\s*(?:PASSIVE SKILL|TITLE|UNIQUE TRAIT|SKILL|ABILITY)"
        r"\s*(?:ACQUIRED|AVAILABLE)?\s*:?\s*(?P<term>[A-Z][A-Z0-9 -]{2,29})\s*$",
        re.MULTILINE,
    ),
]

EXCLUDED_WORDS = {
    "THE", "A", "AN", "AND", "OR", "BUT", "NOT", "IF", "THEN", "WHEN", "WHERE", "WHY",
    "HOW", "WHAT", "WHO", "MY", "YOUR", "HIS", "HER", "ITS", "OUR", "THEIR", "THIS", "THAT",
    "IT", "HE", "SHE", "THEY", "WE", "YOU", "I", "CAN", "COULD", "WOULD", "SHOULD", "MUST",
    "TO", "OF", "FOR", "FROM", "WITH", "WITHOUT", "ON", "IN", "AT", "BY", "AS", "UP", "DOWN",
}

OPAQUE_SENTINEL_PATTERN = re.compile(r"__WTTERM(?:[0-9A-Z_]+)?", re.IGNORECASE)
_HIGH_VOWELS = "ıiuü"
_TURKISH_VOWELS = "aeıioöuü"
_SUFFIX_LETTERS = "A-Za-zÇĞİÖŞÜçğıöşü"
_ENGLISH_CARDINAL_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "trillion", "dozen",
}

# Sayı-sözcüklerinin Türkçe yüzey formları (rakam + yalın + sıra).
# Kapalı sınıf dil verisi — bölümden bağımsız. B87 sınıfı
# ("LEVEL ONE" → "Seviyesiniz", "birinci" düştü) bekçisi.
_TR_NUMBER_SURFACES = {
    "zero": {"0", "sıfır", "sifir"},
    "one": {"1", "bir", "biri", "birinci", "ilk"},
    "two": {"2", "iki", "ikinci"},
    "three": {"3", "üç", "uc", "üçüncü", "ucuncu"},
    "four": {"4", "dört", "dort", "dördüncü", "dorduncu"},
    "five": {"5", "beş", "bes", "beşinci", "besinci"},
    "six": {"6", "altı", "alti", "altıncı", "altinci"},
    "seven": {"7", "yedi", "yedinci"},
    "eight": {"8", "sekiz", "sekizinci"},
    "nine": {"9", "dokuz", "dokuzuncu"},
    "ten": {"10", "on", "onuncu"},
    "eleven": {"11", "onbir", "onbirinci"},
    "twelve": {"12", "oniki", "onikinci"},
}

_DROPPED_CONTENT_MIN_TOKENS = 2
_DROPPED_CONTENT_MAX_RATIO = 0.6
# Kısa kaynaklarda sıkışma meşrudur ("INTO THE WORLD!"→"Dünyaya!");
# buharlaşma hükmü için en az bu kadar kaynak gerekir.
_DROPPED_CONTENT_MIN_SRC_LEN = 20
# Kısa-istisna (B256 sınıfı): ≤25kr kaynak tek içerik sözcüğünü yitirip
# yarıya inerse ("BUT ALLEN..."→"DELİ.") buharlaşmadır. Oran kapısı
# (0.5) doğru kısa çevirileri korur ("DAMMIT"→"Kahretsin!" 1.5).
_DROPPED_SHORT_MAX_SRC_LEN = 25
_DROPPED_SHORT_MAX_RATIO = 0.5


def _word_present(word: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.IGNORECASE) is not None


# "this/that/the one" zamirdir (bu kişi/şey), sayı değildir:
# "THIS ONE FELL" → "bu düştü", "THE ONE WHO HIRED" → "tutan kişi".
# Sayı hükmü (dropped_number_token) bu kullanıma ateşlemez.
_DEMONSTRATIVE_ONE_RE = re.compile(r"\b(this|that|the)\s+one\b", re.IGNORECASE)


def _has_numeral_one(src: str, src_words: set[str]) -> bool:
    """Kaynakta zamir-olmayan 'one' (gerçek sayı) var mı?"""
    if "one" not in src_words:
        return False
    bare = _DEMONSTRATIVE_ONE_RE.sub(" ", src)
    return "one" in set(re.findall(r"[A-Za-z]+", bare.casefold()))


_TR_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+")
_TR_CAPS_RUN_RE = re.compile(r"[A-ZÇĞİÖŞÜ]{4,}")


def find_name_glue(
    source_text: str,
    restored_translation: str,
    protected_source_terms: set[str] | None = None,
) -> list[str]:
    """Ad-yapışma bekçisi (madde 2): kesmesiz ek almış ad.

    "HYUNJInin" (kaynak HYUNJI): kesme işareti yok + gövde kaynakla
    birebir → REVIEW. Türkçe özel-ad imlası kesme ister; kesmesiz
    yapışma ya ek-düşürme ya da ad-bozulmasıdır (HYUNJI→HYUNJIN).
    - Tam yankı (TR==kaynak) muaf (S0 Katman-1; S2/render halleder).
    - Birebir yankı-sözcük muaf ("HYUNJI" aynen duruyorsa).
    - Kilitli terimler muaf (restore kendi imlasını üretir).
    - Küçük-har modern sözcükler etkilenmez (büyük-harf koşusu aranır).
    Çeviri null'lanmaz — uyarı listesine eklenir.
    """
    warnings: list[str] = []
    src = source_text or ""
    tr = restored_translation or ""
    if not src.strip() or not tr.strip():
        return warnings
    norm = lambda t: " ".join(t.split()).casefold()  # noqa: E731
    if norm(src) == norm(tr):
        return warnings
    src_caps = {tok for tok in re.findall(r"[A-Z]{4,}", src) if tok not in EXCLUDED_WORDS}
    if not src_caps:
        return warnings
    protected = {s.casefold() for s in (protected_source_terms or set())}
    for word in _TR_WORD_RE.findall(tr):
        if "'" in word or "\u2019" in word:
            continue
        run = _TR_CAPS_RUN_RE.match(word)
        if not run:
            continue
        stem = run.group(0)
        if stem.casefold() in protected:
            continue
        if any(stem.casefold() == tok.casefold() for tok in src_caps):
            # Birebir yankı-sözcük mü (kesmesiz ama eksiz)?
            if word.casefold() == stem.casefold():
                continue
            warnings.append("name_glue")
            break
    return warnings


def find_dropped_source_tokens(
    source_text: str,
    restored_translation: str,
    protected_source_terms: set[str] | None = None,
) -> list[str]:
    """Ad-düşürme bekçisi (F2): kaynakta durup çeviride ailesi olmayan içerik.

    İki kapalı kural (eşikler mutlak-sayı değil, dilbilgisel):
    - `dropped_number_token`: sayı-sözcüğü (ONE..TWELVE) kaynakta var,
      çeviride rakam/karşılık yok. Sayılar yankılanmaz, düşmesi anlam kaydırır.
    - `dropped_content_token`: ≥2 büyük-içerik-token (≥4 harf, işlev
      sözcüğü değil, kilitli-terim değil) düşmüş VE blok boyu küçülmüş
      (TR/EN<0.6 — içerik buharlaşmış). Normal oranlı çevirilerde ortak
      adların çevrilmesi (WORLD→dünya) asla ateşlemez. Kısa kaynaklarda
      (<20kr) sıkışma meşru sayılır.
    - Kısa-istisna: ≤25kr kaynakta TEK içerik-token düşmüş VE oran<0.5
      ise yine `dropped_content_token` (B256: ad buharlaşmış).
    Çeviri null'lanmaz — uyarı listesine eklenir, REVIEW kararı analyzer'ındır.
    """
    warnings: list[str] = []
    src = source_text or ""
    tr = restored_translation or ""
    if not src.strip() or not tr.strip():
        return warnings

    src_words = set(re.findall(r"[A-Za-z]+", src.casefold()))
    for cardinal, surfaces in _TR_NUMBER_SURFACES.items():
        if cardinal == "one":
            if not _has_numeral_one(src, src_words):
                continue
        elif cardinal not in src_words:
            continue
        if not any(_word_present(s, tr) for s in surfaces):
            warnings.append("dropped_number_token")
            break

    protected = {s.casefold() for s in (protected_source_terms or set())}
    eligible = [
        tok for tok in re.findall(r"[A-Z]{4,}", src)
        if tok not in EXCLUDED_WORDS and tok.casefold() not in protected
    ]
    ratio = len(tr.strip()) / max(1, len(src.strip()))
    if len(src.strip()) >= _DROPPED_CONTENT_MIN_SRC_LEN and len(eligible) >= _DROPPED_CONTENT_MIN_TOKENS:
        dropped = [tok for tok in set(eligible) if not _word_present(tok, tr)]
        if len(dropped) >= _DROPPED_CONTENT_MIN_TOKENS and ratio < _DROPPED_CONTENT_MAX_RATIO:
            warnings.append("dropped_content_token")
    elif len(src.strip()) <= _DROPPED_SHORT_MAX_SRC_LEN and len(eligible) >= 1:
        dropped = [tok for tok in set(eligible) if not _word_present(tok, tr)]
        if dropped and ratio < _DROPPED_SHORT_MAX_RATIO:
            warnings.append("dropped_content_token")
    return warnings


# Akraba-küçük-liste DENEYİ (F6-sonrası, yumuşak iz — pipeline'a bağlı DEĞİL).
# Kapalı sınıf dil verisi, bölümden bağımsız. "PARENTS→Aile", "SISTER→kızı"
# gibi akıcı-ama-yanlışları yakalamak için: kaynakta akraba sözcüğü var,
# çeviride ailesinden HİÇBİR gövde yoksa `kinship_ambiguous` döner.
# ÖLÜMCÜL DEĞİL (fatal sete eklenmez) — basım durmaz, yalnız izlenir.
# Deney yeşilse bağlanır, sel yaparsa öldürülür.
_KINSHIP_TR_STEMS = {
    "parents": ("ebeve", "anne", "baba"),
    "parent": ("ebeve", "anne", "baba"),
    "mother": ("anne",),
    "father": ("baba",),
    "sister": ("kızkardeş", "kız kardeş", "kardeş", "abla", "bacı"),
    "brother": ("erkekkardeş", "erkek kardeş", "kardeş", "abi", "ağabey", "agabey"),
    "daughter": ("kız", "evlat"),
    "son": ("oğu", "oğl", "ogl", "evlat"),
}


def find_kinship_mismatch(
    source_text: str,
    restored_translation: str,
) -> list[str]:
    """Akraba-anlam kayması izi (yumuşak): liste küçük, karar hafif."""
    src = source_text or ""
    tr = restored_translation or ""
    if not src.strip() or not tr.strip():
        return []
    norm = lambda t: " ".join(t.split()).casefold()  # noqa: E731
    if norm(src) == norm(tr):
        return []
    src_words = set(re.findall(r"[A-Za-z]+", src.casefold()))
    present = [k for k in _KINSHIP_TR_STEMS if k in src_words]
    if not present:
        return []
    tr_norm = " ".join(tr.split()).casefold()
    tr_words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", tr_norm)
    for kin in present:
        stems = _KINSHIP_TR_STEMS[kin]
        hit = False
        for stem in stems:
            s = stem.casefold()
            if " " in s:
                if s in tr_norm:
                    hit = True
                    break
            elif any(w.startswith(s) for w in tr_words):
                hit = True
                break
        if not hit:
            return ["kinship_ambiguous"]
    return []


@dataclass
class ProtectedTermMeta:
    sentinel: str
    source_original: str
    target_base: str
    is_approved: bool
    proper_name: bool
    source_term: str = ""
    source_suffix: str = ""
    source_cardinal_quantified: bool = False


def detect_named_terms_in_items(
    items: list[TranslationItem],
    candidate_store: Any | None = None,
) -> set[str]:
    """Scan clean English source items in a batch/chapter to detect named terms."""
    detected: set[str] = set()

    if candidate_store and hasattr(candidate_store, "candidates"):
        for k, cand in candidate_store.candidates.items():
            if cand.status in ("discovered", "provisional", "ready_for_review"):
                detected.add(cand.source.strip())

    for item in items:
        source = item.source.strip()
        if not source:
            continue

        for pattern in NAMED_TERM_PATTERNS:
            for match in pattern.finditer(source):
                term_str = match.group("term").strip(" .?!,:;\"'")

                words = term_str.split()
                if not words:
                    continue

                if len(words) == 1 and words[0].upper() in EXCLUDED_WORDS:
                    continue

                # Function-word-only spans are grammar, never named entities.
                # This prevents false sentinels such as "TO IT" without relying
                # on phrase-specific blacklists.
                lexical_words = [re.sub(r"[^A-Za-z]", "", word).upper() for word in words]
                if lexical_words and all(word in EXCLUDED_WORDS for word in lexical_words):
                    continue

                if all(w[0].isupper() for w in words if w and w[0].isalpha()):
                    detected.add(term_str)

        # A standalone question is a conservative term echo (e.g. "Phantom Thread?").
        # A declarative/imperative TitleCase sentence is not: protecting the whole
        # line would bypass ordinary prose such as "Activate Phantom Thread.".
        clean_line = re.sub(r"^[^\w]+|[^\w]+$", "", source).strip()
        line_words = clean_line.split()
        if (
            re.search(r"\?\s*[!?.]*$", source)
            and len(line_words) >= 2
            and all(w[0].isupper() for w in line_words if w and w[0].isalpha())
            and not any(w.upper() in EXCLUDED_WORDS for w in line_words)
        ):
            detected.add(clean_line)

    return detected


def is_term_only_source(
    source_text: str,
    approved_terms: dict[str, str],
    detected_named_terms: set[str],
) -> tuple[bool, str | None]:
    """Check if the entire source line consists ONLY of a protected named term (+ harmless punctuation).

    Returns (is_bypass: bool, bypass_translation: str | None).
    """
    clean_src = source_text.strip()
    if not clean_src:
        return False, None

    raw_alphanumeric = re.sub(r"^[^\w]+|[^\w]+$", "", clean_src).strip()
    punctuation_suffix = clean_src[len(re.sub(r"[^\w]+$", "", clean_src)):]
    punctuation_prefix = clean_src[:len(clean_src) - len(re.sub(r"^[^\w]+", "", clean_src))]

    all_terms: list[tuple[str, str, bool, bool]] = []
    for src_k, tgt_v in approved_terms.items():
        all_terms.append((src_k, tgt_v, True, True if " " in src_k or src_k[0].isupper() else False))
    for named_t in detected_named_terms:
        if not any(named_t.upper() == k.upper() for k in approved_terms):
            all_terms.append((named_t, named_t, False, True))

    for orig_src, target_val, is_approved, proper_name in all_terms:
        if raw_alphanumeric.upper() == orig_src.upper():
            final_tr = f"{punctuation_prefix}{target_val}{punctuation_suffix}"
            return True, final_tr

    return False, None


def _last_vowel(text: str) -> str:
    for char in reversed(text.lower()):
        if char in _TURKISH_VOWELS:
            return char
    return "e"


def _two_way_suffix(text: str) -> str:
    return "ler" if _last_vowel(text) in "eiöü" else "lar"


def _four_way_vowel(text: str) -> str:
    last = _last_vowel(text)
    if last in "aı":
        return "ı"
    if last in "ei":
        return "i"
    if last in "ou":
        return "u"
    return "ü"


def _two_way_vowel(text: str) -> str:
    return "e" if _last_vowel(text) in "eiöü" else "a"


def _last_alpha(text: str) -> str:
    for char in reversed(text.lower()):
        if char.isalpha():
            return char
    return ""


def _compound_target_parts(target: str) -> tuple[str, str] | None:
    """Return (prefix, unpossessed stem) for a multiword Turkish compound."""
    if " " not in target.strip():
        return None
    prefix, _, last_word = target.rpartition(" ")
    lowered = last_word.lower()
    if len(last_word) > 2 and lowered[-2:] in {"sı", "si", "su", "sü"}:
        return prefix, last_word[:-2]
    if len(last_word) > 2 and lowered[-1] in _HIGH_VOWELS:
        stem = last_word[:-1]
        reverse_softening = {"ğ": "k", "d": "t", "b": "p", "c": "ç"}
        if stem and stem[-1].lower() in reverse_softening:
            replacement = reverse_softening[stem[-1].lower()]
            stem = stem[:-1] + replacement
        return prefix, stem
    return None


def _pluralize_common_target(target: str) -> tuple[str, bool]:
    compound = _compound_target_parts(target)
    if not compound:
        return target + _two_way_suffix(target), False
    prefix, stem = compound
    plural_stem = stem + _two_way_suffix(stem)
    plural_word = plural_stem + _four_way_vowel(plural_stem)
    return f"{prefix} {plural_word}", True


def _apply_turkish_case(form: str, case: str, possessed: bool) -> str:
    ends_vowel = _last_alpha(form) in _TURKISH_VOWELS
    if case == "gen":
        buffer = "n" if ends_vowel else ""
        return form + buffer + _four_way_vowel(form) + "n"
    if case == "acc":
        buffer = "n" if ends_vowel and possessed else ("y" if ends_vowel else "")
        return form + buffer + _four_way_vowel(form)
    if case == "dat":
        buffer = "n" if ends_vowel and possessed else ("y" if ends_vowel else "")
        return form + buffer + _two_way_vowel(form)
    if case in {"loc", "abl"}:
        buffer = "n" if ends_vowel and possessed else ""
        onset = "t" if _last_alpha(form) in "fstkçşhp" else "d"
        suffix = onset + _two_way_vowel(form)
        return form + buffer + suffix + ("n" if case == "abl" else "")
    return form


def _inflect_target(target: str, category: str, proper_name: bool) -> str:
    if category == "bare":
        return target

    if category in {"copular", "person_1sg", "person_2sg", "person_1pl", "person_2pl"}:
        vowel = _four_way_vowel(target)
        ends_vowel = _last_alpha(target) in _TURKISH_VOWELS
        if category == "copular":
            onset = "t" if _last_alpha(target) in "fstkçşhp" else "d"
            suffix = onset + vowel + "r"
        elif category == "person_1sg":
            suffix = ("y" if ends_vowel else "") + vowel + "m"
        elif category == "person_2sg":
            suffix = "s" + vowel + "n"
        elif category == "person_1pl":
            suffix = ("y" if ends_vowel else "") + vowel + "z"
        else:
            suffix = "s" + vowel + "n" + vowel + "z"
        return target + ("'" if proper_name else "") + suffix

    plural = category == "plural" or category.startswith("plural_")
    case = category.removeprefix("plural_") if category.startswith("plural_") else ""

    if proper_name:
        if plural:
            plural_form = target + _two_way_suffix(target)
            return _apply_turkish_case(plural_form, case, False) if case else plural_form
        inflected = _apply_turkish_case(target, category, False)
        return target + "'" + inflected[len(target):]

    compound = _compound_target_parts(target)
    if plural:
        form, possessed = _pluralize_common_target(target)
    else:
        form, possessed = target, compound is not None
    return _apply_turkish_case(form, case or category, possessed) if (case or not plural) else form


def _suffix_category(source_suffix: str, translated_suffix: str) -> str | None:
    source_suffix = source_suffix.lower()
    suffix = translated_suffix.lower().replace("i̇", "i")

    if source_suffix == "'s":
        return "gen"

    nominal_sets = {
        "copular": {"dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"},
        "person_1sg": {"ım", "im", "um", "üm", "yım", "yim", "yum", "yüm"},
        "person_2sg": {"sın", "sin", "sun", "sün"},
        "person_1pl": {"ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz"},
        "person_2pl": {"sınız", "siniz", "sunuz", "sünüz"},
    }
    for category, forms in nominal_sets.items():
        if suffix in forms:
            return category

    plural_sets = {
        "plural_abl": {"lerden", "lardan", "lerinden", "larından"},
        "plural_loc": {"lerde", "larda", "lerinde", "larında"},
        "plural_dat": {"lere", "lara", "lerine", "larına"},
        "plural_gen": {"lerin", "ların", "lerinin", "larının"},
        "plural_acc": {"leri", "ları", "lerini", "larını"},
        "plural": {"ler", "lar"},
    }
    for category, forms in plural_sets.items():
        if suffix in forms:
            return category

    singular_sets = {
        "abl": {"den", "dan", "ten", "tan", "nden", "ndan"},
        "loc": {"de", "da", "te", "ta", "nde", "nda"},
        "dat": {"e", "a", "ye", "ya", "ne", "na"},
        "gen": {"in", "ın", "un", "ün", "nin", "nın", "nun", "nün"},
        "acc": {"i", "ı", "u", "ü", "yi", "yı", "yu", "yü", "ni", "nı", "nu", "nü"},
    }
    for category, forms in singular_sets.items():
        if suffix in forms:
            return f"plural_{category}" if source_suffix in {"s", "es"} else category

    if source_suffix in {"s", "es"}:
        return "plural"
    return "bare" if not suffix else None


def _target_surface_forms(meta: ProtectedTermMeta) -> set[str]:
    categories = {
        "bare", "gen", "acc", "dat", "loc", "abl", "plural",
        "plural_gen", "plural_acc", "plural_dat", "plural_loc", "plural_abl",
        "copular", "person_1sg", "person_2sg", "person_1pl", "person_2pl",
    }
    return {_inflect_target(meta.target_base, category, meta.proper_name) for category in categories}


def _has_immediate_cardinal_quantifier(source_text: str, term_start: int) -> bool:
    """Return whether the protected occurrence is immediately preceded by a cardinal."""
    prefix = source_text[:term_start].rstrip()
    token_match = re.search(r"(?:\d[\d,]*(?:\.\d+)?|[A-Za-z]+(?:-[A-Za-z]+)*)$", prefix)
    if not token_match:
        return False
    token = token_match.group(0).lower().replace(",", "")
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return True
    return all(part in _ENGLISH_CARDINAL_WORDS for part in token.split("-"))


def _remove_plural_from_category(category: str) -> str:
    if category == "plural":
        return "bare"
    if category.startswith("plural_"):
        return category.removeprefix("plural_")
    return category


def protect_source_text(
    source_text: str,
    approved_terms: dict[str, str],
    detected_named_terms: set[str],
    proper_name_terms: set[str] | None = None,
) -> tuple[str, dict[str, ProtectedTermMeta]]:
    """Prepare source text for TranslateGemma using opaque sentinels (__WTTERM0001__).

    Returns (protected_source_text, placeholder_map).
    """
    from core.translation.profile_discovery import candidate_phrase_pattern

    protected_text = source_text
    placeholder_map: dict[str, ProtectedTermMeta] = {}

    all_targets: list[tuple[str, str, bool, bool]] = []

    proper_keys = {term.strip().upper() for term in (proper_name_terms or set())}
    for src_k, tgt_v in approved_terms.items():
        # Yankı-kilitler (hedef==kaynak: ad/terim aynen korunur) imla
        # bakımından özel-addır — ekler kesmeyle gelir ("HYUNJI'nin").
        # Ortak-ad çevirileri (MONEY→Para) bundan etkilenmez.
        is_echo_lock = bool(src_k.strip()) and src_k.strip().casefold() == (tgt_v or "").strip().casefold()
        is_proper = src_k.strip().upper() in proper_keys or is_echo_lock
        all_targets.append((src_k, tgt_v, True, is_proper))

    for named_t in detected_named_terms:
        if not any(named_t.upper() == k.upper() for k in approved_terms):
            all_targets.append((named_t, named_t, False, True))

    all_targets.sort(key=lambda x: len(x[0]), reverse=True)

    sentinel_idx = 1
    for orig_src, target_val, is_approved, is_proper in all_targets:
        if not orig_src.strip():
            continue

        pattern = candidate_phrase_pattern(orig_src)
        if not pattern or not pattern.search(protected_text):
            continue

        def replacer(match: re.Match) -> str:
            nonlocal sentinel_idx
            sentinel = f"__WTTERM{sentinel_idx:04d}__"
            sentinel_idx += 1
            matched_str = match.group(0)
            meta = ProtectedTermMeta(
                sentinel=sentinel,
                source_original=matched_str,
                target_base=target_val,
                is_approved=is_approved,
                proper_name=is_proper,
                source_term=orig_src,
                source_suffix=match.groupdict().get("english_suffix") or "",
                source_cardinal_quantified=(
                    is_approved
                    and not is_proper
                    and _has_immediate_cardinal_quantifier(protected_text, match.start())
                ),
            )
            placeholder_map[sentinel] = meta
            return sentinel

        protected_text = pattern.sub(replacer, protected_text)

    return protected_text, placeholder_map


def restore_protected_translation(
    translated_text: str,
    placeholder_map: dict[str, ProtectedTermMeta],
) -> str:
    """Restore opaque sentinels back to canonical terms, handling Turkish morphology."""
    restored = translated_text

    sorted_placeholders = sorted(placeholder_map.items(), key=lambda x: len(x[0]), reverse=True)

    for sentinel, meta in sorted_placeholders:
        if not re.search(re.escape(sentinel), restored, re.IGNORECASE):
            continue

        pattern = re.compile(
            re.escape(sentinel)
            + rf"(?P<apostrophe>['’]?)(?P<suffix>[{_SUFFIX_LETTERS}]+)?",
            re.IGNORECASE,
        )

        def replacer(match: re.Match[str]) -> str:
            translated_suffix = match.group("suffix") or ""
            source_suffix = meta.source_suffix
            if meta.source_cardinal_quantified and not meta.proper_name:
                # Turkish cardinal noun phrases use the singular lexical noun.
                # Ignore source-side English plural when interpreting the model's
                # suffix, then remove only an explicitly produced plural category.
                source_suffix = ""
            category = _suffix_category(source_suffix, translated_suffix)
            if meta.source_cardinal_quantified and not meta.proper_name and category:
                category = _remove_plural_from_category(category)
            if category is not None:
                return _inflect_target(meta.target_base, category, meta.proper_name)
            # S5 çekim kapısı: bilinmeyen ek ("Dünya"+"ine") uydurma kelime
            # üretir — yalın taban basılır. Uydurma kelime, eksik ekten
            # her zaman kötüdür (S0: yanlış anlam yasak).
            return meta.target_base

        restored = pattern.sub(replacer, restored)

    # Madde 1: kilitli ortak-adlar cümle-ortasında küçük harfe iner
    # ("...çok fazla Para" → "...çok fazla para").
    return decapitalize_common_lock_targets(restored, placeholder_map)


def validate_protected_terms(
    restored_translation: str,
    placeholder_map: dict[str, ProtectedTermMeta],
) -> list[str]:
    """Verify that all approved terms' target bases appear in the restored translation."""
    warnings: list[str] = []

    from core.translation.profile_discovery import contains_candidate_phrase

    for meta in placeholder_map.values():
        if meta.is_approved:
            target_found = any(
                re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", restored_translation, re.IGNORECASE)
                for surface in _target_surface_forms(meta)
            )
            if not target_found and "approved_term_missing" not in warnings:
                warnings.append("approved_term_missing")

            source_term = meta.source_term or meta.source_original
            lexically_changed = source_term.casefold() != meta.target_base.casefold()
            if (
                lexically_changed
                and contains_candidate_phrase(source_term, restored_translation)
                and "approved_source_term_leakage" not in warnings
            ):
                warnings.append("approved_source_term_leakage")

    return warnings


def contains_unrestored_protected_term(text: str) -> bool:
    """Return whether any opaque protection sentinel survived restoration."""
    return bool(OPAQUE_SENTINEL_PATTERN.search(text or ""))


_TR_LOWER_FIRST_MAP = {"I": "ı", "İ": "i"}


def _tr_lower_first(word: str) -> str:
    """Türkçe-duyarlı ilk-harf küçültme (I→ı, İ→i)."""
    if not word:
        return word
    return _TR_LOWER_FIRST_MAP.get(word[0], word[0].lower()) + word[1:]


_SENTENCE_START_TAIL_RE = re.compile(r"[.?!…]['\"”’)\]]*\s*$")


def _is_sentence_start(prefix: str) -> bool:
    """Önek cümle-başı mı (tırnak/parantez toleranslı)?"""
    stripped = prefix.rstrip().rstrip("\"”'’)]}")
    return not stripped or bool(_SENTENCE_START_TAIL_RE.search(prefix.rstrip()))


def decapitalize_common_lock_targets(
    restored_translation: str,
    placeholder_map: dict[str, ProtectedTermMeta],
) -> str:
    """Kilitli ortak-ad yüzeylerini cümle-ortasında küçültür (madde 1).

    Kapsam bilerek DAR tutulur (eski testlerin kilitlediği sözleşme):
    - YALNIZCA tek-kelimelik yüzeyler ("Para"→"para"; "Gizli Diyar",
      "Ruh Taşı" gibi çok-kelimeliler ad/tamlamadır, dokunulmaz).
    - Yankı-hedefler (hedef==kaynak: HYUNJI, PORTAL) atlanır.
    - Sözlük-dışı hedefler atlanır ("Seul" gibi özel-ad riski) —
      hunspell yoksa kural tamamen pas geçer (fail-open).
    - Cümle-başları korunur; bağırma blokları es geçilir.
    """
    text = restored_translation or ""
    if not text:
        return text
    from core.translation.chapter_glossary import _spell_dictionary

    spell = _spell_dictionary()
    if spell is None:
        return text
    metas = [
        meta for meta in placeholder_map.values()
        if not meta.proper_name
        and (meta.target_base or "")
        and (meta.source_term or meta.source_original or "").casefold()
        != (meta.target_base or "").casefold()
    ]
    if not metas:
        return text
    alpha = [c for c in text if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.5:
        return text
    surfaces: set[str] = set()
    for meta in metas:
        for surface in _target_surface_forms(meta):
            if " " not in surface.strip():
                surfaces.add(surface)
    candidates = set()
    for surface in sorted(surfaces, key=len, reverse=True):
        lowered = _tr_lower_first(surface)
        if lowered == surface:
            continue
        try:
            known = bool(spell.lookup(lowered))
        except Exception:
            known = False
        if known:
            candidates.add(surface)
    if not candidates:
        return text
    for surface in sorted(candidates, key=len, reverse=True):
        def _fix(match: re.Match[str], _surface: str = surface) -> str:
            if _is_sentence_start(text[: match.start()]):
                return match.group(0)
            return _tr_lower_first(match.group(0))

        text = re.sub(
            r"(?<!\w)" + re.escape(surface) + r"(?!\w)",
            _fix,
            text,
        )
    return text


_STEM_DUP_MIN_PREFIX = 4


def collapse_stem_doubles(text: str) -> str:
    """Bitişik aynı-kök çiftlemeyi tekler (S5).

    "Seviye seviyesine" → "seviyesine" (çekimli/uzun olan tutulur).
    Eş-form tekrarlar korunur (vurgu/ikileme: "yavaş yavaş", "çok çok").
    Yalnızca farklı-form + ortak ≥4-harf önek çiftleri birleşir.
    """
    if not text:
        return text

    def _fix(match: re.Match[str]) -> str:
        first, second = match.group(1), match.group(2)
        if first.casefold() == second.casefold():
            return match.group(0)
        shared = 0
        for char_a, char_b in zip(first.casefold(), second.casefold()):
            if char_a != char_b:
                break
            shared += 1
        if shared >= _STEM_DUP_MIN_PREFIX:
            return second
        return match.group(0)

    return re.sub(r"(\w{4,}) (\w{4,})", _fix, text, flags=re.UNICODE)


def has_untranslated_source_prose(
    prepared_source_text: str,
    restored_translation: str,
    placeholder_map: dict[str, ProtectedTermMeta],
) -> bool:
    """Conservatively detect unchanged English prose around protected terms."""
    ordinary_source = OPAQUE_SENTINEL_PATTERN.sub(" ", prepared_source_text)
    source_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", ordinary_source.casefold())
    if not source_tokens:
        return False

    residual_output = restored_translation
    from core.translation.profile_discovery import candidate_phrase_pattern

    for meta in placeholder_map.values():
        source_pattern = candidate_phrase_pattern(meta.source_term or meta.source_original)
        if source_pattern:
            residual_output = source_pattern.sub(" ", residual_output)
        for surface in sorted(_target_surface_forms(meta), key=len, reverse=True):
            residual_output = re.sub(
                r"(?<!\w)" + re.escape(surface) + r"(?!\w)",
                " ",
                residual_output,
                flags=re.IGNORECASE,
            )

    output_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", residual_output.casefold())
    if not output_tokens:
        return False
    if source_tokens == output_tokens:
        return True

    matching_tokens = sum(
        block.size
        for block in SequenceMatcher(
            None,
            source_tokens,
            output_tokens,
            autojunk=False,
        ).get_matching_blocks()
    )
    if len(source_tokens) == 1:
        return source_tokens[0] in output_tokens
    if len(source_tokens) == 2:
        return matching_tokens == 2
    return matching_tokens >= 3 and matching_tokens / len(source_tokens) >= 0.75
