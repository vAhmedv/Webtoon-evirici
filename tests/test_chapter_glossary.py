"""Faz 3 bölüm-terim kilidi testleri: tamamı sentetik, LLM yok, bölüm ezberi yok."""

from core.translation.chapter_glossary import (
    ChapterTerm,
    _diff_spans,
    _mask_term_occurrences,
    _standalone_matches_family,
    collect_vote_candidates,
    extract_observed_terms,
    extract_repeated_terms,
    glossary_entries,
    resolve_chapter_glossary,
    vote_rejected_terms,
    vote_surface_hits,
    write_glossary_json,
)


class _StubTranslator:
    def __init__(self, mapping: dict[str, str] | None = None, fail: bool = False):
        self.mapping = mapping or {}
        self.calls: list[list[str]] = []
        self.fail = fail

    def translate_batch(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("model yok")
        return [self.mapping.get(t, f"TR-{t}") for t in texts]


def test_all_caps_repetition_locked() -> None:
    texts = [
        "I AM A CRAFTER BY TRADE",
        "THE CRAFTER GUILD REJECTED ME",
        "EVERY CRAFTER MUST REGISTER",
        "UNRELATED SENTENCE HERE",
    ]
    terms = extract_repeated_terms(texts, [1, 2, 3, 4])
    found = {t.term: t.count for t in terms}
    assert found.get("CRAFTER") == 3


def test_single_occurrence_not_locked() -> None:
    terms = extract_repeated_terms(["ONLY ONCE VISIBLE", "OTHER WORDS HERE", "THIRD LINE HERE"])
    assert all(t.count >= 3 for t in terms)
    assert "VISIBLE" not in {t.term for t in terms}


def test_stopwords_and_short_tokens_excluded() -> None:
    texts = ["THE THE THE AND AND", "A A A BUT BUT", "YOU YOU YOU ARE ARE"] * 2
    terms = extract_repeated_terms(texts, list(range(1, 7)))
    assert terms == []


def test_quoted_phrases_observed_not_locked() -> None:
    texts = [
        'ACTIVATE "SILENT CHAIN" NOW',
        'USE "SILENT CHAIN" AGAIN',
        '"SILENT CHAIN" IS READY',
    ]
    assert extract_repeated_terms(texts, [1, 2, 3]) == []
    observed = extract_observed_terms(texts, [1, 2, 3])
    assert any(t.term == "SILENT CHAIN" and t.count == 3 for t in observed)


def test_inflected_forms_never_locked() -> None:
    """Çekimlenen biçimler kilitlenemez (sentinel yalın hal basar)."""
    texts = [
        "MY WEAPONS ARE READY WEAPONS",
        "TAKE THE WEAPONS WEAPONS",
        "SHARPENING WEAPONS IS HARD",
        "HE IS RUNNING RUNNING FAST",
        "THEY ARE RUNNING RUNNING NOW",
        "STOP RUNNING AROUND HERE",
        "WANTED POSTERS WANTED EVERYWHERE",
        "WANTED DEAD WANTED ALIVE",
        "STILL WANTED BY ALL",
    ]
    terms = extract_repeated_terms(texts, list(range(1, 10)))
    locked = {t.term for t in terms}
    assert "WEAPONS" not in locked  # çoğul
    assert "RUNNING" not in locked  # -ING eylem
    assert "WANTED" not in locked  # -ED geçmiş


def test_lowercase_form_blocks_lock() -> None:
    """Küçük hali bölümde geçen kelime terim değildir (HERE/here)."""
    texts = [
        "COME HERE RIGHT NOW",
        "STAY HERE WITH ME",
        "OVER HERE QUICKLY",
        "please come here quietly",
    ]
    terms = extract_repeated_terms(texts, [1, 2, 3, 4])
    assert "HERE" not in {t.term for t in terms}


def test_caps_only_term_still_locks() -> None:
    """Yalnız büyük yazılan terim kilitlenir."""
    texts = [
        "THE CRAFTER ARRIVED",
        "ASK THE CRAFTER FIRST",
        "FIND ANOTHER CRAFTER",
    ]
    terms = extract_repeated_terms(texts, [1, 2, 3])
    assert any(t.term == "CRAFTER" and t.count == 3 for t in terms)


def test_contractions_never_locked() -> None:
    texts = [
        "YOU'RE LATE AGAIN",
        "YOU'RE WRONG HERE",
        "YOU'RE EARLY TODAY",
    ]
    terms = extract_repeated_terms(texts, [1, 2, 3])
    assert terms == []


def test_non_plural_s_kept() -> None:
    """SS ile bitenler çoğul değildir (CLASS, BOSS)."""
    texts = ["BOSS BOSS BOSS FIGHT", "BOSS BOSS BOSS ROOM", "BOSS BOSS BOSS LOOT"]
    terms = extract_repeated_terms(texts, [1, 2, 3])
    assert any(t.term == "BOSS" for t in terms)


def test_deictics_and_common_adjectives_never_locked() -> None:
    texts = [
        "COME HERE RIGHT NOW",
        "STAY HERE WITH ME",
        "OVER HERE QUICKLY",
        "THE STRONG WARRIOR CAME",
        "A STRONG SHIELD BROKE",
        "THAT STRONG MAGIC FAILED",
    ]
    terms = extract_repeated_terms(texts, list(range(1, 7)))
    locked = {t.term for t in terms}
    assert "HERE" not in locked
    assert "STRONG" not in locked


def test_ed_forms_never_locked() -> None:
    texts = [
        "WANTED POSTERS EVERYWHERE",
        "WANTED DEAD OR ALIVE",
        "STILL WANTED BY ALL",
    ]
    terms = extract_repeated_terms(texts, [1, 2, 3])
    assert "WANTED" not in {t.term for t in terms}


def test_title_case_long_words_observed_not_locked() -> None:
    texts = [
        "Welcome to Adventure Guild",
        "The Adventure Guild master nodded",
        "Leave the Adventure Guild hall",
    ]
    assert extract_repeated_terms(texts, [1, 2, 3]) == []
    observed = extract_observed_terms(texts, [1, 2, 3])
    assert any("Adventure" in t.term for t in observed)


def test_resolve_single_batch_call_and_empty_safe() -> None:
    stub = _StubTranslator({"CRAFTER": "Zanaatkar"})
    terms = [ChapterTerm("CRAFTER", 5, [1, 2]), ChapterTerm("GUILD", 3, [2])]
    mapping, _methods = resolve_chapter_glossary(stub, terms)
    assert mapping == {"CRAFTER": "Zanaatkar", "GUILD": "TR-GUILD"}
    assert len(stub.calls) == 1
    assert stub.calls[0] == ["CRAFTER", "GUILD"]


def test_resolve_failure_returns_empty() -> None:
    mapping, _methods = resolve_chapter_glossary(_StubTranslator(fail=True), [ChapterTerm("X", 9)])
    assert mapping == {}


def test_broken_targets_rejected() -> None:
    stub = _StubTranslator({"A": "", "B": "x", "C": "bu bir açıklama cümlesi olarak döndü", "D": "?!...", "E": "Zanaatkar"})
    terms = [ChapterTerm(k, 5) for k in "ABCDE"]
    mapping, _methods = resolve_chapter_glossary(stub, terms)
    assert mapping == {"E": "Zanaatkar"}


def test_consistency_closure_locks_matching_sense() -> None:
    """Bağımsız hedef geçiş cümlelerinde geçiyorsa kilitlenir."""

    class _CtxStub(_StubTranslator):
        def translate_batch(self, texts: list[str]) -> list[str]:
            self.calls.append(list(texts))
            out = []
            for t in texts:
                if t == "CRAFTER":
                    out.append("Üretici")
                elif "CRAFTER" in t:
                    out.append("Ben bir Üreticiyim")
                else:
                    out.append(f"TR-{t}")
            return out

    terms = [ChapterTerm("CRAFTER", 3, [1, 2])]
    texts = ["CRAFTER", "I AM A CRAFTER BY TRADE"]
    mapping, _methods = resolve_chapter_glossary(_CtxStub(), terms, texts=texts, block_ids=[1, 2])
    assert mapping == {"CRAFTER": "Üretici"}


def test_consistency_closure_rejects_mismatch() -> None:
    """Bağımsız hedef cümlelerde yoksa kilit yok (GUILD/LİG vs Lonca)."""

    class _CtxStub(_StubTranslator):
        def translate_batch(self, texts: list[str]) -> list[str]:
            self.calls.append(list(texts))
            out = []
            for t in texts:
                if t == "GUILD":
                    out.append("Lig")
                elif "GUILD" in t:
                    out.append("Ünlü bir loncadan geliyorum")
                else:
                    out.append(f"TR-{t}")
            return out

    terms = [ChapterTerm("GUILD", 3, [1, 2])]
    texts = ["GUILD", "ARE YOU FROM SOME FAMOUS GUILD"]
    mapping, _methods = resolve_chapter_glossary(_CtxStub(), terms, texts=texts, block_ids=[1, 2])
    assert mapping == {}


def test_resolve_empty_terms_no_call() -> None:
    stub = _StubTranslator()
    assert resolve_chapter_glossary(stub, []) == ({}, {})
    assert stub.calls == []


def test_glossary_entries_format() -> None:
    assert glossary_entries({"CRAFTER": "Zanaatkar"}) == ["CRAFTER->Zanaatkar"]
    assert glossary_entries({}) == []


def test_write_glossary_json_roundtrip(tmp_path) -> None:
    terms = [ChapterTerm("CRAFTER", 5, [11, 22])]
    path = write_glossary_json(
        tmp_path / "analysis" / "glossary.json", {"CRAFTER": "Zanaatkar"}, terms
    )
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    [locked] = payload["locked_terms"]
    assert locked["source"] == "CRAFTER"
    assert locked["target"] == "Zanaatkar"
    assert locked["occurrences"] == 5
    assert locked["block_ids"] == [11, 22]


def test_mask_term_occurrences() -> None:
    assert _mask_term_occurrences("I AM A CRAFTER TODAY", "CRAFTER") == "I AM A ___ TODAY"
    assert _mask_term_occurrences("CRAFTERS ARE COMING", "CRAFTER") == "CRAFTERS ARE COMING"
    assert _mask_term_occurrences("ask the crafter first", "CRAFTER") == "ask the ___ first"


def test_diff_spans_finds_rendering() -> None:
    assert _diff_spans("Ben bir Üreticiyim", "Ben bir ___") == ["Üreticiyim"]
    assert _diff_spans("Aynı cümle burada", "Aynı cümle burada") == []


def test_surface_votes_over_prefix() -> None:
    """kara/karar: önek aynı ama yüzey farklı → kilit yok."""
    assert vote_surface_hits("KARA", ["kara", "karar", "kara"]) == ["kara", "kara"]


class _VoteStub:
    """Maskeli/maskesiz çiftleri hazır cevaplarla çevirir."""

    FULL = {
        "I AM A CRAFTER": "Ben bir Üreticiyim",
        "THE CRAFTER CAME": "Üretici geldi",
        "ASK THE CRAFTER": "Üreticiye sor",
    }
    MASKED = {
        "I AM A ___": "Ben bir ___",
        "THE ___ CAME": "___ geldi",
        "ASK THE ___": "___ sor",
    }

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate_batch(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            if "___" in t:
                out.append(self.MASKED.get(t, t))
            else:
                out.append(self.FULL.get(t, f"TR-{t}"))
        return out


def test_standalone_matches_family() -> None:
    assert _standalone_matches_family("Dünya", ["dünyada", "dünyasında"])
    assert _standalone_matches_family("Üretici", ["Üreticiyim", "Üretici"])
    assert not _standalone_matches_family("Üretici", ["CRAFTER", "crafter"])
    assert not _standalone_matches_family("Lig", ["lonca", "loncadan"])
    assert not _standalone_matches_family("", ["x"])


def test_vote_surface_hits_morphology() -> None:
    """Yüzey üyeliği: usta/ustalar aynı, kara/karar farklı sözcük."""
    assert vote_surface_hits("USTA", ["USTA", "ustalar", "usta"]) == ["USTA", "ustalar", "usta"]
    assert vote_surface_hits("KARA", ["kara", "karar"]) == ["kara"]
    assert vote_surface_hits("DÜNYA", ["dünyada", "dünyasında"]) == ["dünyada"]
    assert vote_surface_hits("LİG", ["lonca", "loncadan"]) == []
    assert vote_surface_hits("Üretici", ["CRAFTER", "crafter"]) == []


def test_vote_winning_family_majority() -> None:
    stub = _VoteStub()
    candidates = collect_vote_candidates(
        stub, "CRAFTER", ["I AM A CRAFTER", "THE CRAFTER CAME", "ASK THE CRAFTER"]
    )
    assert len(candidates) == 3
    assert vote_surface_hits("Üretici", candidates) == candidates
    assert len(stub.calls) == 2  # dolu + maskeli


def test_vote_winning_family_no_consensus() -> None:
    class _NoiseStub:
        def translate_batch(self, texts: list[str]) -> list[str]:
            return [f"YANIT {i} FARKLI KELİMELER" for i, _ in enumerate(texts)]

    candidates = collect_vote_candidates(
        _NoiseStub(), "XQZT", ["A XQZT B", "C XQZT D", "E XQZT F"]
    )
    assert vote_surface_hits("XQZT", candidates) == []


def test_vote_rejected_terms_skips_locked_and_rare() -> None:
    stub = _VoteStub()
    terms = [
        ChapterTerm("CRAFTER", 5, [1, 2, 3]),
        ChapterTerm("LOCKED", 9, [4]),
        ChapterTerm("RARE", 2, [5]),
    ]
    winners = vote_rejected_terms(
        stub,
        terms,
        {"LOCKED": "Kilitli"},
        ["I AM A CRAFTER", "THE CRAFTER CAME", "ASK THE CRAFTER", "x", "y"],
        [1, 2, 3, 4, 5],
        standalone={"CRAFTER": "Üretici"},
    )
    assert winners == {"CRAFTER": "Üretici"}


def test_vote_rejected_mismatch_no_lock() -> None:
    """Aile bağımsız hedefle uyuşmazsa kilit yok (yankı-ailesi)."""
    stub = _VoteStub()
    terms = [ChapterTerm("CRAFTER", 5, [1, 2, 3])]
    winners = vote_rejected_terms(
        stub,
        terms,
        {},
        ["CRAFTER", "CRAFTER.", "CRAFTER!"],
        [1, 2, 3],
        standalone={"CRAFTER": "Üretici"},
    )
    assert winners == {}
