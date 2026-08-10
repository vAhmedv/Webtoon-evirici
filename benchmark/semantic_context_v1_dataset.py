"""Frozen, provenance-labeled dataset for the Semantic Context V1 experiment."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from core.translation.semantic_context import LocalContextRegion, select_local_context


@dataclass(frozen=True)
class ContextualTestItem:
    id: int
    synthetic_context: bool
    source_origin: str
    source_region_id: int | None
    previous_context: tuple[str, ...]
    target_source: str
    next_context: tuple[str, ...]
    named_terms: tuple[str, ...]
    coverage: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["previous_context"] = list(self.previous_context)
        data["next_context"] = list(self.next_context)
        data["named_terms"] = list(self.named_terms)
        data["coverage"] = list(self.coverage)
        return data


# Verbatim 10-item reading-order sequence from
# scripts/qwen_translation_smoke_test.py::BUBBLES. That script labels these as
# real English bubble sources from one chapter.
REAL_CHAPTER_SEQUENCE = (
    (
        10,
        "JUDGING BY LUO TIAN'S PERFORMANCE JUST NOW, HE'S ALMOST ON PAR WITH A LEVEL 1 ABILITY USER WHO SPECIALIZES IN ARCHERY.",
    ),
    (11, "YOUNG MASTER YU, CAPTAIN GAO, WE NEED TO BE CAREFUL FROM HERE ON."),
    (
        12,
        "THESE GRAY WOLF BEASTS ARE SUPPOSED TO BE ACTIVE IN BLACKWIND RAVINE AHEAD OF US.",
    ),
    (13, "HU SAN, YOU'RE THE FASTEST. GO SCOUT THE PATH AHEAD."),
    (14, "THE FACT THAT THEY'VE APPEARED HERE IS PROBABLY NOT A GOOD SIGN."),
    (
        15,
        "CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER, AND THE REST OF THE TEAM ARE NO PUSHOVERS EITHER.",
    ),
    (16, "RELAX, KID. YOU SAW IT YOURSELF JUST NOW."),
    (17, "I'M USED TO IT."),
    (18, "COUNTLESS SPATIAL SECRET REALMS HAVE FORMED ALL AROUND THE WORLD."),
    (
        19,
        "MY NAME IS LUO TIAN. I'M NOT AN ABILITY USER-I'M A SECRET REALM GUIDE.",
    ),
)

_REAL_TERMS = (
    "LUO TIAN",
    "HU SAN",
    "GAO YUAN",
    "YU",
    "ABILITY USER",
    "SECRET REALM",
    "SECRET REALM GUIDE",
    "LEVEL 1",
    "BLACKWIND RAVINE",
)
_REAL_COVERAGE = {
    10: ("pronoun_reference", "comparison_semantics"),
    11: ("register", "dialogue_consistency"),
    12: ("tense_aspect", "location_reference"),
    13: ("register", "imperative"),
    14: ("pronoun_reference", "tense_aspect"),
    15: ("negation", "comparison_semantics"),
    16: ("pronoun_reference", "register"),
    17: ("pronoun_reference", "tense_aspect"),
    18: ("tense_aspect", "narration"),
    19: ("negation", "identity_reference"),
}


def _contains_term(source: str, term: str) -> bool:
    return bool(
        re.search(
            r"(?<!\w)" + r"\s+".join(re.escape(part) for part in term.split()) + r"(?!\w)",
            source,
            re.IGNORECASE,
        )
    )


def _real_items() -> list[ContextualTestItem]:
    regions = [
        LocalContextRegion(
            region_id=region_id,
            reading_order=index,
            source=source,
            region_type="dialogue",
            scene_id="qwen_smoke_real_chapter_sequence",
        )
        for index, (region_id, source) in enumerate(REAL_CHAPTER_SEQUENCE)
    ]
    items: list[ContextualTestItem] = []
    for benchmark_id, (region_id, source) in enumerate(REAL_CHAPTER_SEQUENCE, start=1):
        previous, following = select_local_context(
            regions,
            region_id,
            max_previous=3,
            max_next=1,
            max_order_distance=6,
        )
        items.append(
            ContextualTestItem(
                id=benchmark_id,
                synthetic_context=False,
                source_origin="scripts/qwen_translation_smoke_test.py::BUBBLES",
                source_region_id=region_id,
                previous_context=previous,
                target_source=source,
                next_context=following,
                named_terms=tuple(term for term in _REAL_TERMS if _contains_term(source, term)),
                coverage=_REAL_COVERAGE[region_id],
            )
        )
    return items


SYNTHETIC_ITEMS = (
    ContextualTestItem(
        id=11,
        synthetic_context=True,
        source_origin="synthetic_context_v1_adaptation_of_v4_007",
        source_region_id=None,
        previous_context=(
            "The raiders keep breaking free.",
            "Once Frost Chain catches them, they cannot move.",
        ),
        target_source="Frost Chain can hold three targets at once.",
        next_context=("Use it when they reach the gate.",),
        named_terms=("Frost Chain",),
        coverage=("polysemy", "hold", "ability_term"),
    ),
    ContextualTestItem(
        id=12,
        synthetic_context=True,
        source_origin="synthetic_context_v1_adaptation_of_v4_019",
        source_region_id=None,
        previous_context=(
            "There is an extra meal on my hotel bill.",
            "Only the front desk can add purchases to a room.",
        ),
        target_source="Who charged the meal to my room?",
        next_context=("I want the employee's name.",),
        named_terms=(),
        coverage=("polysemy", "charge", "question_semantics", "who"),
    ),
    ContextualTestItem(
        id=13,
        synthetic_context=True,
        source_origin="synthetic_context_v1_adaptation_of_v4_029",
        source_region_id=None,
        previous_context=(
            "I checked the chamber and found nobody inside.",
            "I stepped into the corridor for only a few seconds.",
        ),
        target_source="The chamber had been empty only a moment earlier.",
        next_context=("Now someone was standing beside the window.",),
        named_terms=(),
        coverage=("narration", "past_perfect", "state"),
    ),
    ContextualTestItem(
        id=14,
        synthetic_context=True,
        source_origin="synthetic_context_v1_adaptation_of_v4_030",
        source_region_id=None,
        previous_context=(
            "The stone floor was clean when I entered.",
            "I heard dripping behind me.",
        ),
        target_source="Now a line of wet footprints crossed the floor.",
        next_context=("They led toward the sealed door.",),
        named_terms=(),
        coverage=("narration", "action_state", "tense_aspect"),
    ),
    ContextualTestItem(
        id=15,
        synthetic_context=True,
        source_origin="synthetic_context_v1_polysemy_charge_motion",
        source_region_id=None,
        previous_context=("The boar lowered its head and scraped the ground.",),
        target_source="The boar charged before I could draw my sword.",
        next_context=("Its tusks struck the tree behind me.",),
        named_terms=(),
        coverage=("polysemy", "charge", "action"),
    ),
    ContextualTestItem(
        id=16,
        synthetic_context=True,
        source_origin="synthetic_context_v1_polysemy_leave_exclude",
        source_region_id=None,
        previous_context=("This argument is between you and me.",),
        target_source="Leave him out of this.",
        next_context=("He has nothing to do with our dispute.",),
        named_terms=(),
        coverage=("polysemy", "leave", "pronoun_reference"),
    ),
    ContextualTestItem(
        id=17,
        synthetic_context=True,
        source_origin="synthetic_context_v1_polysemy_leave_place",
        source_region_id=None,
        previous_context=("The caretaker will need the key later.",),
        target_source="Leave the key where you found it.",
        next_context=("Do not take it with you.",),
        named_terms=(),
        coverage=("polysemy", "leave", "pronoun_reference"),
    ),
    ContextualTestItem(
        id=18,
        synthetic_context=True,
        source_origin="synthetic_context_v1_polysemy_leave_depart",
        source_region_id=None,
        previous_context=("The gates will be locked at midnight.",),
        target_source="We need to leave before midnight.",
        next_context=("Pack only what you can carry.",),
        named_terms=(),
        coverage=("polysemy", "leave", "dialogue"),
    ),
    ContextualTestItem(
        id=19,
        synthetic_context=True,
        source_origin="synthetic_context_v1_pronoun_reference",
        source_region_id=None,
        previous_context=(
            "Mira handed me the cracked compass.",
            "She said the compass still pointed north.",
        ),
        target_source="I don't trust it.",
        next_context=("The needle keeps spinning.",),
        named_terms=(),
        coverage=("pronoun_reference", "negation"),
    ),
    ContextualTestItem(
        id=20,
        synthetic_context=True,
        source_origin="synthetic_context_v1_question_why",
        source_region_id=None,
        previous_context=("Arin abandoned the watch without permission.",),
        target_source="Why did he leave?",
        next_context=("The captain wants an explanation.",),
        named_terms=(),
        coverage=("question_semantics", "why", "pronoun_reference"),
    ),
    ContextualTestItem(
        id=21,
        synthetic_context=True,
        source_origin="synthetic_context_v1_polysemy_duck",
        source_region_id=None,
        previous_context=("An arrow flew toward Lena's head.",),
        target_source="She watched Lena duck.",
        next_context=("The arrow passed over Lena.",),
        named_terms=("Lena",),
        coverage=("polysemy", "pronoun_reference", "action"),
    ),
    ContextualTestItem(
        id=22,
        synthetic_context=True,
        source_origin="synthetic_context_v1_phrasal_verb",
        source_region_id=None,
        previous_context=(
            "The paralysis spell had kept his legs frozen.",
            "A minute later, he could move again.",
        ),
        target_source="The spell wore off.",
        next_context=("He stood up without help.",),
        named_terms=(),
        coverage=("phrasal_verb", "tense_aspect"),
    ),
    ContextualTestItem(
        id=23,
        synthetic_context=True,
        source_origin="synthetic_context_v1_present_perfect_progressive",
        source_region_id=None,
        previous_context=("The sentries began their vigil before sunrise.",),
        target_source="They have been waiting since dawn.",
        next_context=("They are still at the gate.",),
        named_terms=(),
        coverage=("pronoun_reference", "tense_aspect", "ongoing_state"),
    ),
    ContextualTestItem(
        id=24,
        synthetic_context=True,
        source_origin="synthetic_context_v1_state_action",
        source_region_id=None,
        previous_context=("The iron door had not moved all night.",),
        target_source="The door remained shut.",
        next_context=("No one had entered the vault.",),
        named_terms=(),
        coverage=("narration", "state", "tense_aspect"),
    ),
)


def build_semantic_context_v1_dataset() -> list[ContextualTestItem]:
    """Return the frozen 10-real + 14-synthetic contextual target set."""
    items = [*_real_items(), *SYNTHETIC_ITEMS]
    ids = [item.id for item in items]
    if len(items) != 24 or len(set(ids)) != len(ids):
        raise RuntimeError("Semantic Context V1 dataset must contain 24 unique items")
    if any(len(item.previous_context) > 3 or len(item.next_context) > 1 for item in items):
        raise RuntimeError("Semantic Context V1 local-context bounds were exceeded")
    return items
