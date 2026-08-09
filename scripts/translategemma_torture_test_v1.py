"""TranslateGemma Production Hard Translation Torture Test v1 (96 Items).

Executes the current production TranslateGemma translation provider over 96 difficult
webtoon/manhwa dialogue and terminology test items.

Saves output artifacts to benchmark_results/translategemma_torture_v1/:
- results.json
- results.txt
- summary.json
- term_behavior.json
"""
import json
import time
from pathlib import Path

from core.translation.profile_discovery import CandidateStore
from core.translation.series_profile import SeriesProfile
from providers.translation import (
    TranslationInput,
    TranslationItem,
    get_translation_provider,
)

TORTURE_TEST_96_ITEMS = [
    (1, "I'm not an Ability User."),
    (2, "Then how did you survive that attack?"),
    (3, "I never said I didn't have an ability."),
    (4, "I said I wasn't an Ability User."),
    (5, "What's your ability?"),
    (6, "It's called Void Step."),
    (7, "Void Step?"),
    (8, "Don't ask. I didn't name it."),
    (9, "His ability lets him see through walls."),
    (10, "Mine only works when I'm asleep."),
    (11, "That's the worst ability I've ever heard of."),
    (12, "You say that now."),
    (13, "Activate Void Step."),
    (14, "He used Void Step twice in less than a minute."),
    (15, "If he uses Void Step again, his Mana Core will collapse."),
    (16, "Then stop him before he gets the chance."),
    (17, "Crimson Lotus is not an attack."),
    (18, "Then what is it?"),
    (19, "A technique for redirecting incoming force."),
    (20, "That's a very complicated way of saying \"don't get hit.\""),
    (21, "I learned Second Heart when I was twelve."),
    (22, "Second Heart keeps me alive after a fatal injury."),
    (23, "It doesn't make you immortal."),
    (24, "I know. Believe me, I know."),
    (25, "PASSIVE SKILL ACQUIRED: MANA SENSE"),
    (26, "Mana Sense allows the user to detect nearby Mana Cores."),
    (27, "Your Mana Sense is unusually weak."),
    (28, "That's because I turned it off."),
    (29, "TITLE ACQUIRED: NIGHT WALKER"),
    (30, "CLASS ADVANCEMENT AVAILABLE"),
    (31, "UNIQUE TRAIT: MIRROR SKIN"),
    (32, "CONDITION FAILED: INSUFFICIENT MANA"),
    (33, "Skill activation failed."),
    (34, "Ability cooldown: 18 seconds."),
    (35, "Your passive skill is still active."),
    (36, "Then why can't I feel anything?"),
    (37, "Only Inner Disciples may enter the Secret Realm."),
    (38, "Luo Tian stopped being an Inner Disciple three years ago."),
    (39, "Then why does he still have an Inner Disciple's token?"),
    (40, "Because nobody was brave enough to take it from him."),
    (41, "The Guild Master wants to see you."),
    (42, "Tell the Guild Master I'm busy."),
    (43, "You don't tell the Guild Master you're busy."),
    (44, "I just did."),
    (45, "Young Master Yu entered Blackwind Ravine alone."),
    (46, "Young Master Yu's guards followed him an hour later."),
    (47, "If Young Master Yu is still alive, he'll be furious."),
    (48, "That's a surprisingly big \"if.\""),
    (49, "Your Mana Core isn't damaged."),
    (50, "The core problem is that you keep forcing it past its limit."),
    (51, "Those are not the same kind of core."),
    (52, "I noticed."),
    (53, "The seal on the coffin is broken."),
    (54, "Seal the western gate before sunset."),
    (55, "The letter bears the emperor's seal."),
    (56, "Three different seals, three different problems."),
    (57, "Leave the Spirit Stones here."),
    (58, "Leave the room."),
    (59, "Leave him alone."),
    (60, "Leave it to me."),
    (61, "They charged us twenty Spirit Stones for one bottle."),
    (62, "The Spirit Beast charged before we could draw our weapons."),
    (63, "Who charged the account?"),
    (64, "Mei Ren did."),
    (65, "Our party has four Ability Users and one healer."),
    (66, "The victory party begins after sunset."),
    (67, "Don't party until the mission is actually over."),
    (68, "You're no fun."),
    (69, "Master Ren taught me everything I know."),
    (70, "My master key doesn't fit this lock."),
    (71, "You called him \"master.\""),
    (72, "I was talking about the key."),
    (73, "Not everyone who enters the Secret Realm returns."),
    (74, "Almost nobody survives without a guide."),
    (75, "Only three of the twelve teams came back."),
    (76, "At least one of them is lying."),
    (77, "I didn't say Kael Arden killed him."),
    (78, "I said Kael Arden knows who did."),
    (79, "That's not the same accusation."),
    (80, "Try explaining that to the guards."),
    (81, "He barely survived."),
    (82, "He almost survived."),
    (83, "Those two sentences do not mean the same thing."),
    (84, "Apparently someone forgot that."),
    (85, "Brilliant. You woke the Spirit Beast."),
    (86, "I didn't know it was sleeping!"),
    (87, "What did you think the snoring was?"),
    (88, "Ancient machinery?"),
    (89, "Oh, perfect. The bridge is gone."),
    (90, "Technically, half the bridge is still here."),
    (91, "That's very comforting."),
    (92, "Happy to help."),
    (93, "The corridor had been empty when Serin Vale entered it."),
    (94, "When she looked back, a wet footprint rested in the middle of the stone floor."),
    (95, "There was no second footprint, no open door, and nowhere for anyone to hide."),
    (96, "Then the footprint slowly turned toward her."),
]

UNAPPROVED_TERMS_TO_TRACK = [
    "Void Step",
    "Crimson Lotus",
    "Second Heart",
    "Mana Sense",
    "Night Walker",
    "Mirror Skin",
]

APPROVED_TERMS_TO_TRACK = [
    "ABILITY USER",
    "MANA CORE",
    "SECRET REALM",
    "INNER DISCIPLE",
    "GUILD MASTER",
    "SPIRIT STONE",
    "SPIRIT BEAST",
    "BLACKWIND RAVINE",
]


def run_torture_test():
    print("=== STARTING TRANSLATEGEMMA TORTURE TEST V1 (96 ITEMS) ===")

    output_dir = Path("benchmark_results/translategemma_torture_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(
        series_id="translategemma_torture_test_v1",
        known_names={
            "LUO TIAN": "Luo Tian",
            "MEI REN": "Mei Ren",
            "KAEL ARDEN": "Kael Arden",
            "SERIN VALE": "Serin Vale",
            "YOUNG MASTER YU": "Genç Efendi Yu",
            "MASTER REN": "Usta Ren",
        },
        glossary={
            "ABILITY USER": "Yetenek Kullanıcısı",
            "MANA CORE": "Mana Çekirdeği",
            "SECRET REALM": "Gizli Diyar",
            "SPIRIT STONE": "Ruh Taşı",
            "SOUL CONTRACT": "Ruh Sözleşmesi",
            "INNER DISCIPLE": "İç Mürit",
            "GUILD MASTER": "Lonca Lideri",
            "SPIRIT BEAST": "Ruh Canavarı",
            "CELESTIAL FORGE": "Göksel Ocak",
            "BLACKWIND RAVINE": "Karayel Vadisi",
        },
    )

    store = CandidateStore(series_id="translategemma_torture_test_v1")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in TORTURE_TEST_96_ITEMS
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="torture_v1_ch1",
    )

    # Use default production provider (TranslateGemma GGUF)
    provider = get_translation_provider()

    t_load_0 = time.perf_counter()
    provider.load()
    load_time = time.perf_counter() - t_load_0

    t_wall_0 = time.perf_counter()
    out = provider.translate(inp)
    wall_time = time.perf_counter() - t_wall_0

    provider.unload()

    m = provider.metrics

    returned_ids = [r.region_id for r in out.results]
    expected_ids = list(range(1, 97))
    missing_ids = [i for i in expected_ids if i not in returned_ids]
    duplicate_ids = [i for i in returned_ids if returned_ids.count(i) > 1]
    empty_outputs = [r.region_id for r in out.results if not r.translation]

    results_json_data = []
    items_by_id = {item.region_id: item.source for item in items}

    for item_res in out.results:
        rid = item_res.region_id
        ctx_ids = [i for i in range(max(1, rid - 3), rid)]

        app_terms_used = {}
        for k, v in profile.known_names.items():
            if k in item_res.source.upper():
                app_terms_used[k] = v
        for k, v in profile.glossary.items():
            if k in item_res.source.upper():
                app_terms_used[k] = v

        results_json_data.append({
            "id": rid,
            "source": item_res.source,
            "translation": item_res.translation,
            "context_ids": ctx_ids,
            "approved_terms": app_terms_used,
            "validation_warnings": item_res.validation_warnings,
            "requires_review": item_res.requires_review,
        })

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json_data, f, ensure_ascii=False, indent=2)

    results_txt_lines = []
    for item_data in results_json_data:
        rid = item_data["id"]
        results_txt_lines.append(f"[{rid:03d}]\n")

        ctx_ids = item_data["context_ids"]
        if ctx_ids:
            results_txt_lines.append("CONTEXT:")
            for cid in ctx_ids:
                if cid in items_by_id:
                    results_txt_lines.append(f"{cid:03d} | {items_by_id[cid]}")
            results_txt_lines.append("")

        app_terms = item_data["approved_terms"]
        if app_terms:
            results_txt_lines.append("APPROVED TERMS:")
            for k, v in app_terms.items():
                results_txt_lines.append(f"{k} => {v}")
            results_txt_lines.append("")

        results_txt_lines.append("SOURCE:")
        results_txt_lines.append(item_data["source"])
        results_txt_lines.append("")

        results_txt_lines.append("TURKISH:")
        results_txt_lines.append(item_data["translation"] or "<FAILED>")
        results_txt_lines.append("\n" + "-" * 50 + "\n")

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results_txt_lines))

    # Track Terminology Behavior
    term_behavior = {
        "unapproved_named_terms": {},
        "approved_terms": {},
    }

    results_map = {item["id"]: item["translation"] or "" for item in results_json_data}

    for u_term in UNAPPROVED_TERMS_TO_TRACK:
        observations = []
        for rid, src in TORTURE_TEST_96_ITEMS:
            if u_term.lower() in src.lower():
                observations.append({
                    "item_id": rid,
                    "source": src,
                    "translation": results_map.get(rid, ""),
                })
        term_behavior["unapproved_named_terms"][u_term] = observations

    for a_term in APPROVED_TERMS_TO_TRACK:
        observations = []
        for rid, src in TORTURE_TEST_96_ITEMS:
            if a_term in src.upper():
                observations.append({
                    "item_id": rid,
                    "source": src,
                    "translation": results_map.get(rid, ""),
                })
        term_behavior["approved_terms"][a_term] = observations

    with open(output_dir / "term_behavior.json", "w", encoding="utf-8") as f:
        json.dump(term_behavior, f, ensure_ascii=False, indent=2)

    summary_data = {
        "execution": {
            "synthetic_items": f"{len(out.results)}/{len(TORTURE_TEST_96_ITEMS)}",
            "generation_calls": m.generation_call_count,
            "retries": m.retries,
            "input_tokens": m.input_token_count,
            "generated_tokens": m.generated_token_count,
            "generation_seconds": round(m.generation_seconds, 2),
            "wall_time_seconds": round(wall_time, 2),
            "average_tok_per_sec": round(m.tokens_per_sec, 2),
            "load_time_seconds": round(load_time, 2),
            "backend": provider.name,
            "model": m.translation_model,
        },
        "structural_validation": {
            "missing_ids": missing_ids,
            "duplicate_ids": duplicate_ids,
            "empty_outputs": empty_outputs,
            "requires_review_count": sum(1 for r in out.results if r.requires_review),
        },
        "paths": {
            "results_json": str(output_dir / "results.json"),
            "results_txt": str(output_dir / "results.txt"),
            "summary_json": str(output_dir / "summary.json"),
            "term_behavior_json": str(output_dir / "term_behavior.json"),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n=== TRANSLATEGEMMA TORTURE TEST COMPLETED ===")
    print(f"Total items: {len(out.results)}/96")
    print(f"Generation calls: {m.generation_call_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    run_torture_test()
