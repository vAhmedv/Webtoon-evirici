"""Production Translation Quality Stress Benchmark (Part A: 180 Synthetic Webtoon Items).

Executes the production translation pipeline (Qwen3.5-9B GGUF llama-server + CUDA)
across 180 synthetic dialogue items with token-aware batching, rolling context, and
an isolated CandidateStore.

Saves output artifacts to benchmark_results/translation_stress_v1/:
- results.json
- results.txt
- summary.json
"""
import json
import os
import time
from pathlib import Path

from core.translation.profile_discovery import CandidateStore, record_validated_term_observations
from core.translation.series_profile import SeriesProfile
from providers.translation import (
    TranslationInput,
    TranslationItem,
    get_translation_provider,
)

SYNTHETIC_ITEMS = [
    (1, "Relax, kid."),
    (2, "I said I'm fine."),
    (3, "You don't look fine."),
    (4, "Give me five minutes."),
    (5, "Fine. But we're leaving after that."),
    (6, "Don't make me regret this."),
    (7, "You worry too much."),
    (8, "And you don't worry enough."),
    (9, "Excuse me, Captain Gao."),
    (10, "The council is waiting for you."),
    (11, "Tell them I'll be there shortly."),
    (12, "With all due respect, Captain, we don't have time to wait."),
    (13, "Would you mind lowering your voice?"),
    (14, "I heard you the first time."),
    (15, "Don't get the wrong idea."),
    (16, "I wasn't asking for permission."),
    (17, "My name is Luo Tian."),
    (18, "Have you seen Hu San?"),
    (19, "I gave the map to Gao Yuan."),
    (20, "This sword belongs to Luo Tian."),
    (21, "We received a message from Mira Vale."),
    (22, "Tell Kael Arden I'm looking for him."),
    (23, "I haven't spoken to Young Master Yu since yesterday."),
    (24, "Take this directly to Young Master Yu."),
    (25, "We have to enter the Secret Realm before sunset."),
    (26, "Once we're inside the Secret Realm, stay close to me."),
    (27, "Your Mana Core is unstable."),
    (28, "Protect your Mana Core and don't force it any further."),
    (29, "This Spirit Stone isn't worth much."),
    (30, "I paid three Spirit Stones for that map."),
    (31, "Only Inner Disciples are allowed beyond this gate."),
    (32, "The Inner Disciples' quarters are on the eastern side."),
    (33, "The Guild Master ordered us to retreat."),
    (34, "I need to speak with the Guild Master personally."),
    (35, "You signed a Soul Contract, didn't you?"),
    (36, "Breaking a Soul Contract has consequences."),
    (37, "He's not an Ability User."),
    (38, "Then how did he survive that attack?"),
    (39, "The entrance to Blackwind Ravine is two miles north."),
    (40, "The Celestial Forge hasn't been used in nearly a century."),
    (41, "What about the chest?"),
    (42, "Leave it. It's trapped."),
    (43, "But it could contain something valuable."),
    (44, "I said leave it."),
    (45, "He accused you of cheating."),
    (46, "Leave it. It's not worth arguing over."),
    (47, "That's easy for you to say."),
    (48, "Maybe. But fighting him won't change anything."),
    (49, "They're going to charge us at dawn."),
    (50, "Then we'll hit them before they can form ranks."),
    (51, "How much did the merchant charge you?"),
    (52, "More than this junk was worth."),
    (53, "Our party has six members."),
    (54, "Seven, if Luo Tian comes with us."),
    (55, "The victory party starts after sunset."),
    (56, "Assuming we're still alive by then."),
    (57, "Great. The bridge collapsed."),
    (58, "Just what we needed."),
    (59, "Nice going, genius."),
    (60, "Oh, shut up. You were the one who said it was safe."),
    (61, "You've got some nerve blaming me."),
    (62, "Save the lecture for later."),
    (63, "This is not the time."),
    (64, "For once, we agree."),
    (65, "What the hell are you doing?"),
    (66, "Trying to keep us alive!"),
    (67, "Then move your damn hand!"),
    (68, "Shit. We're surrounded."),
    (69, "Don't push your luck."),
    (70, "I wasn't planning to."),
    (71, "You've said that before."),
    (72, "And I'm still alive, aren't I?"),
    (73, "Wait—"),
    (74, "Did you hear that?"),
    (75, "No..."),
    (76, "Not again."),
    (77, "Because if he finds us here..."),
    (78, "Don't."),
    (79, "Don't what?"),
    (80, "Don't finish that sentence."),
    (81, "Where's Master Lin?"),
    (82, "They said they'd meet us here."),
    (83, "Maybe they changed their mind."),
    (84, "Or maybe something happened."),
    (85, "We don't know that."),
    (86, "Exactly. So stop guessing."),
    (87, "Did Ren take the artifact?"),
    (88, "He didn't say he took it."),
    (89, "He said he saw who did."),
    (90, "That's not the same thing."),
    (91, "Everyone who enters the chamber is tested."),
    (92, "Not everyone who enters comes back."),
    (93, "Only two of the five seals have been broken."),
    (94, "We need all five before the gate will open."),
    (95, "At least three guards are still inside."),
    (96, "We have no more than three minutes."),
    (97, "Unless the gate closes before midnight, we still have time."),
    (98, "If I had known what was waiting inside, I wouldn't have come."),
    (99, "Even if he apologizes, I won't trust him again."),
    (100, "If you leave now, I won't stop you."),
    (101, "But if you stay, you follow my orders."),
    (102, "Understood?"),
    (103, "He almost killed me."),
    (104, "But he didn't."),
    (105, "I barely got away."),
    (106, "That's exactly why you shouldn't go back alone."),
    (107, "Three days."),
    (108, "That's all I need."),
    (109, "You said two days yesterday."),
    (110, "Things changed."),
    (111, "Funny how they always seem to change when you make a promise."),
    (112, "Are you calling me a liar?"),
    (113, "One in ten recruits makes it this far."),
    (114, "The third floor is twice as dangerous as the second."),
    (115, "We lost twenty percent of our supplies."),
    (116, "This blade costs 2,500 Spirit Stones."),
    (117, "That's more than I earn in a year."),
    (118, "And it's still the cheapest one here."),
    (119, "I don't need you to save me."),
    (120, "Good, because I wasn't going to."),
    (121, "...You could at least pretend."),
    (122, "Would that make you feel better?"),
    (123, "No."),
    (124, "Maybe a little."),
    (125, "I hate you."),
    (126, "No, you don't."),
    (127, "Don't tell me how I feel."),
    (128, "Then stop looking at me like that."),
    (129, "Like what?"),
    (130, "Exactly."),
    (131, "You came back."),
    (132, "I said I would."),
    (133, "You also said you wouldn't do anything reckless."),
    (134, "I lied about that part."),
    (135, "Idiot."),
    (136, "Yeah. Probably."),
    (137, "The master told me to wait here."),
    (138, "Which master?"),
    (139, "The Guild Master."),
    (140, "Oh. You could've said that first."),
    (141, "The core of the problem isn't your Mana Core."),
    (142, "It's that you refuse to listen."),
    (143, "Those are two completely different things."),
    (144, "Are they?"),
    (145, "The seal on the eastern door is weakening."),
    (146, "Don't break it until the others arrive."),
    (147, "This letter bears the royal seal."),
    (148, "So it's genuine?"),
    (149, "After the first alarm sounded, the guards sealed every exit except the northern passage."),
    (150, "By the time we reached it, someone had already destroyed the bridge."),
    (151, "That means whoever did this knew exactly where we were going."),
    (152, "Or they wanted us to think they did."),
    (153, "Luo Tian had expected the ruins to be empty, but the footprints in the dust were fresh."),
    (154, "Someone had entered the chamber recently, and whoever it was had taken great care not to disturb anything else."),
    (155, "That bothered him more than a broken lock would have."),
    (156, "A thief would have rushed. This person had been looking for something specific."),
    (157, "The rain had stopped before dawn, yet the stone path was still slick beneath their feet."),
    (158, "No one spoke as they climbed toward Blackwind Ravine."),
    (159, "The silence wasn't peaceful; it was the kind that made every distant sound seem closer than it really was."),
    (160, "When Gao Yuan finally raised his hand, everyone stopped at once."),
    (161, '"You knew?"'),
    (162, '"I suspected."'),
    (163, '"That\'s not the same thing."'),
    (164, '"It is when everyone else is dead."'),
    (165, "I never said he was innocent."),
    (166, "You said there wasn't enough evidence."),
    (167, "There still isn't."),
    (168, "Then why are you carrying his sword?"),
    (169, "Maybe he was right."),
    (170, "About what?"),
    (171, "About me."),
    (172, "That's not an answer."),
    (173, "I can do this."),
    (174, "I know."),
    (175, "Then why are you stopping me?"),
    (176, "Because being able to do something doesn't mean you should."),
    (177, "If the first team fails, the second team retreats."),
    (178, "If the second team can't retreat, destroy the gate."),
    (179, "And if we're still inside?"),
    (180, "Then you'd better make sure the first team doesn't fail."),
]


def run_benchmark():
    print("=== STARTING PRODUCTION TRANSLATION STRESS BENCHMARK V1 ===")

    output_dir = Path("benchmark_results/translation_stress_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = SeriesProfile(
        series_id="synthetic_translation_benchmark_v1",
        known_names={
            "LUO TIAN": "Luo Tian",
            "HU SAN": "Hu San",
            "GAO YUAN": "Gao Yuan",
            "MIRA VALE": "Mira Vale",
            "KAEL ARDEN": "Kael Arden",
            "YOUNG MASTER YU": "Genç Efendi Yu",
        },
        glossary={
            "SECRET REALM": "Gizli Diyar",
            "MANA CORE": "Mana Çekirdeği",
            "SPIRIT STONE": "Ruh Taşı",
            "INNER DISCIPLE": "İç Mürit",
            "GUILD MASTER": "Lonca Lideri",
            "SOUL CONTRACT": "Ruh Sözleşmesi",
            "ABILITY USER": "Yetenek Kullanıcısı",
            "BLACKWIND RAVINE": "Karayel Vadisi",
            "CELESTIAL FORGE": "Göksel Ocak",
            # Intentionally Unrelated Terms (MUST NOT be injected for unrelated items)
            "DRAGON VEIN": "Ejder Damarı",
            "MOON PALACE": "Ay Sarayı",
            "CRIMSON SPEAR": "Kızıl Mızrak",
        },
    )

    store = CandidateStore(series_id="synthetic_translation_benchmark_v1")

    items = [
        TranslationItem(region_id=rid, source=src, reading_order=rid)
        for rid, src in SYNTHETIC_ITEMS
    ]

    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="stress_v1_ch1",
    )

    provider = get_translation_provider(backend="gguf")

    t_load_0 = time.perf_counter()
    provider.load()
    load_time = time.perf_counter() - t_load_0

    t_wall_0 = time.perf_counter()
    out = provider.translate(inp)
    wall_time = time.perf_counter() - t_wall_0

    # Record validated observations into benchmark CandidateStore (without mutating production profiles)
    for item_res in out.results:
        record_validated_term_observations(
            candidate_store=store,
            chapter_id="stress_v1_ch1",
            region_id=item_res.region_id,
            source_text=item_res.source,
            translated_text=item_res.translation or "",
            raw_term_usages=item_res.term_usages,
            term_id_map=item_res.term_id_map,
            fidelity_flags=item_res.fidelity_flags,
            requires_review=item_res.requires_review,
        )

    provider.unload()

    m = provider.metrics

    # Check structural validation
    returned_ids = [r.region_id for r in out.results]
    expected_ids = list(range(1, 181))
    missing_ids = [i for i in expected_ids if i not in returned_ids]
    duplicate_ids = [i for i in returned_ids if returned_ids.count(i) > 1]
    requires_review_count = sum(1 for r in out.results if r.requires_review)
    fidelity_flagged_count = sum(1 for r in out.results if r.fidelity_flags)

    # Check unrelated glossary leak
    unrelated_terms = {"DRAGON VEIN", "MOON PALACE", "CRIMSON SPEAR"}
    unrelated_leaked = False
    for res in out.results:
        # Check if term_id_map contains any unrelated term
        for src_t in res.term_id_map.values():
            if src_t.strip().upper() in unrelated_terms:
                unrelated_leaked = True

    # Termbase counts
    cand_counts = {"discovered": 0, "provisional": 0, "ready_for_review": 0, "approved": 0}
    for c in store.candidates.values():
        st = c.status
        if st in cand_counts:
            cand_counts[st] += 1
        elif st == "confirmed":
            cand_counts["approved"] += 1

    # Format results.json
    results_json_data = []

    # Map context IDs for each item (previous 1-3 items in reading order)
    for idx, item_res in enumerate(out.results):
        rid = item_res.region_id
        # Context IDs passed to this item
        ctx_ids = [i for i in range(max(1, rid - 3), rid)]
        
        # Extract approved terms used in source
        app_terms_used = {}
        for k, v in profile.known_names.items():
            if k in item_res.source.upper():
                app_terms_used[k] = v
        for k, v in profile.glossary.items():
            if k in item_res.source.upper() and k not in unrelated_terms:
                app_terms_used[k] = v

        results_json_data.append({
            "id": rid,
            "source": item_res.source,
            "translation": item_res.translation,
            "context_ids": ctx_ids,
            "approved_terms": app_terms_used,
            "validation_warnings": item_res.validation_warnings,
            "fidelity_flags": item_res.fidelity_flags,
            "requires_review": item_res.requires_review,
            "term_usages": item_res.term_usages,
        })

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json_data, f, ensure_ascii=False, indent=2)

    # Format results.txt (human-readable)
    results_txt_lines = []
    items_by_id = {item.region_id: item.source for item in items}

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
        results_txt_lines.append("")

        results_txt_lines.append(f"WARNINGS: {item_data['validation_warnings']}")
        results_txt_lines.append(f"FIDELITY FLAGS: {item_data['fidelity_flags']}")
        results_txt_lines.append(f"REQUIRES REVIEW: {item_data['requires_review']}")
        results_txt_lines.append("-" * 50 + "\n")

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results_txt_lines))

    # Format summary.json
    summary_data = {
        "execution": {
            "synthetic_items": f"{len(out.results)}/{len(SYNTHETIC_ITEMS)}",
            "flores_control": "NOT RUN — dataset unavailable locally",
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
            "json_failures": sum(1 for r in out.results if "json_parse_failure" in r.validation_warnings),
            "requires_review_count": requires_review_count,
            "fidelity_flagged_count": fidelity_flagged_count,
            "unrelated_glossary_leaked": unrelated_leaked,
        },
        "termbase": {
            "discovered": cand_counts["discovered"],
            "provisional": cand_counts["provisional"],
            "ready_for_review": cand_counts["ready_for_review"],
            "approved_automatically": cand_counts["approved"],
        },
        "paths": {
            "results_json": str(output_dir / "results.json"),
            "results_txt": str(output_dir / "results.txt"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n=== BENCHMARK COMPLETED SUCCESSFULLY ===")
    print(f"Total items: {len(out.results)}/180")
    print(f"Generation calls: {m.generation_call_count}, Retries: {m.retries}")
    print(f"Generated tokens: {m.generated_token_count}, Gen time: {m.generation_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s)")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    run_benchmark()
