"""Generator script for Qwen Production Candidate Human Gate V1.

Generates blind human-review pack from benchmark_results/real_chapter_translation_gate_v1/
without running any model inference or modifying source history.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

BENCHMARK_DIR = Path("benchmark_results/real_chapter_translation_gate_v1")
OUTPUT_DIR = Path("benchmark_results/qwen_production_candidate_gate_v1")
EXPECTED_SHA256 = "1fc9fc551797bff89a2ece71ae9cc2e513d282e520a26f647d9518b5eb807406"
RANDOM_SEED = 20260810


def calculate_risk_score(item: dict) -> tuple[int, list[str]]:
    """Calculates objective risk score and high-info reasons for an item."""
    score = 0
    reasons = []

    tg = item.get("translategemma", {})
    qw = item.get("qwen35", {})
    v3 = item.get("semantic_v3", {})

    tg_trans = tg.get("translation") or ""
    qw_trans = qw.get("translation") or ""

    # Feature 1: V3 rewrite used
    if v3.get("rewrite_used"):
        score += 10
        reasons.append("v3_rewrite_used")

    # Feature 2: Translations differ
    if tg_trans != qw_trans:
        score += 2
        reasons.append("translations_differ")

    # Feature 3: Both models require review
    if tg.get("requires_review") and qw.get("requires_review"):
        score += 5
        reasons.append("both_requires_review")
    elif tg.get("requires_review"):
        score += 3
        reasons.append("tg_requires_review")
    elif qw.get("requires_review"):
        score += 3
        reasons.append("qwen_requires_review")

    # Feature 4: Warnings
    tg_warns = tg.get("warnings", [])
    qw_warns = qw.get("warnings", [])
    if tg_warns:
        score += 2 * len(tg_warns)
        reasons.append(f"tg_warnings_{len(tg_warns)}")
    if qw_warns:
        score += 2 * len(qw_warns)
        reasons.append(f"qwen_warnings_{len(qw_warns)}")

    # Feature 5: Length ratio difference
    max_len = max(len(tg_trans), len(qw_trans), 1)
    len_diff = abs(len(tg_trans) - len(qw_trans)) / max_len
    if len_diff > 0.3:
        score += 2
        reasons.append(f"length_ratio_diff_{len_diff:.2f}")

    return score, reasons


def select_stratified_sample(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Selects 60 high-info and 20 control items stratified across 10 chapters."""
    by_chapter: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        key = (item["series"], item["chapter"])
        by_chapter.setdefault(key, []).append(item)

    high_info_items = []
    control_items = []

    # Sort chapters deterministically
    sorted_chapters = sorted(by_chapter.keys())

    for ch_key in sorted_chapters:
        ch_items = by_chapter[ch_key]

        # Partition into high-info candidates and control candidates
        ch_scored = []
        for it in ch_items:
            score, reasons = calculate_risk_score(it)
            ch_scored.append((score, reasons, it))

        # Sort high-info candidates by score descending, then by item id
        ch_scored.sort(key=lambda x: (-x[0], x[2]["id"]))

        # Select top 6 high-info items
        ch_high = ch_scored[:6]
        high_info_items.extend(ch_high)

        # Control candidates: clean items (no warnings, requires_review=False)
        remaining = ch_scored[6:]
        clean_controls = [
            (sc, reas, it) for sc, reas, it in remaining
            if not it["translategemma"].get("requires_review")
            and not it["qwen35"].get("requires_review")
            and not it["translategemma"].get("warnings")
            and not it["qwen35"].get("warnings")
        ]

        if len(clean_controls) < 2:
            # Fallback to least risky remaining items if clean items < 2
            clean_controls = sorted(remaining, key=lambda x: (x[0], x[2]["id"]))

        # Select 2 controls per chapter
        ch_ctrl = clean_controls[:2]
        control_items.extend(ch_ctrl)

    return high_info_items, control_items


def build_html_ui(blind_items_json_str: str) -> str:
    """Generates pure Vanilla JS single-file HTML review UI."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Webtoon Translation Blind Review Gate V1</title>
<style>
:root {{
    --bg-color: #0f172a;
    --card-bg: #1e293b;
    --card-border: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --winner-a: #ec4899;
    --winner-b: #8b5cf6;
    --tie-good: #22c55e;
    --tie-bad: #ef4444;
    --unscorable: #f59e0b;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }}

body {{
    background-color: var(--bg-color);
    color: var(--text-main);
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
}}

.header {{
    width: 100%;
    max-width: 1100px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}}

.header h1 {{ font-size: 1.4rem; font-weight: 700; color: var(--accent); }}
.header-info {{ display: flex; align-items: center; gap: 16px; }}

.badge {{
    background: #0284c7;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}}

.btn {{
    background: var(--accent);
    color: #0f172a;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
}}

.btn:hover {{ background: var(--accent-hover); color: white; }}
.btn-nav {{ background: #334155; color: white; }}
.btn-nav:hover {{ background: #475569; }}

.filter-bar {{
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    width: 100%;
    max-width: 1100px;
}}

.filter-btn {{
    background: #1e293b;
    border: 1px solid #334155;
    color: var(--text-muted);
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
}}

.filter-btn.active {{
    background: var(--accent);
    color: #0f172a;
    font-weight: 700;
}}

.main-container {{
    width: 100%;
    max-width: 1100px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}}

.item-meta {{
    display: flex;
    justify-content: space-between;
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--card-border);
}}

.context-box {{
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
}}

.context-line {{
    font-size: 0.9rem;
    color: var(--text-muted);
    font-style: italic;
    margin-bottom: 4px;
}}

.source-english {{
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 8px 0;
}}

.v3-source {{
    font-size: 0.95rem;
    color: #cbd5e1;
    background: #1e293b;
    padding: 6px 10px;
    border-radius: 4px;
    border-left: 3px solid var(--accent);
    margin-top: 6px;
}}

.translations-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
}}

.trans-card {{
    background: #0f172a;
    border: 2px solid var(--card-border);
    border-radius: 8px;
    padding: 18px;
    transition: all 0.2s ease;
}}

.trans-card.selected-a {{ border-color: var(--winner-a); }}
.trans-card.selected-b {{ border-color: var(--winner-b); }}

.trans-title {{
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-muted);
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.trans-text {{
    font-size: 1.1rem;
    line-height: 1.5;
    color: #f8fafc;
}}

.scoring-section {{
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 20px;
}}

.section-label {{
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 10px;
}}

.button-group {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 18px;
}}

.choice-btn {{
    background: #1e293b;
    border: 2px solid #334155;
    color: var(--text-main);
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    cursor: pointer;
    transition: all 0.15s ease;
}}

.choice-btn:hover {{ border-color: var(--accent); }}

.choice-btn.active[data-val="A"] {{ background: var(--winner-a); border-color: var(--winner-a); color: white; }}
.choice-btn.active[data-val="B"] {{ background: var(--winner-b); border-color: var(--winner-b); color: white; }}
.choice-btn.active[data-val="TIE_GOOD"] {{ background: var(--tie-good); border-color: var(--tie-good); color: white; }}
.choice-btn.active[data-val="TIE_BAD"] {{ background: var(--tie-bad); border-color: var(--tie-bad); color: white; }}
.choice-btn.active[data-val="UNSCORABLE_OCR"] {{ background: var(--unscorable); border-color: var(--unscorable); color: white; }}

.sev-btn.active {{ background: #0284c7; border-color: #0284c7; color: white; }}

.tags-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px;
    margin-bottom: 18px;
}}

.tag-check {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
    color: var(--text-muted);
    cursor: pointer;
    background: #1e293b;
    padding: 6px 10px;
    border-radius: 4px;
    border: 1px solid #334155;
}}

.tag-check input {{ cursor: pointer; }}

.notes-input {{
    width: 100%;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    color: var(--text-main);
    padding: 10px;
    font-size: 0.95rem;
    resize: vertical;
    min-height: 60px;
}}

.nav-bar {{
    display: flex;
    justify-content: space-between;
    margin-top: 20px;
}}

.hotkey-hint {{
    font-size: 0.8rem;
    color: var(--text-muted);
    text-align: center;
    margin-top: 12px;
}}
</style>
</head>
<body>

<div class="header">
    <h1>Webtoon Translation Blind Review V1</h1>
    <div class="header-info">
        <span class="badge" id="progress-badge">Reviewed: 0 / 80</span>
        <button class="btn" id="export-btn">Export Review JSON</button>
    </div>
</div>

<div class="filter-bar">
    <button class="filter-btn active" onclick="setFilter('all')">All Items (80)</button>
    <button class="filter-btn" onclick="setFilter('unreviewed')">Unreviewed (<span id="unreviewed-count">80</span>)</button>
    <button class="filter-btn" onclick="setFilter('reviewed')">Reviewed (<span id="reviewed-count">0</span>)</button>
</div>

<div class="main-container">
    <div class="item-meta">
        <div><strong id="meta-series">Series Name</strong> &bull; <span id="meta-chapter">Chapter X</span></div>
        <div>Review ID: <strong id="meta-id">R001</strong> (<span id="meta-index">1 / 80</span>)</div>
    </div>

    <div class="context-box">
        <div class="section-label">Context &amp; Source English</div>
        <div id="prev-context"></div>
        <div class="source-english" id="source-text">English source text...</div>
        <div id="next-context"></div>
        <div id="v3-selected" class="v3-source" style="display:none;"></div>
    </div>

    <div class="translations-grid">
        <div class="trans-card" id="card-a">
            <div class="trans-title">Translation A</div>
            <div class="trans-text" id="text-a">Translation A text...</div>
        </div>
        <div class="trans-card" id="card-b">
            <div class="trans-title">Translation B</div>
            <div class="trans-text" id="text-b">Translation B text...</div>
        </div>
    </div>

    <div class="scoring-section">
        <div class="section-label">1. Choose Winner (Keys: 1=A, 2=B, 3=Tie Good, 4=Tie Bad, 5=Unscorable)</div>
        <div class="button-group" id="winner-group">
            <button class="choice-btn" data-val="A" onclick="setWinner('A')">Translation A</button>
            <button class="choice-btn" data-val="B" onclick="setWinner('B')">Translation B</button>
            <button class="choice-btn" data-val="TIE_GOOD" onclick="setWinner('TIE_GOOD')">TIE GOOD</button>
            <button class="choice-btn" data-val="TIE_BAD" onclick="setWinner('TIE_BAD')">TIE BAD</button>
            <button class="choice-btn" data-val="UNSCORABLE_OCR" onclick="setWinner('UNSCORABLE_OCR')">UNSCORABLE OCR</button>
        </div>

        <div class="section-label">2. Worst Error Severity</div>
        <div class="button-group" id="severity-group">
            <button class="choice-btn sev-btn" data-val="NONE" onclick="setSeverity('NONE')">NONE</button>
            <button class="choice-btn sev-btn" data-val="MINOR" onclick="setSeverity('MINOR')">MINOR</button>
            <button class="choice-btn sev-btn" data-val="MAJOR" onclick="setSeverity('MAJOR')">MAJOR</button>
            <button class="choice-btn sev-btn" data-val="CRITICAL" onclick="setSeverity('CRITICAL')">CRITICAL</button>
        </div>

        <div class="section-label">3. Optional Category Tags</div>
        <div class="tags-grid" id="tags-container">
            <!-- Checkboxes generated dynamically -->
        </div>

        <div class="section-label">4. Optional Notes</div>
        <textarea class="notes-input" id="notes-input" placeholder="Add optional reviewer notes..." oninput="updateNotes()"></textarea>
    </div>

    <div class="nav-bar">
        <button class="btn btn-nav" onclick="prevItem()">&larr; Previous</button>
        <button class="btn btn-nav" onclick="nextItem()">Next &rarr;</button>
    </div>

    <div class="hotkey-hint">Shortcuts: [&larr;] Prev &bull; [&rarr;] Next &bull; [1] A &bull; [2] B &bull; [3] Tie Good &bull; [4] Tie Bad &bull; [5] Unscorable</div>
</div>

<script>
const BLIND_ITEMS = {blind_items_json_str};

const TAG_LIST = [
    "meaning", "naturalness", "terminology", "name", "omission", "addition",
    "hallucination", "pronoun", "tense_aspect", "state_action", "question_semantics",
    "register", "english_leak", "wrapper_chatbot", "grammar", "ocr_problem", "other"
];

let currentIndex = 0;
let currentFilter = "all";
let answers = {{}};

// Load from localStorage if available
function loadLocalAnswers() {{
    try {{
        const saved = localStorage.getItem("human_review_answers_v1");
        if (saved) {{
            answers = JSON.parse(saved);
        }}
    }} catch (e) {{
        console.error("Failed to load localStorage:", e);
    }}
}}

function saveLocalAnswers() {{
    try {{
        localStorage.setItem("human_review_answers_v1", JSON.stringify(answers));
    }} catch (e) {{
        console.error("Failed to save localStorage:", e);
    }}
}}

function init() {{
    loadLocalAnswers();
    renderTags();
    renderItem();
    updateCounts();

    document.getElementById("export-btn").addEventListener("click", exportJSON);
    document.addEventListener("keydown", handleKeydown);
}}

function renderTags() {{
    const container = document.getElementById("tags-container");
    container.innerHTML = TAG_LIST.map(tag => `
        <label class="tag-check">
            <input type="checkbox" value="${{tag}}" onchange="toggleTag('${{tag}}')" id="tag-${{tag}}">
            ${{tag}}
        </label>
    `).join("");
}}

function getFilteredIndices() {{
    return BLIND_ITEMS.map((item, idx) => idx).filter(idx => {{
        const id = BLIND_ITEMS[idx].review_id;
        const isReviewed = answers[id] && answers[id].winner;
        if (currentFilter === "unreviewed") return !isReviewed;
        if (currentFilter === "reviewed") return isReviewed;
        return true;
    }});
}}

function renderItem() {{
    const item = BLIND_ITEMS[currentIndex];
    const ans = answers[item.review_id] || {{ winner: null, severity: "NONE", tags: [], notes: "" }};

    document.getElementById("meta-series").textContent = item.series;
    document.getElementById("meta-chapter").textContent = item.chapter;
    document.getElementById("meta-id").textContent = item.review_id;
    document.getElementById("meta-index").textContent = `${{currentIndex + 1}} / ${{BLIND_ITEMS.length}}`;

    // Context
    const prevBox = document.getElementById("prev-context");
    prevBox.innerHTML = (item.previous_context || []).map(c => `<div class="context-line">&uarr; ${{escapeHtml(c)}}</div>`).join("");

    document.getElementById("source-text").textContent = item.original_accepted_english;

    const nextBox = document.getElementById("next-context");
    nextBox.innerHTML = (item.next_context || []).map(c => `<div class="context-line">&darr; ${{escapeHtml(c)}}</div>`).join("");

    const v3Box = document.getElementById("v3-selected");
    if (item.v3_selected_english && item.v3_selected_english !== item.original_accepted_english) {{
        v3Box.style.display = "block";
        v3Box.textContent = "V3 Clarified English: " + item.v3_selected_english;
    }} else {{
        v3Box.style.display = "none";
    }}

    // Translations
    document.getElementById("text-a").textContent = item.translation_a;
    document.getElementById("text-b").textContent = item.translation_b;

    // Highlight cards
    document.getElementById("card-a").className = "trans-card" + (ans.winner === "A" ? " selected-a" : "");
    document.getElementById("card-b").className = "trans-card" + (ans.winner === "B" ? " selected-b" : "");

    // Winner buttons
    document.querySelectorAll("#winner-group .choice-btn").forEach(btn => {{
        btn.classList.toggle("active", btn.dataset.val === ans.winner);
    }});

    // Severity buttons
    document.querySelectorAll("#severity-group .choice-btn").forEach(btn => {{
        btn.classList.toggle("active", btn.dataset.val === (ans.severity || "NONE"));
    }});

    // Tags
    TAG_LIST.forEach(tag => {{
        const chk = document.getElementById(`tag-${{tag}}`);
        chk.checked = (ans.tags || []).includes(tag);
    }});

    // Notes
    document.getElementById("notes-input").value = ans.notes || "";
}}

function setWinner(val) {{
    const id = BLIND_ITEMS[currentIndex].review_id;
    if (!answers[id]) answers[id] = {{ winner: null, severity: "NONE", tags: [], notes: "" }};
    answers[id].winner = val;
    saveLocalAnswers();
    renderItem();
    updateCounts();
}}

function setSeverity(val) {{
    const id = BLIND_ITEMS[currentIndex].review_id;
    if (!answers[id]) answers[id] = {{ winner: null, severity: "NONE", tags: [], notes: "" }};
    answers[id].severity = val;
    saveLocalAnswers();
    renderItem();
}}

function toggleTag(tag) {{
    const id = BLIND_ITEMS[currentIndex].review_id;
    if (!answers[id]) answers[id] = {{ winner: null, severity: "NONE", tags: [], notes: "" }};
    let tags = answers[id].tags || [];
    if (tags.includes(tag)) {{
        tags = tags.filter(t => t !== tag);
    }} else {{
        tags.push(tag);
    }}
    answers[id].tags = tags;
    saveLocalAnswers();
}}

function updateNotes() {{
    const id = BLIND_ITEMS[currentIndex].review_id;
    if (!answers[id]) answers[id] = {{ winner: null, severity: "NONE", tags: [], notes: "" }};
    answers[id].notes = document.getElementById("notes-input").value;
    saveLocalAnswers();
}}

function updateCounts() {{
    const total = BLIND_ITEMS.length;
    const reviewedCount = Object.values(answers).filter(a => a.winner).length;
    const unreviewedCount = total - reviewedCount;

    document.getElementById("progress-badge").textContent = `Reviewed: ${{reviewedCount}} / ${{total}}`;
    document.getElementById("reviewed-count").textContent = reviewedCount;
    document.getElementById("unreviewed-count").textContent = unreviewedCount;
}}

function setFilter(filter) {{
    currentFilter = filter;
    document.querySelectorAll(".filter-btn").forEach(btn => btn.classList.remove("active"));
    event.target.classList.add("active");
    const indices = getFilteredIndices();
    if (indices.length > 0 && !indices.includes(currentIndex)) {{
        currentIndex = indices[0];
    }}
    renderItem();
}}

function prevItem() {{
    const indices = getFilteredIndices();
    const pos = indices.indexOf(currentIndex);
    if (pos > 0) {{
        currentIndex = indices[pos - 1];
    }} else if (currentIndex > 0) {{
        currentIndex--;
    }}
    renderItem();
}}

function nextItem() {{
    const indices = getFilteredIndices();
    const pos = indices.indexOf(currentIndex);
    if (pos >= 0 && pos < indices.length - 1) {{
        currentIndex = indices[pos + 1];
    }} else if (currentIndex < BLIND_ITEMS.length - 1) {{
        currentIndex++;
    }}
    renderItem();
}}

function handleKeydown(e) {{
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft") prevItem();
    else if (e.key === "ArrowRight") nextItem();
    else if (e.key === "1") setWinner("A");
    else if (e.key === "2") setWinner("B");
    else if (e.key === "3") setWinner("TIE_GOOD");
    else if (e.key === "4") setWinner("TIE_BAD");
    else if (e.key === "5") setWinner("UNSCORABLE_OCR");
}}

function exportJSON() {{
    const output = {{
        metadata: {{
            total_items: BLIND_ITEMS.length,
            exported_at: new Date().toISOString()
        }},
        reviews: answers
    }};
    const blob = new Blob([JSON.stringify(output, null, 2)], {{ type: "application/json" }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "human_review_answers.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}}

function escapeHtml(str) {{
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}}

init();
</script>
</body>
</html>
"""


def build_readme() -> str:
    return """Qwen Production Candidate Human Gate V1 — Blind Review Instructions
======================================================================

1. Open 'human_review_blind.html' directly in your web browser (Chrome, Edge, Firefox, etc.).
2. Read the source English text, previous context, and optional V3 clarified English.
3. Compare 'Translation A' and 'Translation B' objectively.
4. Select the Winner:
   - [ A ]: Translation A is clearly preferable
   - [ B ]: Translation B is clearly preferable
   - [ TIE GOOD ]: Both translations are acceptable and approximately equal
   - [ TIE BAD ]: Both translations contain meaningful problems
   - [ UNSCORABLE OCR ]: English source or OCR is too broken to judge translation fairly
5. Select worst Error Severity (NONE, MINOR, MAJOR, CRITICAL) when applicable.
6. Check optional category tags (meaning, naturalness, terminology, etc.) if useful.
7. Add optional reviewer notes if desired.
8. Use Keyboard shortcuts ([Left/Right Arrows], [1], [2], [3], [4], [5]) or Previous/Next buttons to navigate through all 80 items.
9. Progress automatically saves in browser localStorage.
10. When all 80 items are reviewed, click 'Export Review JSON'.
11. Save the downloaded file as 'human_review_answers.json'.
12. Provide 'human_review_answers.json' back to the automated analyzer script:
    python scripts/analyze_production_candidate_human_review.py benchmark_results/qwen_production_candidate_gate_v1/human_review_answers.json

NOTE: Neither the HTML interface nor the blind item files disclose model identities (TranslateGemma vs Qwen).
"""


def main() -> None:
    print("=== Qwen Production Candidate Human Gate V1 Pack Generator ===")

    # 1. Verify existence of benchmark artifacts
    summary_file = BENCHMARK_DIR / "summary.json"
    comp_file = BENCHMARK_DIR / "comparison.json"
    if not summary_file.exists() or not comp_file.exists():
        raise FileNotFoundError(f"Benchmark dir {BENCHMARK_DIR} missing required summary.json or comparison.json")

    with open(summary_file, encoding="utf-8") as f:
        summary_data = json.load(f)

    actual_hash = summary_data.get("dataset_hash")
    print(f"Dataset SHA-256: {actual_hash}")
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"Dataset SHA-256 mismatch! Expected {EXPECTED_SHA256}, found {actual_hash}")

    with open(comp_file, encoding="utf-8") as f:
        items = json.load(f)

    print(f"Loaded {len(items)} benchmark items from comparison.json")

    # 2. Perform stratified sampling (60 high-info + 20 controls across 10 chapters)
    high_info_tuples, control_tuples = select_stratified_sample(items)

    print(f"Selected {len(high_info_tuples)} high-info items and {len(control_tuples)} control items.")

    # Combine into 80 review items
    combined_tuples = []
    # Interleave or concatenate (6 high-info + 2 controls per chapter)
    by_ch: dict[str, tuple[list, list]] = {}
    for sc, reas, it in high_info_tuples:
        ch_key = f"{it['series']}_{it['chapter']}"
        by_ch.setdefault(ch_key, ([], []))[0].append((sc, reas, it))
    for sc, reas, it in control_tuples:
        ch_key = f"{it['series']}_{it['chapter']}"
        by_ch.setdefault(ch_key, ([], []))[1].append((sc, reas, it))

    sample_items = []
    sample_manifest = []
    blind_review_items = []
    blind_answer_key = {}

    item_counter = 1
    tg_warn_count = 0
    qw_warn_count = 0
    both_warn_count = 0
    no_warn_count = 0
    v3_rewrite_count = 0
    series_counts: dict[str, int] = {}
    chapter_counts: dict[str, int] = {}

    # Deterministic randomization seed
    rng = random.Random(RANDOM_SEED)

    for ch_key in sorted(by_ch.keys()):
        h_list, c_list = by_ch[ch_key]

        chapter_items_raw = [("high_information", sc, reas, it) for sc, reas, it in h_list] + \
                            [("control", sc, reas, it) for sc, reas, it in c_list]

        for bucket, score, reas, it in chapter_items_raw:
            review_id = f"R{item_counter:03d}"
            item_counter += 1

            s_id = it["series"]
            ch_name = it["chapter"]

            series_counts[s_id] = series_counts.get(s_id, 0) + 1
            chapter_counts[ch_name] = chapter_counts.get(ch_name, 0) + 1

            v3 = it.get("semantic_v3", {})
            if v3.get("rewrite_used"):
                v3_rewrite_count += 1

            tg = it.get("translategemma", {})
            qw = it.get("qwen35", {})

            has_tg_warn = bool(tg.get("warnings") or tg.get("requires_review"))
            has_qw_warn = bool(qw.get("warnings") or qw.get("requires_review"))

            if has_tg_warn and has_qw_warn:
                both_warn_count += 1
            elif has_tg_warn:
                tg_warn_count += 1
            elif has_qw_warn:
                qw_warn_count += 1
            else:
                no_warn_count += 1

            # Deterministic A/B assignment
            assign_qwen_to_a = rng.choice([True, False])

            tg_text = tg.get("translation") or ""
            qw_text = qw.get("translation") or ""

            if assign_qwen_to_a:
                trans_a = qw_text
                trans_b = tg_text
                a_model = "qwen35"
                b_model = "translategemma"
            else:
                trans_a = tg_text
                trans_b = qw_text
                a_model = "translategemma"
                b_model = "qwen35"

            # 1. Manifest
            manifest_entry = {
                "review_id": review_id,
                "benchmark_id": it["id"],
                "series": s_id,
                "chapter": ch_name,
                "source": it["original_accepted_english"],
                "v3_selected_source": v3.get("selected_english") or it["original_accepted_english"],
                "sampling_bucket": bucket,
                "sampling_reason": reas,
            }
            sample_manifest.append(manifest_entry)

            # 2. Blind Item (NO MODEL NAMES)
            blind_entry = {
                "review_id": review_id,
                "series": s_id,
                "chapter": ch_name,
                "previous_context": it.get("previous_context", []),
                "original_accepted_english": it["original_accepted_english"],
                "next_context": it.get("next_context", []),
                "v3_selected_english": v3.get("selected_english", ""),
                "translation_a": trans_a,
                "translation_b": trans_b,
            }
            blind_review_items.append(blind_entry)

            # 3. Answer Key
            blind_answer_key[review_id] = {
                "review_id": review_id,
                "benchmark_id": it["id"],
                "translation_a_model": a_model,
                "translation_b_model": b_model,
            }

    # Prepare summary
    sampling_summary = {
        "random_seed": RANDOM_SEED,
        "total_review_items": len(blind_review_items),
        "high_information_count": len(high_info_tuples),
        "control_count": len(control_tuples),
        "items_per_series": series_counts,
        "items_per_chapter": chapter_counts,
        "v3_rewrite_count": v3_rewrite_count,
        "warning_statistics": {
            "tg_warnings_only": tg_warn_count,
            "qwen_warnings_only": qw_warn_count,
            "both_warnings": both_warn_count,
            "no_warnings": no_warn_count,
        },
    }

    # Write output files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sample_manifest, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "blind_review_items.json", "w", encoding="utf-8") as f:
        json.dump(blind_review_items, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "blind_answer_key.json", "w", encoding="utf-8") as f:
        json.dump(blind_answer_key, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "sampling_summary.json", "w", encoding="utf-8") as f:
        json.dump(sampling_summary, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "README_REVIEW.txt", "w", encoding="utf-8") as f:
        f.write(build_readme())

    # Build HTML UI
    blind_items_str = json.dumps(blind_review_items, ensure_ascii=False)
    html_content = build_html_ui(blind_items_str)
    with open(OUTPUT_DIR / "human_review_blind.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"All pack files generated in {OUTPUT_DIR}/")

    # Blinding verification check
    forbidden = ["translategemma", "qwen", "gemma"]
    for fn in ["blind_review_items.json", "human_review_blind.html"]:
        p = OUTPUT_DIR / fn
        content_lower = p.read_text(encoding="utf-8").lower()
        for forb in forbidden:
            if forb in content_lower:
                raise ValueError(f"BLINDNESS LEAK ALERT! Found forbidden string '{forb}' in {fn}!")

    print("BLINDNESS VERIFICATION PASSED 100%! Zero model names leaked into blind items or HTML UI.")


if __name__ == "__main__":
    main()
