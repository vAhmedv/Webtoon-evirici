Qwen Production Candidate Human Gate V1 — Blind Review Instructions
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
