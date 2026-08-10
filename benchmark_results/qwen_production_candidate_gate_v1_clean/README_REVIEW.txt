Clean Webtoon Translation Blind Review V1 — Review Instructions
====================================================================

1. Open 'human_review_blind.html' directly in your web browser.
2. Read the clean source English text, previous/next context, and optional V3 clarified English.
3. Compare 'Translation A' and 'Translation B' objectively.
4. Select Winner:
   - [ A ]: Translation A is clearly preferable
   - [ B ]: Translation B is clearly preferable
   - [ TIE GOOD ]: Both translations are acceptable and approximately equal
   - [ TIE BAD ]: Both translations contain meaningful problems
   - [ UNSCORABLE OCR ]: English source or OCR is too broken to judge translation fairly
5. Select worst Error Severity (NONE, MINOR, MAJOR, CRITICAL).
6. Check optional category tags if useful.
7. Add optional reviewer notes.
8. Use Keyboard shortcuts ([Left/Right Arrows], [1], [2], [3], [4], [5]) or Previous/Next.
9. Progress automatically saves in browser localStorage.
10. Click 'Export Review JSON' when finished.
11. Save as 'human_review_answers.json'.
12. Run analyzer script:
    python scripts/analyze_clean_production_candidate_human_review.py benchmark_results/qwen_production_candidate_gate_v1_clean/human_review_answers.json
