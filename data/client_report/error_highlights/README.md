# Error Highlights Reel

**`model_errors_highlight.mp4`** — 53 clips back-to-back, ~8 min total. Each clip:
- **Top bar:** shot #, game, GT class, TRUTH vs MODEL call + probability + error type (FP/FN)
- **Bottom bar (yellow):** plain-language *why* the model got it wrong, derived from the per-shot features (swish-misread, possible mislabel, intrinsic rim-grazer, etc.)
- Order: all False Positives (model→MAKE / truth=MISS) sorted by descending confidence, then False Negatives (model→MISS / truth=MAKE) sorted by ascending confidence — so the most confident model errors play first.

**The file itself is not tracked in git** (110 MB binary) — gitignored. Regenerate with `pipeline/build_highlights.py` (see source in `data/client_report/CLIENT_REPORT.md` section on highlights).
