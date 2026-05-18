# 05 — Validation & Success Criteria

## The metric
The **fused made/miss confusion matrix vs. human `plays` GT**, computed on **held-out TEST games only** (games never seen in feature design or fitting). Report, per game and pooled:
- Detection recall (CV produced an event for the GT shot)
- Made/miss accuracy (on detected) = (TP+TN)/matched
- "Made" **precision** = TP/(TP+FP)
- "Make" **recall** = TP/(TP+FN)
- Full confusion (TP/FN/FP/TN), and by shot type (2/3/4pt, FT)

## Baselines to beat (v16, c2a354fe, fused, vs `plays`)
Detection **92.6%** · accuracy **85.7%** · precision **79.5%** · recall **89.2%**.
A change **ships only if, on held-out games, precision AND recall both ≥ baseline** (no trading one for the other — the v1 `fix/shot-geometry-AB` failure mode). Detection must not regress.

## Honest target & ceiling (state plainly; do not promise 100%)
- The made/miss decision cannot exceed **human label self-consistency** (~a few % ambiguity on rim in-and-outs) nor resolve shots that are physically occluded in both angles.
- **Target band:** made/miss accuracy **≥ 92%** and precision **≥ 90%** with recall **≥ 90%** on held-out games (a meaningful, defensible jump from 85.7/79.5/89.2), with detection held ≥ 92%.
- "100%" is not a credible scientific target and must not be quoted to stakeholders. The deliverable is "converged to the practical ceiling, validated cross-game," with the gap to 100% explained (label noise + occlusion physics, quantified by reviewing the residual errors).

## Anti-overfitting protocol
- Whole games assigned to TRAIN/VAL/TEST; never split a game.
- Feature design & threshold selection use TRAIN+VAL only. TEST is opened **once per candidate model**, not iterated against.
- Report VAL→TEST gap each iteration; large gap ⇒ overfit ⇒ reject.
- Keep model interpretable; sanity-check top features against the visual error review (they must be physically sensible, e.g. "post-min upward displacement" for in-and-out).

## Per-iteration record (committed to repo, `results/`)
For every iteration: dataset hash, feature set, model + params, TRAIN/VAL/TEST game lists, full confusion per split, ship/no-ship decision, AWS $ spent. This is the audit trail and the client-evidence trail.
