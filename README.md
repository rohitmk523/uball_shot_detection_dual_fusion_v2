# uball_shot_detection_dual_fusion_v2

A clean restart for the made/missed decision: replace the brittle hand-coded near logic **and** the hand-coded fusion disagreement resolver with a **dual-angle, trajectory-feature, ground-truth-fit, interpretable** made/miss model — trained and evaluated **entirely on AWS**, validated cross-game against human annotation.

This repo is **spec + plan first**. Implementation happens in a fresh session driven by `docs/`.

## Why v2 (what we learned in v1)

- The far-angle **retrain** lifted *detection* 68% → 93% (V1→v16, vs human annotation, 189-shot game). Detection is largely solved.
- The remaining error is **made/missed judgement**, and it is a **logic** problem, not a model problem: on the near angle, the misjudged shots were detected just as confidently (0.83 vs 0.88) and with 100% ball-in-rim overlap as the correct ones — the detector saw them fine; the decision rule failed.
- The error is **bidirectional** and the legacy fix attempt proved it: one-sided threshold loosening (branch `fix/shot-geometry-AB`) did **not** recover false-negatives and **regressed precision −6.5 pp**. Threshold-tuning a flawed rule cannot win.
- Conclusion: the lever is **better features (full ball trajectory, both angles) + a decision boundary fit from ground truth**, not bigger models and not hand-tuned thresholds.

## Ground truth

Human annotators (`plays` table, uball.ai Supabase `mhbrsftxvxxtfgbajrlc`) — every shot labeled made/missed. This is the only ground truth used. No operator-scoreboard data.

## Read order (`docs/`)

1. `00_CONTEXT.md` — what v1 proved, the numbers, the physics ceiling, why this approach.
2. `01_ARCHITECTURE.md` — the dual-angle trajectory → interpretable GT-fit decision design.
3. `02_DATA_AND_GROUND_TRUTH.md` — `plays` schema, game selection, S3 layout, train/val/test split.
4. `03_FEATURES.md` — exact per-angle trajectory feature list + fusion features.
5. `04_AWS_RUNTIME.md` — everything runs on AWS (GPU spot EC2 recipe, .env, cost guardrails). **No laptop compute.**
6. `05_VALIDATION.md` — the success metric, baselines to beat, honest target & ceiling, ship criteria.
7. `06_ROADMAP_AND_AUTONOMY.md` — staged plan + the autonomous-loop protocol, guardrails, stop conditions.

## Non-negotiables

- All training/inference on **AWS**, never the laptop.
- Validate on **held-out games**; ship only if **precision AND recall both improve**.
- Model stays **interpretable** (auditable for client claims).
- Secrets only in `.env` (gitignored). Claude never writes real credentials.
