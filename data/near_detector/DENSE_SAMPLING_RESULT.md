# Dense Sampling (stride 1) — Spotter Recall Recovered (2026-06-14)

Triggered by re-reading the old NR fusion logic: the old probe showed the ball
is detected AT the rim in 100% of SHOTS (per-shot), so our spotter-recall gap
(~0.82 at stride 2-3) is SAMPLING-limited, not sensor-limited. Tested by running
the end-to-end at stride 1 (every frame) with v0.1's exact config (tight zone,
no reach filter) on the 4 frozen games -- only the sampling density changed.

## Result (4 frozen games, no GT windows)
| | spot recall | make/miss | false/game | found-AND-correct |
|---|---|---|---|---|
| v0.1 stride 3 | 0.818 | 0.930 | 21.8 | 0.761 |
| **stride 1 dense** | **0.918** | 0.909 | 30.8 | **0.834** |

Per game recall: 6d601c99 .906->.969, ee8745f1 .657->.829, 0fa23810 .933->1.000,
c2a354fe .775->.875.

## Read
- **Spotter recall +10pt (0.818->0.918)** -- the ball WAS there per shot; we just
  needed to sample every frame to catch the fast/clean passages we were missing.
  CONFIRMS the old probe + the offline sweep (stride 3 0.833 -> stride 1 0.946).
- **make/miss -2pt (0.930->0.909)** -- small dip because the newly-found shots are
  the harder/faster ones; not a regression in the classifier itself.
- **Overall found-AND-correctly-classified: 0.761 -> 0.834 (+7.3pt)** -- the real
  net win.
- **Cost: false events 21.8 -> 30.8/game** -- more frames = more false triggers.
  Controllable with the rim-reach filter (REACH_FRAC) at a small recall cost;
  it's a precision/recall knob, not a hard problem.

## Revision to the "sensor limit" finding
- Per-FRAME ball recall IS blur/sensor-limited (detector v2 confirmed, ~0.80-0.82).
- But per-SHOT SPOTTING recall is SAMPLING-limited and recoverable: 0.82 -> 0.92
  via stride 1. The make/miss DECISION quality (~0.91-0.93) is the 30fps ceiling;
  120fps lifts that further. SPOT recall now has a software answer.

## Disposition
- Adopt stride 1 (or stride 2 as a speed/recall compromise) for the spotter.
- Manage false events with the reach filter per product tolerance.
- This is the concrete payoff of re-reading the old NR fusion logic.
