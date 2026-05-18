# 03 — Trajectory Features (the core of v2)

Per shot, extract the per-frame ball track + rim bbox within the shot window, for **each** angle, then derive the features below. All positional features normalized by rim width (scale-invariant; resolution-independent — see note).

## Raw track (export this, per angle, per shot)
`[(frame_idx, t_sec, ball_x, ball_y, ball_conf, rim_x1,rim_y1,rim_x2,rim_y2), …]`
Persist raw — every feature is derived from it and we will iterate on features.

## Per-angle geometric features
Let rim center `C`, rim width `W`, rim plane `y = rim_y2` (bottom) / `rim_y1` (top).

1. **entry_angle** — angle of descent of the ball path approaching the rim.
2. **min_dist_to_rim_center** — closest approach, in rim-widths.
3. **crossed_top_plane** / **crossed_bottom_plane** — did the path cross `rim_y1` / `rim_y2` while horizontally within ±k·W.
4. **through_depth** — how far below the rim plane the ball reached (rim-widths); separates swish from front-rim graze.
5. **post_min_upward_disp** — max upward displacement *after* closest approach → the **rim-in-and-out signal** (the #1 precision error). Magnitude + duration.
6. **oscillation** — count/energy of vertical direction changes near the rim (rattle).
7. **horizontal_drift_at_plane** — |ball_x − C_x| / W at the plane crossing (in front of vs through rim).
8. **approach/exit speed** (px/frame, normalized) and **speed_ratio** (slow rattle vs clean fast swish).
9. **time_in_zone**, **n_tracked_points**, **occlusion_ratio** (fraction of window with no/low-conf ball), **mean_ball_conf**.
10. **size_ratio_at_plane** — ball/rim size at crossing (foreground/pass rejection).

## Cross-angle / fusion features
11. **near_present**, **far_present**, per-angle `n_tracked_points`, `occlusion_ratio`.
12. **near_far_agree_raw** — do the two angles' geometric verdicts agree (legacy signal, as a feature only).
13. **time_offset_residual** — matched Δt after offset fit (sync quality).
14. **best_angle_hint** — which angle had lower occlusion / more tracked points this shot.
15. **shot_kind** if reliably inferable (FT vs field) from court position — FT has distinct trajectory; otherwise omit (don't leak GT class).

## Hard rule (anti-leakage)
Features may use **only** detector/track outputs — never the GT label, never `plays.classification`, never the legacy made/miss outcome as an input. (Legacy outcome may be logged for comparison but must NOT be a model feature, else it just relearns the broken rule.)

## Resolution note
Pipeline delivers 1080p (uniform downscale from 4K, no crop). Normalizing every positional feature by **rim width in the same frame** makes features invariant to 4K/1080p and to per-camera zoom — required because camera lens/zoom is not standardized (see v1 camera findings).

## Visual review (Trap 1, done right)
Frame-by-frame review is used **only** on error cases (misclassified + the borderline in-and-outs) to (a) confirm a feature actually captures the failure and (b) discover missing features — NOT to hand-pick thresholds. The decision boundary is always fit from GT across games. Render annotated review clips on AWS (recipe in `04`), never locally.
