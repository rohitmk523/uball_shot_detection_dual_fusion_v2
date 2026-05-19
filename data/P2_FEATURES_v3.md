# P2 Feature Dictionary (iteration 3 / v3)

Superset of `p2_features_v2.parquet`: every v2 column is kept verbatim (see `P2_FEATURES_v2.md` for those) and v3 ADDS the trajectory levers L1 (bounce-out) and L2 (arc-fit) recomputed on the **Kalman-cleaned** ball track (`pipeline/track_clean.py`) plus per-angle imputation / track-quality signals. The raw-track L1/L2 (`bo_*`, `arc_*`) are UNCHANGED so the model can use both and the ablation can isolate the cleaning effect. v1/v2 parquet are never modified.

## Track-cleaning pipeline (per play_id x angle)

1. **Robust outlier rejection** — drop isolated detections whose two-sided frame step exceeds `2.5 * rim_width` (detector teleports onto heads/logos).
2. **Constant-acceleration Kalman filter + RTS smoother** on (x,y); state = (x,vx,ax,y,vy,ay); gravity-aware process noise (vertical accel std 4.0 > horizontal 1.5).
3. **Bounded gap interpolation** — expose the smoother estimate ONLY inside occlusion gaps <=12 frames bracketed by confident (conf>=0.30) detections on BOTH sides; longer / unbounded gaps stay missing and are flagged (never hallucinate through a full occlusion).

## Coverage

- **n_shots**: 3161
- **n_makes**: 1499
- **n_misses**: 1662
- **n_v1_in_scope**: 3023
- **n_4pt_make**: 138
- **shots_with_through_hoop_ge1_angle**: 2690
- **frac_with_through_hoop**: 0.851
- **mean_usable_angles**: 3.846
- **shots_0_usable_angles**: 0
- **by_split**: {'train': 2194, 'val': 482, 'test': 485}
- **through_hoop_make_rate**: 0.5167
- **no_through_hoop_make_rate**: 0.2314

## v3-added columns

- `cln_arc_*`: L2 arc-fit recomputed on the cleaned track (entry angle, vy@rim, curvature, fit residual, apex offset/height). De-jittered => sharper.
- `cln_bo_*`: L1 bounce-out recomputed on the cleaned track (reappearance, outward displacement, then-rises).
- `cln_min_dist_rw`: closest cleaned ball-rim distance (rim-widths).
- `cln_through_hoop`: clean above->below rim-plane crossing on the cleaned track.
- `cln_frac_inside_rim`: fraction of cleaned points in rim box.
- `imp_frac`: fraction of usable cleaned points that were imputed across a bounded gap (higher => weaker).
- `n_imputed`: # bridged frames.
- `n_rejected_outlier`: # raw detections killed as teleports.
- `track_quality`: per-angle cleaned-track reliability [0,1] = (0.6*coverage + 0.4*mean_conf) * (1 - 0.5*imputed_frac).
- `jitter_raw / jitter_clean`: median |2nd-diff| of the y-center before / after cleaning.
- `jitter_reduction`: jitter_raw - jitter_clean (>0 => de-jittered; the cleaning sanity metric).
- `*_fused / *_max / *_votes (cln_)`: cross-angle aggregates of the cleaned levers.
- `track_quality_mean/max, imp_frac_mean/min_usable, n_rejected_outlier_total, jitter_reduction_mean`: shot-level imputation/quality so the model can globally discount unreliable shots.

Total feature columns: **279** (v3-added: **104**).
