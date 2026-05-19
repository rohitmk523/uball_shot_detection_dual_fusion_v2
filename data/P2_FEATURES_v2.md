# P2 Feature Dictionary (iteration 2 / v2)

One row per shot `(game_id, play_id)`. Features describe the ball's geometry relative to the (near-static) rim over the buffered GT window, computed per camera angle (FL/FR/NL/NR) and fused. All coordinates are pixels in 1920x1080; (x,y)=box top-left. Distances are normalised by the robust median rim width so they are scale/zoom invariant. Iteration 2 adds three interpretable lever groups on top of the v1 features (kept intact so each lever can be ablated): `bo_*` rim-out/bounce-out recovery (L1), `arc_*` parametric trajectory-arc fit (L2), and `q_*` per-angle quality-gated fusion (L3).

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

## Per-angle features (suffix `_FL/_FR/_NL/_NR`)

- `n_frames_*`: window length for this angle (context).
- `ball_frac_*`: fraction of frames the ball was detected — detection quality / imputation awareness.
- `ball_conf_mean_*`: mean ball detector confidence.
- `ball_conf_max_*`: peak ball confidence (clear sighting).
- `rim_frac_*`: fraction of frames the rim was detected.
- `rim_detected_*`: 1 if a robust rim reference exists for this angle.
- `min_dist_rw_*`: closest ball-center to rim-center distance over the window, in rim-widths. Low => ball reached the hoop (necessary for a make).
- `min_dist_frac_time_*`: when (0..1 through window) closest approach occurred.
- `enters_rim_box_*`: 1 if the ball center was ever inside the rim bounding box.
- `frac_inside_rim_*`: fraction of ball frames with center in rim box.
- `through_hoop_*`: 1 if a clean above->below rim-plane crossing occurred within ~1 rim-width horizontally — the core make cue.
- `through_hoop_count_*`: number of such crossing frames.
- `through_hoop_conf_*`: mean ball confidence on crossing frames.
- `vy_sign_change_near_rim_*`: 1 if the ball's vertical velocity flipped sign while near the rim — a rim bounce (miss cue).
- `min_dist_conf_*`: ball confidence at the closest-approach frame.
- `approach_drop_*`: how much the rim distance fell from window start to the closest approach (rim-widths).
- `post_min_rebound_*`: how much distance increased after the closest approach — bounce-out (miss cue).
- `angle_usable_*`: 1 if rim ref exists AND ball seen at least once.
- `bo_any_*`: L1: 1 if ball reappeared BELOW+OUTSIDE the rim after closest approach — rim-in-and-out MISS cue.
- `bo_conf_*`: L1: ball confidence at that reappearance frame.
- `bo_frames_to_reappear_*`: L1: frames from closest approach to the below+outside reappearance.
- `bo_outward_disp_*`: L1: post-rim outward displacement / rim width.
- `bo_then_rises_*`: L1: 1 if the ball then rises/leaves (clean bounce-out, strongest MISS evidence).
- `arc_ok_*`: L2: 1 if a parabola could be fit (>=5 ball points).
- `arc_entry_angle_deg_*`: L2: descent angle into the rim plane (deg); steeper => more make-like.
- `arc_vy_at_rim_*`: L2: vertical velocity at the rim plane (px/frame / rim height; +ve = downward).
- `arc_curvature_*`: L2: parabola curvature a (dimensionless, a*rw).
- `arc_fit_resid_*`: L2: RMS parabola residual / rim width — track noisiness / fit quality.
- `arc_apex_dx_rw_*`: L2: apex x minus rim center, in rim-widths.
- `arc_apex_above_rim_rh_*`: L2: apex height above the rim, rim-heights.
- `q_quality_*`: L3: angle quality = ball_frac * rim_frac (0..1); far cams that lost the ball off-frame are zeroed.

## Fused cross-angle features

- `n_usable_angles`: # angles with rim ref + a ball detection.
- `min_dist_rw_best`: min over usable angles of min_dist_rw (strongest make-distance evidence).
- `min_dist_rw_mean`: mean over usable angles.
- `min_dist_rw_confw`: ball-conf-weighted mean of per-angle min distance (robust fusion).
- `through_hoop_any`: any angle saw a clean through-hoop.
- `through_hoop_votes`: # usable angles with a through-hoop event.
- `through_hoop_count_max`: max crossing-frame count across angles.
- `through_hoop_conf_max`: best through-hoop confidence.
- `enters_rim_votes`: # usable angles where ball entered rim box.
- `frac_inside_rim_max`: max frac_inside_rim across angles.
- `post_min_rebound_max`: max bounce-out rebound across angles (miss cue).
- `vy_sign_change_votes`: # angles with a near-rim vy sign flip.
- `min_dist_rw_near_best`: best min-dist among near cams (NL/NR).
- `min_dist_rw_far_best`: best min-dist among far cams (FL/FR).
- `through_hoop_near_any`: through-hoop seen by a near cam.
- `through_hoop_far_any`: through-hoop seen by a far cam.
- `ball_frac_mean`: mean ball-detection fraction across angles.
- `ball_conf_max_overall`: global peak ball confidence.
- `any_angle_usable`: 1 if at least one angle is usable.
- `far_cam_left_frame_*`: L3: 1 if a far cam's rebound blew up (ball left frame) => its quality zeroed.
- `q_quality_max`: L3: best per-angle quality.
- `q_quality_sum_usable`: L3: total quality mass over usable cams.
- `min_dist_rw_qw`: L3: quality-weighted closest approach (replaces naive max/mean fusion).
- `arc_entry_angle_qw`: L3: quality-weighted arc entry angle.
- `arc_vy_at_rim_qw`: L3: quality-weighted vy at the rim plane.
- `close_agree_votes`: L3: # usable angles agreeing the ball came near the rim.
- `agree_make_like`: L3: 1 only if >=2 angles agree on a low distance (single-cam FP guard).
- `through_hoop_agree2`: L3: 1 if >=2 angles saw a through-hoop.
- `bo_any_fused`: L1: any usable angle saw a bounce-out (MISS).
- `bo_votes`: L1: # usable angles with a bounce-out.
- `bo_conf_fused`: L1: quality-weighted bounce-out confidence.
- `bo_outward_disp_max`: L1: max post-rim outward displacement.
- `bo_then_rises_any`: L1: any angle saw the ball leave after.
- `bo_clean_votes`: L1: # angles with reappear-below+outside AND then-leaves (sharpest rim-out MISS cue).
- `bo_clean_any`: L1: 1 if any clean bounce-out (strong MISS).
- `arc_ok_any`: L2: any angle produced a parabola fit.
- `arc_ok_votes`: L2: # angles with a parabola fit.
- `arc_fit_resid_min`: L2: best (lowest) arc fit residual.
- `arc_entry_angle_max`: L2: steepest entry angle across angles.
- `arc_apex_above_rim_rh_max`: L2: max apex-above-rim height.
- `arc_apex_dx_rw_min`: L2: closest apex-to-rim horizontal offset.
- `arc_vy_at_rim_max`: L2: max downward vy at the rim plane.

Total feature columns: **167**.

### Sanity: through-hoop vs label
- Shots with a through-hoop on >=1 angle: make rate = 0.5167 (n=2690)
- Shots with NO through-hoop: make rate = 0.2314
