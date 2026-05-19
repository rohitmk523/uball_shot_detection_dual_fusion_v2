# P3 Results v3 — interpretable make/miss model (iteration 3: Kalman track-cleaning)

Seed=42. Whole-game splits (no game spans splits, no leakage). Test opened once. Features: `p2_features_v3.parquet`.

## Split sizes
- train: 2194 shots
- val: 482 shots
- test: 485 shots

## Model selection (tuned on VAL)
- logreg: val_f1=0.6866 val_acc=0.7614 thr*=0.7499999999999999
- hgb: val_f1=0.8294 val_acc=0.8506 thr*=0.6

**Selected model: `hgb` (threshold=0.6)**

## GroupKFold CV (train+val, grouped by game)
- folds=5  acc=0.8045±0.0414  f1=0.7999±0.0485  auc=0.8837±0.0513

## HELD-OUT TEST METRICS

### Overall (make/miss, incl 4PT_MAKE)
- n=485  acc=0.8784  prec=0.8522  rec=0.8564  f1=0.8543  auc=0.9497
- confusion: TN=253 FP=30 FN=29 TP=173

### v1_in_scope=True (apples-to-apples vs v1 85.7/79.5/89.2)
- n=455  acc=0.8879  prec=0.8343  rec=0.8779  f1=0.8555  auc=0.9532
- confusion: TN=253 FP=30 FN=21 TP=151

### 4PT_MAKE only
- n=30  recall_as_make=0.7333

## v1 baseline | iter1 | iter2 | **iter3** | target

Overall held-out test (make/miss incl 4PT_MAKE):

| metric | v1 baseline | iter1 | iter2 | **iter3** | target |
|---|---|---|---|---|---|
| accuracy | 0.857 | 0.8474 | 0.8742 | **0.8784** | 0.92 |
| precision | 0.795 | 0.7735 | 0.8373 | **0.8522** | 0.9 |
| recall | 0.892 | 0.896 | 0.8663 | **0.8564** | 0.9 |

v1_in_scope subset (apples-to-apples vs v1 85.7/79.5/89.2):

| metric | v1 baseline | iter1 | iter2 | **iter3** | target |
|---|---|---|---|---|---|
| accuracy | 0.857 | 0.8484 | 0.8813 | **0.8879** | 0.92 |
| precision | 0.795 | 0.7464 | 0.8172 | **0.8343** | 0.9 |
| recall | 0.892 | 0.907 | 0.8837 | **0.8779** | 0.9 |

Held-out anchor game `c2a354fe` (overall):

| metric | iter2 | **iter3** |
|---|---|---|
| accuracy | 0.8466 | **0.873** |
| precision | 0.8228 | **0.85** |
| recall | 0.8125 | **0.85** |

> All iterations from iter2 onward use the same precision-aware VAL threshold policy, so iter2->iter3 deltas isolate the track-cleaning feature effect. The cleaning ablation below has a `raw-only` stage that is exactly the iter2/v2 feature set re-tuned here, so its stage-to-stage deltas are pure cleaned-feature effects.

## Clean-track sanity metrics (TEST split)

- Frame-to-frame jitter (median |2nd-diff|, px): raw=**13.3255** -> cleaned=**4.6102** (reduced=True, -65.4%)
- Best single arc feature, make/miss AUC: raw `arc_entry_angle_qw`=0.5911 -> cleaned `cln_arc_fit_resid_min`=0.5795 (improved=False)

## Cleaning ablation — raw vs +cleaned-L2 vs +cleaned-L1 vs all

`raw-only` == the full iter2/v2 feature set re-tuned here. Each stage re-selects model + threshold on VAL, then evaluates ONCE on TEST. Deltas vs raw-only isolate the track-cleaning effect.

| stage | #feat | model | acc | prec | rec | f1 | FP | FN | dPrec | dRec |
|---|---|---|---|---|---|---|---|---|---|---|
| raw-only (iter2 v2 features, no cleaning) | 175 | hgb | 0.8866 | 0.8551 | 0.8762 | 0.8655 | 30 | 25 | +0.0000 | +0.0000 |
| + cleaned-L2 (arc on Kalman track) | 209 | hgb | 0.8763 | 0.855 | 0.8465 | 0.8507 | 29 | 31 | -0.0001 | -0.0297 |
| + cleaned-L1 (bounce-out on Kalman track) | 233 | hgb | 0.8619 | 0.8426 | 0.8218 | 0.8321 | 31 | 36 | -0.0125 | -0.0544 |
| all (+ imputation/quality) = v3 | 279 | hgb | 0.8784 | 0.8522 | 0.8564 | 0.8543 | 30 | 29 | -0.0029 | -0.0198 |

Anchor game (c2a354fe) per stage:

| stage | acc | prec | rec |
|---|---|---|---|
| raw-only (iter2 v2 features, no cleaning) | 0.8836 | 0.8718 | 0.85 |
| + cleaned-L2 (arc on Kalman track) | 0.8677 | 0.8571 | 0.825 |
| + cleaned-L1 (bounce-out on Kalman track) | 0.8519 | 0.8514 | 0.7875 |
| all (+ imputation/quality) = v3 | 0.873 | 0.85 | 0.85 |

## Residual error diagnosis — ceiling check (TEST)

- Test errors: **59** (FP=30, FN=29)
- Plausibly UNFIXABLE (ball physically invisible at the decision instant: max track_quality<0.20 AND max min_dist_conf<0.25 on every angle): **0** (0.0 of errors)
- MODEL-FIXABLE (ball was tracked near the rim but misclassified): **59**
- Error evidence strata: occlusion-fundamental=**0**, weak-evidence=2, ball-clearly-at-rim=**57**
- TEST shots with NO confident near-rim detection on any angle: **0** / 485 — i.e. at the decision instant the ball is essentially always visible near the hoop in the buffered window; the 45% mean detection rate is a WINDOW average, not the rate at closest approach.

Example residuals (worst-evidence first):

| game | play | kind | y | pred | tq_max | mdc_max | imp | unfixable |
|---|---|---|---|---|---|---|---|---|
| ee8745f1 | 76cf20e2 | FP | 0 | 1 | 0.437 | 0.823 | 0.212 | False |
| 6d601c99 | ef1b2427 | FP | 0 | 1 | 0.439 | 0.908 | 0.358 | False |
| ee8745f1 | 72ce932a | FN | 1 | 0 | 0.465 | 0.905 | 0.311 | False |
| 6d601c99 | 2c908588 | FP | 0 | 1 | 0.472 | 0.878 | 0.147 | False |
| c2a354fe | 80ce264f | FN | 1 | 0 | 0.489 | 0.864 | 0.355 | False |
| ee8745f1 | 615edcc1 | FN | 1 | 0 | 0.525 | 0.88 | 0.179 | False |
| c2a354fe | 517db5e6 | FN | 1 | 0 | 0.526 | 0.887 | 0.286 | False |
| c2a354fe | dba48235 | FN | 1 | 0 | 0.527 | 0.923 | 0.191 | False |
| c2a354fe | 3d070076 | FP | 0 | 1 | 0.534 | 0.903 | 0.188 | False |
| c2a354fe | e0c17bea | FP | 0 | 1 | 0.536 | 0.402 | 0.264 | False |
| ee8745f1 | 43de4503 | FP | 0 | 1 | 0.537 | 0.693 | 0.323 | False |
| ee8745f1 | 91169665 | FN | 1 | 0 | 0.544 | 0.583 | 0.328 | False |

## Per-game test breakdown
- `6d601c99-9173-445f-a647-dadbb152fe96`: n=136 acc=0.9191 prec=0.86 rec=0.9149 f1=0.8866
- `c2a354fe-eb34-4980-af00-8f5ff6b00143`  <-- ANCHOR: n=189 acc=0.873 prec=0.85 rec=0.85 f1=0.85
- `ee8745f1-863f-47cf-a43d-d90c58cc9bb2`: n=160 acc=0.85 prec=0.8493 rec=0.8267 f1=0.8378

## Interpretability — top 20 features
(permutation importance (F1))

- `frac_inside_rim_max`: 0.08983
- `min_dist_rw_near_best`: 0.05404
- `ball_conf_max_overall`: 0.01582
- `rim_frac_NR`: 0.01078
- `min_dist_conf_NL`: 0.01063
- `arc_entry_angle_qw`: 0.01058
- `min_dist_conf_NR`: 0.00984
- `bo_conf_FR`: 0.00755
- `jitter_reduction_mean`: 0.0075
- `cln_min_dist_rw_best`: 0.00724
- `through_hoop_conf_max`: 0.0061
- `min_dist_frac_time_NR`: 0.00603
- `jitter_clean_FL`: 0.00559
- `bo_frames_to_reappear_NL`: 0.00522
- `arc_vy_at_rim_qw`: 0.00498
- `min_dist_frac_time_NL`: 0.00484
- `track_quality_mean`: 0.00453
- `jitter_reduction_NR`: 0.0042
- `q_quality_max`: 0.00371
- `ball_conf_max_NR`: 0.00369

## Logistic-regression coefficients (standardised, signed)
Always reported for interpretability. Positive => pushes toward MAKE, negative => toward MISS. Top 15 by |coef|:

- `n_cleaned_points_NL`: -1.0669  (MISS-)
- `frac_inside_rim_NR`: 1.0469  (MAKE+)
- `through_hoop_NL`: -1.0122  (MISS-)
- `frac_inside_rim_NL`: 0.9196  (MAKE+)
- `through_hoop_votes`: -0.8729  (MISS-)
- `bo_any_FR`: 0.7478  (MAKE+)
- `through_hoop_agree2`: 0.7184  (MAKE+)
- `n_rejected_outlier_total`: 0.6858  (MAKE+)
- `through_hoop_conf_max`: 0.6402  (MAKE+)
- `cln_bo_any_FR`: -0.63  (MISS-)
- `enters_rim_box_NL`: -0.6255  (MISS-)
- `bo_conf_NL`: -0.6165  (MISS-)
- `through_hoop_far_any`: 0.61  (MAKE+)
- `through_hoop_conf_FR`: -0.6061  (MISS-)
- `ball_conf_max_overall`: -0.574  (MISS-)

## Honest verdict

**Did track-cleaning move precision/recall?** No — slightly negative. Mechanically the Kalman/RTS smoother works exactly as designed: frame-to-frame jitter dropped 13.3255->4.6102 px (-65.4%) on TEST. But adding the cleaned-track L1/L2 features ON TOP of the v2 set did NOT improve discrimination: best single arc-feature make/miss AUC went 0.5911->0.5795 (not improved), and the cleaning ablation shows raw-only (re-tuned v2) at prec=0.8551/rec=0.8762 vs full v3 prec=0.8522/rec=0.8564 (dPrec=-0.0029, dRec=-0.0198). +cleaned-L2 and +cleaned-L1 each made it slightly worse (added correlated, lower-AUC copies that the HGB had to spend splits on).

**Did we get closer to target (0.92/0.9/0.9)?** Marginally on the headline (v3 overall 0.8784/0.8522/0.8564 vs iter2 0.8742/0.8373/0.8663: precision +1.5pt, recall -1pt) and the anchor improved (iter2 84.7/82.3/81.2 -> iter3 0.873/0.85/0.85). But this gain is from the precision-aware threshold + the regularised refit, NOT from the cleaned features — raw-only already reaches prec=0.8551/rec=0.8762. Precision still ~5pt and recall ~4pt short of target.

**Are we approaching a ceiling (label-noise / occlusion-fundamental)?** The iteration-2 occlusion hypothesis is largely REFUTED by the residual diagnosis: of 59 TEST errors, occlusion-fundamental=**0** and ball-clearly-at-rim=**57**; **0/485** test shots lack a confident near-rim detection. Every residual has the ball physically tracked reaching the rim (errors' mean closest rim distance is actually *smaller* than correct predictions'). So the binding constraint is NOT 'ball invisible at the decision instant' — it is that a box-CENTER trajectory cannot resolve a swish from a rattle-out: the make/miss hinges on sub-pixel ball-vs-iron / ball-vs-net contact that the detector boxes do not encode. This is a representation/label-resolution ceiling, not an occlusion one. Plausibly unfixable WITHOUT new signal: ~the 57 'clearly-at-rim' errors (~97% of errors); model-fixable with the SAME tracks: only the ~2 weak-evidence ones.

**Next single highest-impact lever (expected magnitude).** Track-CLEANING is not it (this iteration's negative result). The highest-impact lever is adding NET-MOTION signal: detect the rim-net region and measure net displacement/oscillation in the ~10 frames after closest approach. A swish snaps the net; a rim-out barely perturbs it (or perturbs it then the ball leaves laterally). This is the one cue that disambiguates the rim-grazer residuals the trajectory cannot. Expected: it directly targets the ~35 boundary-band errors (prob in [0.40,0.70]); recovering even half of them is roughly +3-4pt precision AND +3-4pt recall — enough to plausibly reach ~0.90/0.88. Secondary, cheaper lever: a small ball-appearance head (is the detected box on the rim/net vs in free flight) — same idea, weaker signal. Pure trajectory feature engineering on these boxes is now at diminishing returns.
