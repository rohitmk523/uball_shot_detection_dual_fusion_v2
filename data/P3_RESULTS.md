# P3 Results — interpretable make/miss model

Seed=42. Whole-game splits (no game spans splits, no leakage). Test opened once.

## Split sizes
- train: 2194 shots
- val: 482 shots
- test: 485 shots

## Model selection (tuned on VAL)
- logreg: val_f1=0.8176 val_acc=0.8195 thr*=0.42999999999999994
- hgb: val_f1=0.8603 val_acc=0.8672 thr*=0.48999999999999994

**Selected model: `hgb` (threshold=0.48999999999999994)**

## GroupKFold CV (train+val, grouped by game)
- folds=5  acc=0.8±0.0418  f1=0.7965±0.048  auc=0.8779±0.0496

## HELD-OUT TEST METRICS

### Overall (make/miss, incl 4PT_MAKE)
- n=485  acc=0.8474  prec=0.7735  rec=0.896  f1=0.8303  auc=0.9296
- confusion: TN=230 FP=53 FN=21 TP=181

### v1_in_scope=True (apples-to-apples vs v1 85.7/79.5/89.2)
- n=455  acc=0.8484  prec=0.7464  rec=0.907  f1=0.8189  auc=0.935
- confusion: TN=230 FP=53 FN=16 TP=156

### 4PT_MAKE only
- n=30  recall_as_make=0.8333

### v1 baseline comparison
| metric | v1 baseline | v2 (v1_in_scope) | target |
|---|---|---|---|
| accuracy | 0.857 | 0.8484 | >=0.92 |
| precision | 0.795 | 0.7464 | >=0.90 |
| recall | 0.892 | 0.907 | >=0.90 |

## Per-game test breakdown
- `6d601c99-9173-445f-a647-dadbb152fe96`: n=136 acc=0.8824 prec=0.7541 rec=0.9787 f1=0.8519
- `c2a354fe-eb34-4980-af00-8f5ff6b00143`  <-- ANCHOR: n=189 acc=0.8571 prec=0.8046 rec=0.875 f1=0.8383
- `ee8745f1-863f-47cf-a43d-d90c58cc9bb2`: n=160 acc=0.8063 prec=0.7558 rec=0.8667 f1=0.8075

## Interpretability — top 20 features
(permutation importance (F1))

- `frac_inside_rim_max`: 0.14081
- `min_dist_rw_near_best`: 0.06978
- `through_hoop_conf_max`: 0.02304
- `min_dist_conf_NL`: 0.01954
- `min_dist_conf_NR`: 0.01698
- `rim_frac_NR`: 0.01523
- `ball_conf_max_overall`: 0.01484
- `ball_frac_NL`: 0.01269
- `min_dist_rw_FR`: 0.01104
- `ball_conf_mean_FR`: 0.00958
- `min_dist_frac_time_NL`: 0.00941
- `min_dist_frac_time_NR`: 0.00721
- `min_dist_rw_best`: 0.00703
- `n_frames_FL`: 0.00572
- `ball_frac_NR`: 0.00517
- `ball_conf_max_FR`: 0.00502
- `rim_frac_FL`: 0.00501
- `rim_frac_NL`: 0.00464
- `post_min_rebound_FR`: 0.00444
- `rim_frac_FR`: 0.00441

## Logistic-regression coefficients (standardised, signed)
Always reported for interpretability. Positive => pushes toward MAKE, negative => toward MISS. Top 15 by |coef|:

- `frac_inside_rim_NL`: 1.1849  (MAKE+)
- `frac_inside_rim_NR`: 0.9364  (MAKE+)
- `through_hoop_far_any`: 0.9164  (MAKE+)
- `through_hoop_conf_max`: 0.619  (MAKE+)
- `ball_conf_mean_NR`: -0.5965  (MISS-)
- `ball_conf_max_overall`: -0.5491  (MISS-)
- `through_hoop_count_FR`: -0.5224  (MISS-)
- `through_hoop_conf_FR`: -0.4974  (MISS-)
- `through_hoop_NL`: -0.4973  (MISS-)
- `min_dist_conf_NR`: 0.4707  (MAKE+)
- `enters_rim_box_NR`: -0.4527  (MISS-)
- `ball_conf_max_NR`: 0.4363  (MAKE+)
- `through_hoop_FR`: 0.4362  (MAKE+)
- `min_dist_conf_NL`: 0.4312  (MAKE+)
- `through_hoop_count_NL`: 0.427  (MAKE+)
