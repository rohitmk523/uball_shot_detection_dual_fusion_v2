# P3 Results v2 — interpretable make/miss model (iteration 2)

Seed=42. Whole-game splits (no game spans splits, no leakage). Test opened once. Features: `p2_features_v2.parquet`.

## Split sizes
- train: 2194 shots
- val: 482 shots
- test: 485 shots

## Model selection (tuned on VAL)
- logreg: val_f1=0.6831 val_acc=0.7593 thr*=0.72
- hgb: val_f1=0.8345 val_acc=0.8527 thr*=0.58

**Selected model: `hgb` (threshold=0.58)**

## GroupKFold CV (train+val, grouped by game)
- folds=5  acc=0.8041±0.0453  f1=0.7991±0.0531  auc=0.8816±0.05

## HELD-OUT TEST METRICS

### Overall (make/miss, incl 4PT_MAKE)
- n=485  acc=0.8742  prec=0.8373  rec=0.8663  f1=0.8516  auc=0.944
- confusion: TN=249 FP=34 FN=27 TP=175

### v1_in_scope=True (apples-to-apples vs v1 85.7/79.5/89.2)
- n=455  acc=0.8813  prec=0.8172  rec=0.8837  f1=0.8492  auc=0.95
- confusion: TN=249 FP=34 FN=20 TP=152

### 4PT_MAKE only
- n=30  recall_as_make=0.7667

## Iteration 2 vs Iteration 1 vs v1 baseline vs target

Overall held-out test (make/miss incl 4PT_MAKE):

| metric | v1 baseline | iter1 | **iter2** | target |
|---|---|---|---|---|
| accuracy | 0.857 | 0.8474 | **0.8742** | 0.92 |
| precision | 0.795 | 0.7735 | **0.8373** | 0.9 |
| recall | 0.892 | 0.896 | **0.8663** | 0.9 |

v1_in_scope subset (apples-to-apples vs v1 85.7/79.5/89.2):

| metric | v1 baseline | iter1 | **iter2** | target |
|---|---|---|---|---|
| accuracy | 0.857 | 0.8484 | **0.8813** | 0.92 |
| precision | 0.795 | 0.7464 | **0.8172** | 0.9 |
| recall | 0.892 | 0.907 | **0.8837** | 0.9 |

> Note: iter1 numbers used plain F1-optimal thresholding; iter2 uses a precision-aware threshold (precision is the stated bottleneck), tuned on VAL only. The ablation below isolates the feature levers because its `V1 only` stage ALSO uses the iter2 threshold policy — so stage-to-stage deltas are pure feature effects, while the iter1-vs-iter2 table reflects the combined (features + threshold policy) improvement.

## Ablation — levers added incrementally

Each stage re-selects model + threshold on VAL, then evaluates ONCE on TEST. Overall metrics:

| stage | #feat | model | acc | prec | rec | f1 | FP | FN |
|---|---|---|---|---|---|---|---|---|
| V1 only (iteration-1 features) | 91 | hgb | 0.8515 | 0.825 | 0.8168 | 0.8209 | 35 | 37 |
| V1 + L3 (quality-gated fusion) | 101 | hgb | 0.8474 | 0.8265 | 0.802 | 0.8141 | 34 | 40 |
| V1 + L3 + L2 (arc fit) | 138 | hgb | 0.866 | 0.8341 | 0.8465 | 0.8403 | 34 | 31 |
| V1 + L3 + L2 + L1 (bounce-out) = ALL | 167 | hgb | 0.8742 | 0.8373 | 0.8663 | 0.8516 | 34 | 27 |

Anchor game (c2a354fe) per stage:

| stage | acc | prec | rec |
|---|---|---|---|
| V1 only (iteration-1 features) | 0.8095 | 0.8143 | 0.7125 |
| V1 + L3 (quality-gated fusion) | 0.8201 | 0.8194 | 0.7375 |
| V1 + L3 + L2 (arc fit) | 0.8519 | 0.8421 | 0.8 |
| V1 + L3 + L2 + L1 (bounce-out) = ALL | 0.8466 | 0.8228 | 0.8125 |

## Per-game test breakdown
- `6d601c99-9173-445f-a647-dadbb152fe96`: n=136 acc=0.9338 prec=0.88 rec=0.9362 f1=0.9072
- `c2a354fe-eb34-4980-af00-8f5ff6b00143`  <-- ANCHOR: n=189 acc=0.8466 prec=0.8228 rec=0.8125 f1=0.8176
- `ee8745f1-863f-47cf-a43d-d90c58cc9bb2`: n=160 acc=0.8562 prec=0.825 rec=0.88 f1=0.8516

## Interpretability — top 20 features
(permutation importance (F1))

- `frac_inside_rim_max`: 0.13119
- `min_dist_rw_near_best`: 0.08542
- `rim_frac_NR`: 0.01718
- `min_dist_conf_NL`: 0.01196
- `min_dist_conf_NR`: 0.01187
- `min_dist_rw_best`: 0.01098
- `through_hoop_conf_max`: 0.01044
- `ball_conf_max_overall`: 0.00977
- `ball_conf_mean_FR`: 0.00922
- `bo_conf_fused`: 0.00878
- `q_quality_max`: 0.0083
- `bo_frames_to_reappear_NL`: 0.00783
- `min_dist_frac_time_NR`: 0.00762
- `frac_inside_rim_NL`: 0.00656
- `min_dist_frac_time_NL`: 0.00605
- `ball_conf_max_FR`: 0.00562
- `min_dist_rw_FL`: 0.00555
- `arc_entry_angle_qw`: 0.00505
- `min_dist_rw_NL`: 0.00481
- `min_dist_rw_far_best`: 0.0043

## Logistic-regression coefficients (standardised, signed)
Always reported for interpretability. Positive => pushes toward MAKE, negative => toward MISS. Top 15 by |coef|:

- `frac_inside_rim_NL`: 1.1774  (MAKE+)
- `frac_inside_rim_NR`: 0.8967  (MAKE+)
- `bo_any_FR`: 0.8027  (MAKE+)
- `through_hoop_NL`: -0.7878  (MISS-)
- `through_hoop_votes`: -0.7815  (MISS-)
- `through_hoop_far_any`: 0.7637  (MAKE+)
- `through_hoop_conf_max`: 0.6451  (MAKE+)
- `through_hoop_conf_FR`: -0.6147  (MISS-)
- `through_hoop_count_FR`: -0.6135  (MISS-)
- `bo_conf_NL`: -0.6035  (MISS-)
- `ball_conf_max_overall`: -0.5929  (MISS-)
- `through_hoop_agree2`: 0.5891  (MAKE+)
- `enters_rim_box_NL`: -0.5721  (MISS-)
- `enters_rim_box_NR`: -0.5674  (MISS-)
- `through_hoop_NR`: -0.5612  (MISS-)
