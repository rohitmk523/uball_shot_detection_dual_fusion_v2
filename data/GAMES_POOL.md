# Annotated games pool (uball.ai `plays`, source=manual, ≥60 shots)

P0 query (2026-05-18): **~135 games** with full human shot annotation, 60–224 shots each.
Annotation availability is NOT the constraint. The constraint is which of these also
have the 4 angle videos in `s3://uball-videos-production/`.

Top candidates by shot count (game_id : shots = makes/misses):
- c3f84436-2bbf-4ffa-89bb-5ab280498aeb : 224 = 102/122
- 26bb5808-3925-435e-9f83-d4bebe03c5be : 202 = 92/110
- 2399cfac-684d-4768-b6ac-d24e87c2427b : 201 = 106/95
- 50a8463b-58b6-4de1-b2d0-f054269162fe : 198 = 74/124
- 4a247d0d-51be-4643-a532-455cf5da4382 : 196 = 109/87
- 81661ff3-1b04-4862-a08d-1f1aed09262f : 195 = 88/107
- c2a354fe-eb34-4980-af00-8f5ff6b00143 : 189 = 80/109  (v1-validated reference)
- … ~127 more games with 92–190 shots.

NEXT (autonomous P0→P1): for each game_id, resolve court/date and confirm the 4
`{FL,FR,NL,NR}` videos exist in S3 → write `data/games_manifest.json` with
TRAIN/VAL/TEST split (whole-game assignment, no leakage). Target ≥6 train / ≥2 val / ≥2 test.
