# 02 — Data & Ground Truth

## Ground truth: uball.ai `plays`
- Project: **uball.ai** Supabase, ref `mhbrsftxvxxtfgbajrlc` (NOT Uball-core `kjgnswlxsqhayabdpheh`). URL in `.env`.
- Table `public.plays`, columns: `id, game_id, timestamp_seconds, start_timestamp, end_timestamp, classification, angle, source, confidence`.
- Shot classes: `FG_MAKE, FG_MISS, 3PT_MAKE, 3PT_MISS, 4PT_MAKE, 4PT_MISS, FREE_THROW_MAKE, FREE_THROW_MISS`. `source='manual'` = human annotator. `*_MAKE`→made, `*_MISS`→missed. `angle` L/R = hoop side.
- `timestamp_seconds` is on the **video** axis (same as detector `timestamp_seconds`); join CV↔GT by nearest within ±3 s, fitting the best constant offset per game (offset varies per game; search −15..+15 s, pick max matches).
- Access: Supabase MCP (`mcp__supabase__execute_sql`) or REST with `SUPABASE_SERVICE_ROLE_KEY`. Example query:
  `SELECT timestamp_seconds, classification, angle FROM plays WHERE game_id='<uuid>' AND classification IN (...8 shot classes...) ORDER BY timestamp_seconds;`

## Game selection (cross-game — Trap 2)
Need games that have BOTH: (a) human `plays` annotation (`source='manual'`, full game), and (b) 4 angle videos in S3.
- Validated reference game: `c2a354fe-eb34-4980-af00-8f5ff6b00143` (189 shots) — videos at `s3://uball-videos-production/court-a/2026-03-19/c2a354fe-eb34-4980-af00/2026-03-19_c2a354fe-eb34-4980-af00_{FL,FR,NL,NR}.mp4`.
- **First implementation step:** enumerate candidate games — query `plays` for `game_id`s with a full set of 8-class shots and `source='manual'`, then check S3 `s3://uball-videos-production/` for the matching 4 angle videos. Produce `data/games_manifest.json` (game_id, date, court, S3 video keys, #annotated shots).
- **Split:** assign whole games to TRAIN / VAL / TEST (never split a game across sets — prevents leakage). Minimum: ≥3 train, ≥1 val, ≥1 test; more is better. TEST games are touched only for the final number.

## CV side (reuse v1 outputs where possible)
- v16 fused + per-angle session JSONs for c2a354fe already exist:
  `s3://uball-cv-results/cv-results/court-a/2026-03-19/12a088eb-be66-4514-91b1/side-{A,B}/{detection_results,near_session,far_session}.json`.
- For other games, run the v16 pipeline (model `v2-prod-far`) on AWS per `04_AWS_RUNTIME.md` to produce the per-angle session JSONs + raw tracks.

## Models (frozen, reused)
`s3://uball-cv-models/yolov11/v2-prod-far/` — near `best.pt` sha `4dc41e14751b…`, far `best.pt` sha `73eb79c66c6c…`, `MANIFEST.json`. Do not retrain these in this track.

## Dataset artifact
`data/dataset.parquet`: one row per GT shot across all selected games — feature vector (`03_FEATURES.md`) + label (made/missed) + game_id + split + matched flags (near_present/far_present). This is the single input to model fitting.
