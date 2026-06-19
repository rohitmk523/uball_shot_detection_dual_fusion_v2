#!/bin/bash
# Repo cleanup for version control — deletes ~7GB of bulky binary cruft from DISK,
# keeps the RF-DETR-fusion + reel + report (the peak-accuracy work) and the
# triangulation CODE (pipeline/*.py). Review before running.
#
#   bash scripts/cleanup_repo.sh
#
# The two keepers that live inside delete-wholesale dirs are already stashed at
# /tmp/uball_keep (logo_results.json + FUSION_ERRORS_reel.mp4); this restores them.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "repo before: $(du -sh . | awk '{print $1}')"

# stash keepers (idempotent — already done by the assistant)
mkdir -p /tmp/uball_keep
[ -f data/near_rimcrop/cache/logo_results.json ] && cp data/near_rimcrop/cache/logo_results.json /tmp/uball_keep/ || true
[ -f data/near_detector/demo/games/FUSION_ERRORS_reel.mp4 ] && cp data/near_detector/demo/games/FUSION_ERRORS_reel.mp4 /tmp/uball_keep/ || true

# --- DELETE bulky binary cruft (frames / videos / crops / caches) ---
rm -rf data/crops data/near_rimcrop
rm -rf data/near_detector/demo data/near_detector/frames data/near_detector/aws_out \
       data/near_detector/review_clips data/near_detector/labels data/near_detector/overlays \
       data/near_detector/camera_mode_check data/near_detector/spotter_cache data/near_detector/near_rimcrop
rm -rf data/client_report/near_angle data/client_report/error_highlights \
       data/client_report/error_highlights_v8far data/client_report/error_highlights_FINAL \
       data/client_report/fresh_error_reel data/client_report/calib_freethrow \
       data/client_report/validation_sync data/client_report/mode_probe \
       data/client_report/triangulation_test

# old feature/prediction parquets — keep only v8far, v8far_angleaware, p3_logo_oof
find data -maxdepth 1 -name "*.parquet" ! -name "*v8far*" ! -name "p3_logo_oof_predictions.parquet" -delete
# scattered media at data/ root
find data -maxdepth 1 \( -name "Screenshot*.png" -o -name "audio_features_*.csv" \) -delete 2>/dev/null || true

# --- restore the two keepers ---
mkdir -p data/near_rimcrop/cache data/near_detector/demo/games
[ -f /tmp/uball_keep/logo_results.json ] && mv /tmp/uball_keep/logo_results.json data/near_rimcrop/cache/ || true
[ -f /tmp/uball_keep/FUSION_ERRORS_reel.mp4 ] && mv /tmp/uball_keep/FUSION_ERRORS_reel.mp4 data/near_detector/demo/games/ || true

echo "repo after:  $(du -sh . | awk '{print $1}')"
echo ""
echo "=== keep-set check ==="
for f in data/p2_features_v8far.parquet data/p2_features_v8far_angleaware.parquet \
         data/near_rimcrop/cache/logo_results.json data/p3_logo_oof_predictions.parquet \
         data/near_detector/demo/games/FUSION_ERRORS_reel.mp4 data/client_report/CLIENT_REPORT.md; do
  [ -f "$f" ] && echo "  ok  $f" || echo "  !! MISSING $f"
done
echo "  demo_data2: $(ls data/near_detector/demo_data2/*.json 2>/dev/null | wc -l | tr -d ' ') jsons"
echo "  frozen_manifests: $(ls data/near_detector/frozen_manifests/*.json 2>/dev/null | wc -l | tr -d ' ') jsons"
echo "  weights: $(ls near_v0/weights/*.pt 2>/dev/null | wc -l | tr -d ' ') | triangulation code: $(ls pipeline/triangulate_shot.py pipeline/calibrate_v5.py 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "next:  git add -A && git commit -m 'chore: prune bulky binaries; clean repo for version control'"
