#!/bin/bash
# AWS GPU worker: end-to-end shot-detection validation on FROZEN games.
# Pure inference (detector + classifier) -- NO training, so no DataLoader/shm
# issues. Each game's NR video is pulled from S3 in-region (no local disk
# pressure on the M4). Runs the LOCKED config (stride 3, rim-crossing timing),
# NO tuning. First arg "health" runs a 1-game smoke check only.
set -euo pipefail
MODE="${1:-full}"
P=s3://uball-videos-production/_tmp_tri/near_det/e2e
SRC=s3://uball-videos-production/court-a

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 curl >/dev/null
pip3 install -q awscli torch torchvision ultralytics opencv-python-headless numpy 2>&1 | tail -1 || true

mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz
mkdir -p /work/data/near_detector
python3 -c "import torch;print('cuda',torch.cuda.is_available())"

# game: gid8 date uuid t0 t1
GAMES=(
  "6d601c99 2026-04-18 6d601c99-9173-445f-a647 1320 2520"
  "ee8745f1 2026-04-16 ee8745f1-863f-47cf-a43d 1680 2880"
  "0fa23810 2026-05-15 0fa23810-d6e8-4799-995a 0 1200"
  "c2a354fe 2026-03-19 c2a354fe-eb34-4980-af00 0 1200"
)
[ "$MODE" = "health" ] && GAMES=("${GAMES[0]}")

for row in "${GAMES[@]}"; do
  set -- $row; G=$1; D=$2; U=$3; T0=$4; T1=$5
  echo "##### $G ($D) window $T0-$T1 #####"
  aws s3 cp "$SRC/$D/$U/${D}_${U}_NR.mp4" /work/vid.mp4 --only-show-errors
  python3 near_v0/test_end_to_end.py --game "$G" --video /work/vid.mp4 \
    --manifest "/work/frozen_manifests/$G.json" --t0 "$T0" --t1 "$T1" --stride 2 \
    --det /work/weights/near_det_v1_best.pt \
    --clf /work/weights/classifier_all17.pt 2>&1 | grep -E "END-TO-END|SPOT|make recall|miss recall|GT shots" | tee "/work/out_$G.txt"
  aws s3 cp "/work/data/near_detector/e2e_$G.json" "$P/out_v2/e2e_$G.json" >/dev/null 2>&1 || true
  aws s3 cp "/work/out_$G.txt" "$P/out_v2/out_$G.txt" >/dev/null 2>&1 || true
  rm -f /work/vid.mp4
done
echo "E2E_VALIDATION_DONE"
