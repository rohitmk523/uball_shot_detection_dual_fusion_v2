#!/bin/bash
# AWS GPU worker: cache detector raw detections (stride 1) for the spotter
# tuning games. Pure inference. Uploads compact per-frame ball detections JSON.
set -euo pipefail
P=s3://uball-videos-production/_tmp_tri/near_det/spotter
SRC=s3://uball-videos-production/court-a

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 curl >/dev/null
pip3 install -q awscli torch torchvision ultralytics opencv-python-headless numpy 2>&1 | tail -1 || true

mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz
python3 -c "import torch;print('cuda',torch.cuda.is_available())"

# gid8 date uuid t0 t1
GAMES=(
  "72c08cb7 2026-06-02 72c08cb7-0d7c-4dbd-95ea 0 1200"
  "9eb51980 2026-04-17 9eb51980-1219-492e-8aa3 0 1200"
)
for row in "${GAMES[@]}"; do
  set -- $row; G=$1; D=$2; U=$3; T0=$4; T1=$5
  echo "##### caching $G #####"
  aws s3 cp "$SRC/$D/$U/${D}_${U}_NR.mp4" /work/vid.mp4 --only-show-errors
  python3 near_v0/spotter_cache.py --game "$G" --video /work/vid.mp4 \
    --t0 "$T0" --t1 "$T1" --stride 1 \
    --det /work/weights/near_det_v1_best.pt \
    --out "/work/cache_$G.json"
  aws s3 cp "/work/cache_$G.json" "$P/cache/$G.json" >/dev/null
  rm -f /work/vid.mp4
done
echo "SPOTTER_CACHE_DONE"
