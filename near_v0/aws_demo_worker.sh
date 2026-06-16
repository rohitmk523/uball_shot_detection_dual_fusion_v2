#!/bin/bash
# AWS GPU worker: render the per-game 4-panel demo for ONE game. Downloads NR+FR
# from S3, runs demo_game.py over ALL shots, h264-encodes, uploads. One game per
# job (a spot reclaim loses only one game).
set -euo pipefail
G="$1"
P=s3://uball-videos-production/_tmp_tri/near_det/demo
SRC=s3://uball-videos-production/court-a

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 fonts-dejavu-core curl >/dev/null
# robust install: retry (torch pip can fail transiently); verify torch before proceeding.
# Pin torch to a CUDA 12.1 build: the default wheel is now cu128 and needs a newer
# driver than these instances have (driver=12.4), which silently drops to CPU.
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 180 -q \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && break
  echo "torch pip attempt $i failed; retrying in 15s"; sleep 15
done
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 180 -q \
    awscli ultralytics opencv-python-headless pillow numpy && break
  echo "deps pip attempt $i failed; retrying in 15s"; sleep 15
done
python3 -c "import torch" || { echo "FATAL: torch not installed after retries"; exit 1; }
python3 -c "import torch;print('CUDA available:', torch.cuda.is_available())"

mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz
python3 -c "import torch;print('cuda',torch.cuda.is_available())"

read GG DATE UUID T < <(grep "^$G " games.txt)
echo "game $G  $DATE  $UUID  (both baskets, disk-safe 2-pass)"
TOTAL=$(python3 -c "import json;print(json.load(open('demo_data/$G.json'))['n'])")
W=weights/near_det_v1_best.pt; FW=weights/far_v16_best.pt
SEGS=()

# ---- RIGHT basket (NR + FR), then free the disk ----
aws s3 cp "$SRC/$DATE/$UUID/${DATE}_${UUID}_NR.mp4" /work/nr.mp4 --only-show-errors
aws s3 cp "$SRC/$DATE/$UUID/${DATE}_${UUID}_FR.mp4" /work/fr.mp4 --only-show-errors
python3 near_v0/demo_game.py --game "$G" --basket RIGHT --near /work/nr.mp4 --far /work/fr.mp4 \
  --shots "demo_data/$G.json" --out /work/right.mp4 --pts-out /work/pts.json --total "$TOTAL" \
  --near-w "$W" --far-w "$FW"
rm -f /work/nr.mp4 /work/fr.mp4
[ -f /work/right.mp4 ] && SEGS+=(/work/right.mp4)

# ---- LEFT basket (NL + FL), rim map continues from pts.json ----
aws s3 cp "$SRC/$DATE/$UUID/${DATE}_${UUID}_NL.mp4" /work/nl.mp4 --only-show-errors
aws s3 cp "$SRC/$DATE/$UUID/${DATE}_${UUID}_FL.mp4" /work/fl.mp4 --only-show-errors
python3 near_v0/demo_game.py --game "$G" --basket LEFT --near /work/nl.mp4 --far /work/fl.mp4 \
  --shots "demo_data/$G.json" --out /work/left.mp4 --pts-in /work/pts.json --total "$TOTAL" \
  --near-w "$W" --far-w "$FW"
rm -f /work/nl.mp4 /work/fl.mp4
[ -f /work/left.mp4 ] && SEGS+=(/work/left.mp4)

# ---- stitch the two baskets + h264 ----
if [ ${#SEGS[@]} -eq 2 ]; then
  ffmpeg -y -i /work/right.mp4 -i /work/left.mp4 -filter_complex "[0:v][1:v]concat=n=2:v=1[v]" \
    -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 21 -movflags +faststart "/work/${G}_demo.mp4" 2>/dev/null
else
  ffmpeg -y -i "${SEGS[0]}" -c:v libx264 -pix_fmt yuv420p -crf 21 -movflags +faststart "/work/${G}_demo.mp4" 2>/dev/null
fi
aws s3 cp "/work/${G}_demo.mp4" "$P/out_both/${G}_demo.mp4" >/dev/null
echo "DEMO_DONE $G"
