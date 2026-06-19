#!/bin/bash
# AWS GPU worker: render the FUSION-ERROR highlight clips for ONE game
# (far+near + GT-vs-prediction panel). Args: <game8> <date> <uuid> <start_idx>
set -euo pipefail
G="$1"; D="$2"; U="$3"; START="${4:-0}"
P=s3://uball-videos-production/_tmp_tri/near_det/e2e_rfdetr
SRC=s3://uball-videos-production/court-a

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 fonts-dejavu-core curl >/dev/null
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 240 -q \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && break
  sleep 15
done
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 240 -q awscli ultralytics opencv-python-headless pillow numpy && break
  sleep 15
done
mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz
for ang in FR FL NR NL; do
  aws s3 cp "$SRC/$D/$U/${D}_${U}_${ang}.mp4" "/work/${ang}.mp4" --only-show-errors
done
python3 near_v0/error_reel.py --game "$G" --fr /work/FR.mp4 --fl /work/FL.mp4 \
  --nr /work/NR.mp4 --nl /work/NL.mp4 --manifest "errmanifests/$G.json" \
  --out "/work/raw_$G.mp4" --far-w weights/far_v16_best.pt --near-w weights/near_det_v1_best.pt \
  --start "$START" --total 34
ffmpeg -y -i "/work/raw_$G.mp4" -c:v libx264 -pix_fmt yuv420p -crf 21 -movflags +faststart \
  "/work/err_$G.mp4" 2>/dev/null
aws s3 cp "/work/err_$G.mp4" "$P/errors/err_$G.mp4" >/dev/null
rm -f /work/*.mp4
echo "ERR_DONE $G"
