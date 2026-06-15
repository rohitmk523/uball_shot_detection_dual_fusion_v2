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
pip3 install -q awscli torch torchvision ultralytics opencv-python-headless pillow numpy 2>&1 | tail -1 || true

mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz
python3 -c "import torch;print('cuda',torch.cuda.is_available())"

read GG DATE UUID T < <(grep "^$G " games.txt)
echo "game $G  $DATE  $UUID  (both baskets)"
for CAM in NR FR NL FL; do
  aws s3 cp "$SRC/$DATE/$UUID/${DATE}_${UUID}_${CAM}.mp4" "/work/${CAM,,}.mp4" --only-show-errors
done

python3 near_v0/demo_game.py --game "$G" \
  --nr /work/nr.mp4 --fr /work/fr.mp4 --nl /work/nl.mp4 --fl /work/fl.mp4 \
  --shots "demo_data/$G.json" --out "/work/${G}_raw.mp4" \
  --near-w weights/near_det_v1_best.pt --far-w weights/far_v16_best.pt

ffmpeg -y -i "/work/${G}_raw.mp4" -c:v libx264 -pix_fmt yuv420p -crf 21 \
  -movflags +faststart "/work/${G}_demo.mp4" 2>/dev/null
aws s3 cp "/work/${G}_demo.mp4" "$P/out_both/${G}_demo.mp4" >/dev/null
rm -f /work/nr.mp4 /work/fr.mp4 /work/nl.mp4 /work/fl.mp4
echo "DEMO_DONE $G"
