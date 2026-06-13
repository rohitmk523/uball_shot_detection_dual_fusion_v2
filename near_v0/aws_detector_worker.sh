#!/bin/bash
# AWS Batch GPU worker: fine-tune YOLO11n near-angle detector.
# Resumable across spot reclaims: pulls last.pt from S3 if present, uploads
# checkpoints + best.pt back every run. Retry attempts re-enter and resume.
set -euo pipefail
P=s3://uball-videos-production/_tmp_tri/near_det/detector
RUN=/work/runs/near-det-v1

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 curl >/dev/null
pip3 install -q awscli ultralytics 2>&1 | tail -1 || true

mkdir -p /work && cd /work
aws s3 cp "$P/dataset.tar.gz" . >/dev/null
tar xzf dataset.tar.gz                       # -> /work/data/yolo_split + base.pt
aws s3 cp "$P/train_detector.py" train_detector.py >/dev/null

# resume from checkpoint if a prior attempt uploaded one
RESUME=""
mkdir -p "$RUN/weights"
if aws s3 cp "$P/ckpt/last.pt" "$RUN/weights/last.pt" >/dev/null 2>&1; then
  echo "resuming from S3 checkpoint"; RESUME="--resume"
fi

# fix dataset.yaml path to the in-container location
sed -i "s#^path:.*#path: /work/data/yolo_split#" /work/data/yolo_split/dataset.yaml

# background uploader: push checkpoints to S3 every 3 min during training
( while true; do
    sleep 180
    [ -f "$RUN/weights/last.pt" ] && aws s3 cp "$RUN/weights/last.pt" "$P/ckpt/last.pt" >/dev/null 2>&1 || true
    [ -f "$RUN/weights/best.pt" ] && aws s3 cp "$RUN/weights/best.pt" "$P/ckpt/best.pt" >/dev/null 2>&1 || true
  done ) &
UP=$!

python3 train_detector.py --data /work/data/yolo_split/dataset.yaml \
  --base /work/data/base.pt --project /work/runs \
  --epochs 100 --imgsz 1280 --batch 16 ${RESUME} 2>&1 | tee /work/train_log.txt

kill $UP 2>/dev/null || true
aws s3 cp "$RUN/weights/best.pt" "$P/best.pt" >/dev/null
aws s3 cp /work/train_log.txt    "$P/train_log.txt" >/dev/null
# results table (mAP etc.)
[ -f "$RUN/results.csv" ] && aws s3 cp "$RUN/results.csv" "$P/results.csv" >/dev/null || true
echo "DETECTOR_DONE"
