#!/bin/bash
# AWS GPU worker: detector v2 = fine-tune v1 best.pt on motion-blur-augmented
# ball frames to lift ball-at-rim recall (blur is the measured enemy at 30fps).
# Resumable; checkpoints best.pt to S3 every 3 min.
set -euo pipefail
P=s3://uball-videos-production/_tmp_tri/near_det/detector
P2=s3://uball-videos-production/_tmp_tri/near_det/detector_v2

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 curl >/dev/null
pip3 install -q awscli ultralytics 2>&1 | tail -1 || true

mkdir -p /work && cd /work
aws s3 cp "$P/dataset.tar.gz" . >/dev/null
tar xzf dataset.tar.gz 2>/dev/null            # -> /work/yolo_split/...
aws s3 cp "$P2/train_detector.py" train_detector.py >/dev/null
aws s3 cp "$P2/blur_augment.py" blur_augment.py >/dev/null
aws s3 cp "$P/ckpt/best.pt" v1_best.pt >/dev/null   # warm-start base

# motion-blur augmentation (adds _mblur ball frames to train)
python3 blur_augment.py /work/yolo_split
echo "train frames now: $(ls /work/yolo_split/train/images | wc -l)"

sed -i "s#^path:.*#path: /work/yolo_split#" /work/yolo_split/dataset.yaml

RUN=/work/runs/near-det-v1     # train_detector.py hardcodes this run name
mkdir -p "$RUN/weights"
RESUME=""
if aws s3 cp "$P2/ckpt/last.pt" "$RUN/weights/last.pt" >/dev/null 2>&1; then
  echo "resuming from v2 checkpoint"; RESUME="--resume"
fi

( while true; do
    sleep 180
    for f in last best; do
      [ -f "$RUN/weights/$f.pt" ] && aws s3 cp "$RUN/weights/$f.pt" "$P2/ckpt/$f.pt" >/dev/null 2>&1 || true
    done
  done ) &
UP=$!

# fine-tune from v1: fewer epochs, warm start, workers=0 (shm-safe)
NW=0 python3 train_detector.py --data /work/yolo_split/dataset.yaml \
  --base /work/v1_best.pt --project /work/runs \
  --epochs 45 --imgsz 1280 --batch 16 ${RESUME} 2>&1 | tee /work/train_log.txt

kill $UP 2>/dev/null || true
BEST=$(ls -t /work/runs/*/weights/best.pt | head -1)
aws s3 cp "$BEST" "$P2/best.pt" >/dev/null
aws s3 cp /work/train_log.txt "$P2/train_log.txt" >/dev/null
echo "DETECTOR_V2_DONE"
