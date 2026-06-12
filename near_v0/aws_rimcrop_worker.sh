#!/bin/bash
# AWS Batch worker: build Phase 1 rim-crop classifier clips for one game
# (NL+NR where available). CPU-bound (ffmpeg+cv2); no torch needed -- rim
# boxes come precomputed in the bundle (DINO prelabels).
# Usage: aws_rimcrop_worker.sh <gid8>
set -euo pipefail
GID="$1"
S3BASE=s3://uball-videos-production/_tmp_tri/near_det

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg curl >/dev/null
pip3 install -q awscli opencv-python-headless numpy

mkdir -p /work && cd /work
aws s3 cp "$S3BASE/rimcrop_bundle.tar.gz" . >/dev/null
tar xzf rimcrop_bundle.tar.gz
# bundle layout mirrors the repo paths the script expects:
# near_v0/build_rimcrop_dataset.py resolves REPO=/work via parents[1]
mkdir -p data/near_detector data/near_rimcrop
mv plays_all.json data/near_detector/
mv labels data/near_detector/labels

INV=$(python3 -c "import json;print(' '.join(json.load(open('s3_inventory.json'))['$GID']['near']))")
DATE=$(python3 -c "import json;print(json.load(open('s3_inventory.json'))['$GID']['date'])")
UUID=$(python3 -c "import json;print(json.load(open('s3_inventory.json'))['$GID']['uuid'])")
for CAM in $INV; do
  S3V="s3://uball-videos-production/court-a/$DATE/$UUID/${DATE}_${UUID}_${CAM}.mp4"
  URL=$(aws s3 presign "$S3V" --expires-in 14400)
  python3 near_v0/build_rimcrop_dataset.py --game "$GID" --angle "$CAM" \
    --video "$URL" --out data/near_rimcrop || echo "WARN: $CAM exited nonzero"
done

tar czf "${GID}_rimcrop.tar.gz" -C data/near_rimcrop .
aws s3 cp "${GID}_rimcrop.tar.gz" "$S3BASE/rimcrop_out/" >/dev/null
echo "JOB_DONE ${GID}"
