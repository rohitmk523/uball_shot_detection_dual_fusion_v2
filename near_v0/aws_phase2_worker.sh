#!/bin/bash
# AWS Batch GPU worker: full 17-fold LOGO training for the v0 rim-crop
# classifier. Pulls tensor cache + trainer, runs --all, uploads results.
# phase2_train.py resolves REPO=/work via parents[1]; cache lives at
# /work/data/near_rimcrop/cache to match.
set -euo pipefail
P2=s3://uball-videos-production/_tmp_tri/near_det/phase2

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 curl >/dev/null
pip3 install -q awscli
pip3 install -q torch torchvision opencv-python-headless numpy 2>&1 | tail -1 || true

mkdir -p /work/data/near_rimcrop/cache /work/near_v0 && cd /work
aws s3 cp "$P2/frames_u8.npy"   data/near_rimcrop/cache/frames_u8.npy >/dev/null
aws s3 cp "$P2/meta.json"       data/near_rimcrop/cache/meta.json     >/dev/null
aws s3 cp "$P2/phase2_train.py" near_v0/phase2_train.py               >/dev/null
python3 -c "import torch;print('cuda',torch.cuda.is_available())"

python3 near_v0/phase2_train.py --all --epochs 18 \
  --out /work/logo_results.json 2>&1 | tee /work/train_log.txt

aws s3 cp /work/logo_results.json "$P2/logo_results.json" >/dev/null
aws s3 cp /work/train_log.txt     "$P2/train_log.txt"     >/dev/null
echo "PHASE2_DONE"
