#!/bin/bash
# AWS GPU worker: PROBE the rim-map dot positions (new near_cross) for one
# game's candidate pool, both baskets. Downloads only the NEAR videos (NR, NL)
# -> no rendering, no far video. Uploads probe_<G>_R.json / _L.json so we can
# verify the localization fix and curate a clean reel before rendering.
set -euo pipefail
G="$1"
P=s3://uball-videos-production/_tmp_tri/near_det/demo
SRC=s3://uball-videos-production/court-a

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 fonts-dejavu-core curl >/dev/null
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 180 -q \
    awscli torch torchvision ultralytics opencv-python-headless pillow numpy && break
  echo "pip attempt $i failed; retrying"; sleep 15
done
python3 -c "import torch" || { echo "FATAL: torch missing"; exit 1; }

mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz
python3 -c "import torch;print('cuda',torch.cuda.is_available())"

read GG DATE UUID T < <(grep "^$G " games.txt)
W=weights/near_det_v1_best.pt
POOL=demo_data2/$G.json
echo "probe $G  $DATE  $UUID  pool=$POOL"

aws s3 cp "$SRC/$DATE/$UUID/${DATE}_${UUID}_NR.mp4" /work/nr.mp4 --only-show-errors
python3 near_v0/dot_probe.py --near /work/nr.mp4 --shots "$POOL" --basket RIGHT \
  --near-w "$W" --out /work/probe_R.json
rm -f /work/nr.mp4

aws s3 cp "$SRC/$DATE/$UUID/${DATE}_${UUID}_NL.mp4" /work/nl.mp4 --only-show-errors
python3 near_v0/dot_probe.py --near /work/nl.mp4 --shots "$POOL" --basket LEFT \
  --near-w "$W" --out /work/probe_L.json
rm -f /work/nl.mp4

aws s3 cp /work/probe_R.json "$P/out_both/probe_${G}_R.json" >/dev/null
aws s3 cp /work/probe_L.json "$P/out_both/probe_${G}_L.json" >/dev/null
echo "PROBE_UPLOADED $G"
