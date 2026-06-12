#!/bin/bash
# AWS Batch worker: extract sampled near-angle frames for one game (NL+NR
# where available) and pre-label with Grounding DINO on GPU.
# Usage: aws_near_det_worker.sh <gid8>
set -euo pipefail
GID="$1"
S3BASE=s3://uball-videos-production/_tmp_tri/near_det

apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq ffmpeg libgl1 >/dev/null 2>&1 || true
pip install -q --no-input transformers pillow numpy awscli 2>/dev/null | tail -1 || true

mkdir -p /work && cd /work
aws s3 cp "$S3BASE/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz

python3 - "$GID" <<'PY'
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/work")
gid = sys.argv[1]
inv = json.loads(Path("s3_inventory.json").read_text())[gid]
manifest = json.loads(Path("sampling_manifest.json").read_text())
from mine_detector_frames import extract

for cam in inv["near"]:
    key = f"{gid}_{cam}"
    frames = manifest.get(key)
    if not frames:
        print(f"skip {key}: not in manifest"); continue
    s3 = (f"s3://uball-videos-production/court-a/{inv['date']}/{inv['uuid']}/"
          f"{inv['date']}_{inv['uuid']}_{cam}.mp4")
    url = subprocess.run(["aws", "s3", "presign", s3, "--expires-in", "7200"],
                         capture_output=True, text=True, check=True).stdout.strip()
    n = extract(gid, cam, url, Path("frames"), frames)
    print(f"{key}: {n}/{len(frames)} frames", flush=True)
    if n < len(frames) * 0.8:  # one retry pass for S3 hiccups
        n = extract(gid, cam, url, Path("frames"), frames)
        print(f"{key} retry: {n}/{len(frames)}", flush=True)
PY

python3 prelabel_near.py --mode full --frames-dir frames \
  --labels-dir labels --overlays overlays

tar czf "${GID}_neardet.tar.gz" frames labels overlays
aws s3 cp "${GID}_neardet.tar.gz" "$S3BASE/out/" >/dev/null
echo "JOB_DONE ${GID}"
