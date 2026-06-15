#!/bin/bash
# Reinstall the Python environments this project uses (miniconda was wiped).
# Best-effort reconstruction from the packages used across the session:
#   base            : pandas / numpy / scipy / scikit-learn / pyarrow  (analysis, parquet)
#   far_angle       : ultralytics / opencv / torch / pillow            (detection, demo render)
#   uball_shot_detection : torch / torchvision / numpy                 (classifier training)
# macOS arm64. Re-run is safe (skips install if conda already present).
set -euo pipefail
MC="$HOME/miniconda3"

if [ ! -x "$MC/bin/conda" ]; then
  echo "== installing miniconda (arm64) =="
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$MC"
fi
source "$MC/etc/profile.d/conda.sh"
conda config --set always_yes true >/dev/null 2>&1 || true

echo "== base: analysis stack =="
"$MC/bin/pip" install -q numpy pandas scipy scikit-learn pyarrow pillow || true

echo "== env: far_angle (CV + detection + render) =="
conda create -n far_angle python=3.11 -y >/dev/null 2>&1 || true
conda run -n far_angle pip install -q \
  ultralytics opencv-python-headless torch torchvision numpy pillow scipy || true

echo "== env: uball_shot_detection (classifier training) =="
conda create -n uball_shot_detection python=3.11 -y >/dev/null 2>&1 || true
conda run -n uball_shot_detection pip install -q \
  torch torchvision numpy opencv-python-headless || true

echo ""
echo "== verify =="
"$MC/bin/python" -c "import pandas,numpy,pyarrow;print('base: pandas/numpy/pyarrow OK')" || echo "base FAILED"
conda run -n far_angle python -c "import cv2,ultralytics,torch;print('far_angle: cv2/ultralytics/torch OK')" || echo "far_angle FAILED"
conda run -n uball_shot_detection python -c "import torch;print('uball: torch OK')" || echo "uball FAILED"
echo ""
echo "NOTE: exact package versions weren't pinned (envs were wiped). If a version"
echo "mismatch bites, pin it here. ultralytics auto-pulls a matching torch."
