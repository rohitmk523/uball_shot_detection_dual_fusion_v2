#!/bin/bash
# AWS GPU worker: FAR-angle geometric make/miss, YOLO vs RF-DETR, per frozen game.
# Streams the FR (far-right) video, runs far_makemiss_eval.py (both detectors in
# one pass) over the GT shot windows, uploads far_<game>.json. Arg1 = game | all.
set -euo pipefail
G="${1:-all}"
P=s3://uball-videos-production/_tmp_tri/near_det/e2e_rfdetr
SRC=s3://uball-videos-production/court-a

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 fonts-dejavu-core curl >/dev/null
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 240 -q \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && break
  echo "torch attempt $i failed"; sleep 15
done
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 240 -q awscli rfdetr ultralytics opencv-python-headless numpy && break
  echo "deps attempt $i failed"; sleep 15
done
python3 -c "import torch;print('CUDA available:', torch.cuda.is_available())"

mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz

ALL=(
  "6d601c99 2026-04-18 6d601c99-9173-445f-a647 1320 2520"
  "ee8745f1 2026-04-16 ee8745f1-863f-47cf-a43d 1680 2880"
  "0fa23810 2026-05-15 0fa23810-d6e8-4799-995a 0 1200"
  "c2a354fe 2026-03-19 c2a354fe-eb34-4980-af00 0 1200"
)
GAMES=()
for row in "${ALL[@]}"; do set -- $row; [ "$G" = "all" ] || [ "$G" = "$1" ] && GAMES+=("$row"); done

YW=weights/far_v16_best.pt; RW=weights/rfdetr_far_best.pth
for row in "${GAMES[@]}"; do
  set -- $row; g=$1; D=$2; U=$3; T0=$4; T1=$5
  echo "##### FAR $g ($D) window $T0-$T1 #####"
  aws s3 cp "$SRC/$D/$U/${D}_${U}_FR.mp4" /work/fr.mp4 --only-show-errors
  python3 near_v0/far_makemiss_eval.py --game "$g" --video /work/fr.mp4 \
    --manifest "frozen_manifests/$g.json" --t0 "$T0" --t1 "$T1" \
    --yolo "$YW" --rfdetr "$RW" 2>&1 | grep -E "FAR make/miss|YOLO|RFDETR|GT shots|wrote|shots" | tee "/work/farout_$g.txt"
  aws s3 cp "/work/far_$g.json" "$P/out/far_$g.json" >/dev/null 2>&1 || true
  aws s3 cp "/work/farout_$g.txt" "$P/out/farout_$g.txt" >/dev/null 2>&1 || true
  rm -f /work/fr.mp4
done
echo "FAR_DONE $G"
