#!/bin/bash
# AWS GPU worker: near-angle pipeline e2e, YOLO vs RF-DETR, apples-to-apples.
# For each frozen game: stream the NR window from S3, run test_end_to_end.py
# (YOLO) AND test_end_to_end_rfdetr.py (RF-DETR) with the SAME spotter +
# classifier + GT matching, stride 3. Uploads spot_recall + e2e_acc for both.
# Arg1 = game id (one game per job) or "all".
set -euo pipefail
G="${1:-all}"
P=s3://uball-videos-production/_tmp_tri/near_det/e2e_rfdetr
SRC=s3://uball-videos-production/court-a

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 fonts-dejavu-core curl >/dev/null
# torch cu121 FIRST (g4dn driver=12.4 can't run the default cu128 wheel -> CPU)
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 240 -q \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && break
  echo "torch attempt $i failed; retry"; sleep 15
done
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 240 -q \
    awscli rfdetr ultralytics opencv-python-headless numpy && break
  echo "deps attempt $i failed; retry"; sleep 15
done
python3 -c "import torch;print('CUDA available:', torch.cuda.is_available())"
python3 -c "import rfdetr, ultralytics; print('rfdetr+ultralytics import OK')"

mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz
mkdir -p /work/data/near_detector

# game: gid8 date uuid t0 t1   (windows from aws_e2e_worker.sh)
ALL=(
  "6d601c99 2026-04-18 6d601c99-9173-445f-a647 1320 2520"
  "ee8745f1 2026-04-16 ee8745f1-863f-47cf-a43d 1680 2880"
  "0fa23810 2026-05-15 0fa23810-d6e8-4799-995a 0 1200"
  "c2a354fe 2026-03-19 c2a354fe-eb34-4980-af00 0 1200"
)
GAMES=()
for row in "${ALL[@]}"; do set -- $row; [ "$G" = "all" ] || [ "$G" = "$1" ] && GAMES+=("$row"); done

W=weights/near_det_v1_best.pt; CLF=weights/classifier_all17.pt; RW=weights/rfdetr_near_best.pth
for row in "${GAMES[@]}"; do
  set -- $row; g=$1; D=$2; U=$3; T0=$4; T1=$5
  echo "##### $g ($D) window $T0-$T1 #####"
  aws s3 cp "$SRC/$D/$U/${D}_${U}_NR.mp4" /work/vid.mp4 --only-show-errors
  echo "----- YOLO $g -----"
  python3 near_v0/test_end_to_end.py --game "$g" --video /work/vid.mp4 \
    --manifest "frozen_manifests/$g.json" --t0 "$T0" --t1 "$T1" --stride 3 \
    --det "$W" --clf "$CLF" 2>&1 | grep -E "SPOT|END-TO-END|make recall|miss recall|GT shots" \
    | tee "/work/yolo_$g.txt"
  echo "----- RF-DETR $g -----"
  python3 near_v0/test_end_to_end_rfdetr.py --game "$g" --video /work/vid.mp4 \
    --manifest "frozen_manifests/$g.json" --t0 "$T0" --t1 "$T1" --stride 3 \
    --det "$RW" --clf "$CLF" 2>&1 | grep -E "SPOT|END-TO-END|make recall|miss recall|GT shots|progress" \
    | tee "/work/rfdetr_$g.txt"
  aws s3 cp "/work/data/near_detector/e2e_$g.json" "$P/out/e2e_yolo_$g.json" >/dev/null 2>&1 || true
  aws s3 cp "/work/data/near_detector/e2e_rfdetr_$g.json" "$P/out/e2e_rfdetr_$g.json" >/dev/null 2>&1 || true
  aws s3 cp "/work/yolo_$g.txt" "$P/out/yolo_$g.txt" >/dev/null 2>&1 || true
  aws s3 cp "/work/rfdetr_$g.txt" "$P/out/rfdetr_$g.txt" >/dev/null 2>&1 || true
  rm -f /work/vid.mp4
done
echo "E2E_RFDETR_DONE $G"
