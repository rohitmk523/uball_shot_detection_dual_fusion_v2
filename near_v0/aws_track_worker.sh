#!/bin/bash
# AWS GPU worker: extract per-frame ball+rim TRACKS for one game with BOTH YOLO
# and RF-DETR (all 4 angles via track_extract.py). Uploads tracks_yolo_<g> +
# tracks_rfdetr_<g> parquets for the downstream fusion comparison.
# Args: <game8> <date> <uuid>
set -euo pipefail
G="$1"; D="$2"; U="$3"
P=s3://uball-videos-production/_tmp_tri/near_det/e2e_rfdetr
SRC=s3://uball-videos-production/court-a

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3-pip ffmpeg libgl1 fonts-dejavu-core curl >/dev/null
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 240 -q \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && break
  echo "torch attempt $i"; sleep 15
done
for i in 1 2 3 4; do
  pip3 install --retries 5 --timeout 240 -q \
    awscli rfdetr ultralytics opencv-python-headless numpy pandas pyarrow && break
  echo "deps attempt $i"; sleep 15
done
python3 -c "import torch;print('CUDA available:', torch.cuda.is_available())"

mkdir -p /work && cd /work
aws s3 cp "$P/bundle.tar.gz" . >/dev/null
tar xzf bundle.tar.gz

for ang in FR FL NR NL; do
  aws s3 cp "$SRC/$D/$U/${D}_${U}_${ang}.mp4" "/work/${ang}.mp4" --only-show-errors
done
python3 near_v0/track_extract.py --game "$G" --shots "demo_data2/$G.json" \
  --fr /work/FR.mp4 --fl /work/FL.mp4 --nr /work/NR.mp4 --nl /work/NL.mp4 \
  --yolo-far weights/far_v16_best.pt --yolo-near weights/near_det_v1_best.pt \
  --rfdetr-far weights/rfdetr_far_best.pth --rfdetr-near weights/rfdetr_near_best.pth \
  --outdir /work 2>&1 | tee "/work/tracklog_$G.txt"
aws s3 cp "/work/tracks_yolo_$G.parquet"   "$P/tracks/tracks_yolo_$G.parquet"   >/dev/null
aws s3 cp "/work/tracks_rfdetr_$G.parquet" "$P/tracks/tracks_rfdetr_$G.parquet" >/dev/null
rm -f /work/*.mp4
echo "TRACKS_DONE $G"
