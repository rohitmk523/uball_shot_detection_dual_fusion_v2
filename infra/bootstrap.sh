#!/usr/bin/env bash
# =============================================================================
# P1 GPU bootstrap (EC2 user-data / cloud-init).
#
# Runs on a g4dn.xlarge spot box. It:
#   1. Arms a HARD shutdown timer FIRST (shutdown -h +<MINUTES>) so the box
#      self-terminates at the budget cap REGARDLESS of job state.
#   2. Installs deps, pulls + sha-verifies the frozen detector bundle,
#      pulls the staged pipeline/ code from S3.
#   3. Runs run_batch.py for the assigned games.
#   4. On EXIT (success OR failure OR error trap): uploads logs to S3 and
#      terminates the instance.
#
# No credentials are embedded: the box uses its IAM instance profile.
# Values below are substituted by infra/launch_p1.sh before submission.
# =============================================================================
set -uo pipefail

# ---- Parameters injected by launch_p1.sh (placeholders) ---------------------
S3_WORK_PREFIX="__S3_WORK_PREFIX__"
S3_CODE_TARBALL="__S3_CODE_TARBALL__"          # s3://.../jobs/<run_id>/pipeline.tgz
FROZEN_BUNDLE_S3="__FROZEN_BUNDLE_S3__"
FROZEN_BUNDLE_SHA_S3="__FROZEN_BUNDLE_SHA_S3__"
RUN_ID="__RUN_ID__"
SPLIT="__SPLIT__"
GAME_ARGS="__GAME_ARGS__"                       # e.g. '--split train' or '--games a b'
HARD_CAP_MINUTES="__HARD_CAP_MINUTES__"         # integer; absolute kill switch
SPOT_USD_PER_HR="__SPOT_USD_PER_HR__"
SUPABASE_URL="__SUPABASE_URL__"
SUPABASE_SERVICE_ROLE_KEY="__SUPABASE_SERVICE_ROLE_KEY__"
UPLOAD_BUCKET="__UPLOAD_BUCKET__"
AWS_REGION="__AWS_REGION__"

WORK=/opt/p1
LOG=/var/log/p1_bootstrap.log
mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

echo "[p1] bootstrap start $(date -u +%FT%TZ) run=$RUN_ID"

# ---- 1. HARD cap: arm the absolute kill switch BEFORE doing anything --------
# shutdown at the budget cap no matter what. instance-initiated-shutdown-
# behavior=terminate (set at launch) turns this halt into a terminate.
if [[ "$HARD_CAP_MINUTES" =~ ^[0-9]+$ ]] && [[ "$HARD_CAP_MINUTES" -gt 0 ]]; then
  echo "[p1] arming hard shutdown in +$HARD_CAP_MINUTES min (budget cap)"
  shutdown -h +"$HARD_CAP_MINUTES" "p1 budget hard cap" || true
else
  echo "[p1] FATAL: invalid HARD_CAP_MINUTES='$HARD_CAP_MINUTES'; terminating now"
  shutdown -h now || true
  exit 1
fi

export AWS_DEFAULT_REGION="$AWS_REGION"

# Set BEFORE terminate_self is defined/trapped: under `set -u` an early
# failure must not hit an unbound $PHASE inside the cleanup trap (which
# would abort the trap before its final self-terminate).
PHASE="P1_extract"

terminate_self() {
  echo "[p1] uploading logs + terminating ($(date -u +%FT%TZ))"
  aws s3 cp "$LOG" "$S3_WORK_PREFIX/progress/$PHASE/${RUN_ID}.bootstrap.log" \
    --region "$AWS_REGION" || true
  IID=$(curl -s --max-time 5 http://169.254.169.254/latest/meta-data/instance-id || true)
  if [[ -n "${IID:-}" ]]; then
    aws ec2 terminate-instances --instance-ids "$IID" \
      --region "$AWS_REGION" || true
  fi
  # Belt and braces: if terminate API failed, still power off (ISB=terminate).
  shutdown -h now || true
}
# MUST self-terminate on ANY exit path (success, error, signal).
trap terminate_self EXIT
trap 'echo "[p1] ERR at line $LINENO"; exit 1' ERR

# ---- 2. Dependencies -------------------------------------------------------
# Assumes a Deep Learning / NVIDIA-driver AMI (CUDA + driver preinstalled).
# We only add Python deps. nvidia-smi is best-effort (CPU still works, slow).
nvidia-smi || echo "[p1] WARNING: nvidia-smi unavailable (CPU fallback, slow)"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y python3-pip awscli ffmpeg libgl1 || true

PIP="python3 -m pip"
$PIP install --upgrade pip >/dev/null 2>&1 || true

# ---- 3. Pull + verify frozen bundle ---------------------------------------
cd "$WORK"
aws s3 cp "$FROZEN_BUNDLE_S3" "$WORK/frozen_detector_v16.tar.gz" --region "$AWS_REGION"
aws s3 cp "$FROZEN_BUNDLE_SHA_S3" "$WORK/frozen.sha256" --region "$AWS_REGION"
EXPECT=$(awk '{print $1}' "$WORK/frozen.sha256")
ACTUAL=$(sha256sum "$WORK/frozen_detector_v16.tar.gz" | awk '{print $1}')
if [[ "$EXPECT" != "$ACTUAL" ]]; then
  echo "[p1] FATAL: bundle sha mismatch expected=$EXPECT actual=$ACTUAL"
  exit 1
fi
echo "[p1] bundle sha OK $ACTUAL"
mkdir -p "$WORK/bundle"
tar -xzf "$WORK/frozen_detector_v16.tar.gz" -C "$WORK/bundle"
BUNDLE_REQ=$(find "$WORK/bundle" -maxdepth 3 -name requirements.txt | head -n1)
if [[ -n "$BUNDLE_REQ" ]]; then
  $PIP install -r "$BUNDLE_REQ"
fi
$PIP install pyarrow boto3

# Reuse the already-verified, already-unpacked bundle from the pipeline code.
export P1_WORK_DIR="$WORK"
mkdir -p "$WORK/p1_work"
cp "$WORK/frozen_detector_v16.tar.gz" "$WORK/p1_work/" 2>/dev/null || true

# ---- 4. Pull staged pipeline code -----------------------------------------
aws s3 cp "$S3_CODE_TARBALL" "$WORK/pipeline.tgz" --region "$AWS_REGION"
mkdir -p "$WORK/repo"
tar -xzf "$WORK/pipeline.tgz" -C "$WORK/repo"

# ---- 5. Run the batch ------------------------------------------------------
export S3_WORK_PREFIX UPLOAD_BUCKET
export SUPABASE_URL SUPABASE_SERVICE_ROLE_KEY
export P1_RUN_ID="$RUN_ID"
export P1_SPOT_USD_PER_HR="$SPOT_USD_PER_HR"
export P1_WORK_DIR="$WORK/p1_work"
export GIT_SHA="$RUN_ID"

cd "$WORK/repo/pipeline" || cd "$WORK/repo"
echo "[p1] starting run_batch $GAME_ARGS"
set +e
python3 run_batch.py $GAME_ARGS --run-id "$RUN_ID" \
  --spot-usd-per-hr "$SPOT_USD_PER_HR"
RC=$?
set -e
echo "[p1] run_batch exited rc=$RC"

aws s3 cp "$LOG" "$S3_WORK_PREFIX/progress/$PHASE/${RUN_ID}.bootstrap.log" \
  --region "$AWS_REGION" || true
echo "DONE rc=$RC" > "$WORK/DONE"
aws s3 cp "$WORK/DONE" "$S3_WORK_PREFIX/progress/$PHASE/${RUN_ID}.DONE" \
  --region "$AWS_REGION" || true

echo "[p1] bootstrap complete $(date -u +%FT%TZ); EXIT trap will terminate"
exit $RC
