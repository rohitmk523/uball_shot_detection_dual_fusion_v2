#!/usr/bin/env bash
# =============================================================================
# P1 launcher (runs on the laptop). Reads .env, enforces a HARD BUDGET GUARD,
# stages pipeline/ to S3, then launches ONE g4dn.xlarge spot instance whose
# user-data self-terminates at the budget cap regardless of job state.
#
# SAFETY MODEL (human-audited):
#   1. spot_price        = current lowest g4dn.xlarge spot price (this region).
#   2. budget_price      = max(spot_price, FLOOR)   # never divide by a tiny
#                                                   #  or zero price
#   3. cap_hours         = AWS_MAX_USD_PER_ITERATION / budget_price
#   4. HARD_CAP_MINUTES  = floor(cap_hours * 60)     # ALWAYS rounded DOWN
#   5. worst_case_usd    = (HARD_CAP_MINUTES / 60) * ON_DEMAND_PRICE
#                          # priced at on-demand (= our spot max-price), the
#                          # most we can possibly be charged for this run.
#   6. Refuse to launch if  cumulative_iteration_cost + worst_case_usd
#                           > AWS_MAX_USD_TOTAL.
#   7. The instance is launched with
#        --instance-initiated-shutdown-behavior terminate
#        --instance-market-options spot, max-price = ON_DEMAND_PRICE
#      and bootstrap.sh arms `shutdown -h +HARD_CAP_MINUTES` BEFORE any work.
#
# Rounding DOWN at step 4 + pricing worst case at ON-DEMAND at step 5 makes
# the guard strictly conservative: the true bill is always <= worst_case_usd.
#
# --dry-run does everything EXCEPT the run-instances call.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
SPLIT=""
GAME_ARGS=""
# Default ON-DEMAND: spot reclaim lost ~48min of correct work and stalls
# autonomous mode. On-demand has no reclaim. --spot opts back in.
USE_ONDEMAND=1
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --spot) USE_ONDEMAND=0 ;;
    --on-demand) USE_ONDEMAND=1 ;;
    --split=*) SPLIT="${a#*=}"; GAME_ARGS="--split ${a#*=}" ;;
    --games=*) GAME_ARGS="--games ${a#*=}" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
[[ -n "$GAME_ARGS" ]] || { echo "usage: launch_p1.sh --split=train|val|test|all [--on-demand|--spot] [--dry-run]" >&2; exit 2; }

# ---- Load .env (gitignored; never echoed) ----------------------------------
ENV_FILE="$REPO_ROOT/.env"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: $ENV_FILE not found" >&2; exit 1; }
# macOS ships bash 3.2 where `set -a; source <(...)` silently fails to
# export. Parse line-by-line and export without eval (value-safe).
while IFS= read -r _line || [[ -n "$_line" ]]; do
  case "$_line" in ''|\#*) continue ;; esac
  [[ "$_line" == *=* ]] || continue
  _k="${_line%%=*}"
  [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
  _v="${_line#*=}"
  _v="${_v%\"}"; _v="${_v#\"}"; _v="${_v%\'}"; _v="${_v#\'}"
  export "$_k=$_v"
done < "$ENV_FILE"
unset _line _k _v

: "${AWS_DEFAULT_REGION:?set in .env}"
: "${AWS_MAX_USD_PER_ITERATION:?}"
: "${AWS_MAX_USD_TOTAL:?}"
: "${S3_WORK_PREFIX:?}"
: "${UPLOAD_BUCKET:?}"
: "${AWS_INSTANCE_PROFILE:?}"
AWS_GPU_INSTANCE_TYPE="${AWS_GPU_INSTANCE_TYPE:-g4dn.xlarge}"
REGION="$AWS_DEFAULT_REGION"

# Documented proven recipe (04_AWS_RUNTIME.md). Overridable via env.
AMI_ID="${P1_AMI_ID:-ami-0cff16a9a375c81a5}"
SEC_GROUP="${P1_SECURITY_GROUP:-sg-d1ad2a8c}"
ROOT_GB="${P1_ROOT_GB:-80}"
# us-east-1 g4dn on-demand list price. Used as spot max-price AND as the
# conservative worst-case unit cost. Override only with a higher number.
ON_DEMAND_PRICE="${P1_ONDEMAND_USD_PER_HR:-0.526}"
# Never divide the per-iteration budget by an unrealistically small price.
SPOT_FLOOR="${P1_SPOT_FLOOR_USD_PER_HR:-0.15}"

RUN_ID="${P1_RUN_ID:-p1-$(date -u +%Y%m%dT%H%M%SZ)}"
PHASE="P1_extract"

awsbin() { aws --region "$REGION" "$@"; }

# ---- 1. Current spot price -------------------------------------------------
echo "[guard] querying current spot price for $AWS_GPU_INSTANCE_TYPE in $REGION ..."
SPOT_PRICE=$(awsbin ec2 describe-spot-price-history \
  --instance-types "$AWS_GPU_INSTANCE_TYPE" \
  --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u +%FT%TZ)" \
  --query 'SpotPriceHistory[0].SpotPrice' --output text 2>/dev/null || echo "")

if [[ -z "$SPOT_PRICE" || "$SPOT_PRICE" == "None" ]]; then
  echo "[guard] no live spot price; falling back to on-demand price (conservative)"
  SPOT_PRICE="$ON_DEMAND_PRICE"
fi

# ---- 2..5 Budget math (all rounding DOWN; awk for float math) --------------
read -r BUDGET_PRICE HARD_CAP_MINUTES WORST_CASE_USD <<EOF
$(awk -v spot="$SPOT_PRICE" -v floor="$SPOT_FLOOR" \
       -v per="$AWS_MAX_USD_PER_ITERATION" -v od="$ON_DEMAND_PRICE" 'BEGIN {
  bp = (spot > floor) ? spot : floor;          # reported expected price
  # Size the HARD cap off the on-demand CEILING (= our spot max-price),
  # NOT the live spot price, so worst-case bill <= per-iteration budget
  # BY CONSTRUCTION even if spot rises to the ceiling mid-run.
  cap_hours = per / od;                        # ceiling-priced affordable hrs
  cap_min  = int(cap_hours * 60);              # floor() -> round DOWN
  if (cap_min < 1) cap_min = 0;                # 0 => caller will refuse
  worst = (cap_min / 60.0) * od;               # <= per by construction
  printf "%.4f %d %.4f", bp, cap_min, worst;
}')
EOF

# ---- 6. Cumulative-cost ceiling check --------------------------------------
CUM_COST="0"
if [[ -n "${SUPABASE_URL:-}" && -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]]; then
  CUM_JSON=$(curl -s --max-time 20 \
    -H "apikey: $SUPABASE_SERVICE_ROLE_KEY" \
    -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
    -H "Accept-Profile: dual_fusion_v2" \
    "$SUPABASE_URL/rest/v1/iterations?select=cost_usd" || echo "")
  if [[ -n "$CUM_JSON" && "$CUM_JSON" == \[* ]]; then
    CUM_COST=$(printf '%s' "$CUM_JSON" \
      | grep -oE '"cost_usd":[ ]*[0-9.]+' \
      | grep -oE '[0-9.]+' | awk '{s+=$1} END {printf "%.4f", s+0}')
  else
    echo "[guard] WARNING: could not read cumulative cost from Supabase;"
    echo "[guard]          treating prior spend as 0 (verify manually)."
  fi
else
  echo "[guard] WARNING: no Supabase service key; cannot verify cumulative"
  echo "[guard]          spend. Treating prior spend as 0 (verify manually)."
fi

PROJECTED=$(awk -v c="$CUM_COST" -v w="$WORST_CASE_USD" 'BEGIN{printf "%.4f", c+w}')

cat <<PLAN

================ P1 LAUNCH PLAN ================
 run_id                : $RUN_ID
 region / instance     : $REGION / $AWS_GPU_INSTANCE_TYPE (spot)
 AMI / SG / root       : $AMI_ID / $SEC_GROUP / ${ROOT_GB}GB
 instance profile      : $AWS_INSTANCE_PROFILE
 games                 : $GAME_ARGS
 ---- budget guard ----
 live spot price       : \$$SPOT_PRICE /hr
 spot floor            : \$$SPOT_FLOOR /hr
 budget price (used)   : \$$BUDGET_PRICE /hr
 per-iteration cap     : \$$AWS_MAX_USD_PER_ITERATION
 HARD shutdown cap     : $HARD_CAP_MINUTES min  (= floor of cap, hard kill)
 spot max-price        : \$$ON_DEMAND_PRICE /hr (= on-demand)
 worst-case this run   : \$$WORST_CASE_USD   (cap_min/60 * on-demand)
 prior cumulative      : \$$CUM_COST
 projected total       : \$$PROJECTED  (ceiling \$$AWS_MAX_USD_TOTAL)
================================================
PLAN

# Refusals (fail closed).
if [[ "$HARD_CAP_MINUTES" -lt 1 ]]; then
  echo "REFUSING: computed hard cap is 0 min (per-iteration budget too small" >&2
  echo "          relative to price \$$BUDGET_PRICE/hr)." >&2
  exit 1
fi
if awk -v w="$WORST_CASE_USD" -v per="$AWS_MAX_USD_PER_ITERATION" \
       'BEGIN{exit !(w>per)}'; then
  echo "REFUSING: worst-case \$$WORST_CASE_USD exceeds per-iteration cap \$$AWS_MAX_USD_PER_ITERATION." >&2
  exit 1
fi
if awk -v p="$PROJECTED" -v t="$AWS_MAX_USD_TOTAL" 'BEGIN{exit !(p>t)}'; then
  echo "REFUSING: projected total \$$PROJECTED exceeds AWS_MAX_USD_TOTAL \$$AWS_MAX_USD_TOTAL." >&2
  exit 1
fi

# ---- 7. Stage pipeline/ + manifest to S3 ----------------------------------
# common.py resolves the manifest at <repo>/data/games_manifest.json
# (REPO_ROOT = pipeline/..). The tarball MUST carry data/ too or run_batch
# fails with FileNotFoundError before any extraction.
STAGE_TGZ="/tmp/${RUN_ID}_pipeline.tgz"
[[ -f "$REPO_ROOT/data/games_manifest.json" ]] || {
  echo "FATAL: data/games_manifest.json missing; cannot stage" >&2; exit 1; }
tar -czf "$STAGE_TGZ" -C "$REPO_ROOT" pipeline data/games_manifest.json
S3_CODE_TARBALL="$S3_WORK_PREFIX/jobs/$RUN_ID/pipeline.tgz"
echo "[stage] uploading pipeline/ -> $S3_CODE_TARBALL"
awsbin s3 cp "$STAGE_TGZ" "$S3_CODE_TARBALL"

FROZEN_BUNDLE_S3="s3://uball-cv-results/cv-results/dual-fusion-v2/frozen_detector_v16/frozen_detector_v16.tar.gz"
FROZEN_BUNDLE_SHA_S3="${FROZEN_BUNDLE_S3}.sha256"

# ---- Render user-data from bootstrap.sh ------------------------------------
UD="/tmp/${RUN_ID}_userdata.sh"
cp "$SCRIPT_DIR/bootstrap.sh" "$UD"
subst() { python3 - "$UD" "$1" "$2" <<'PY'
import sys, io
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
s = io.open(path, encoding="utf-8").read().replace("__%s__" % key, val)
io.open(path, "w", encoding="utf-8").write(s)
PY
}
subst S3_WORK_PREFIX            "$S3_WORK_PREFIX"
subst S3_CODE_TARBALL           "$S3_CODE_TARBALL"
subst FROZEN_BUNDLE_S3          "$FROZEN_BUNDLE_S3"
subst FROZEN_BUNDLE_SHA_S3      "$FROZEN_BUNDLE_SHA_S3"
subst RUN_ID                    "$RUN_ID"
subst SPLIT                     "${SPLIT:-na}"
subst GAME_ARGS                 "$GAME_ARGS"
subst HARD_CAP_MINUTES          "$HARD_CAP_MINUTES"
subst SPOT_USD_PER_HR           "$BUDGET_PRICE"
subst SUPABASE_URL              "${SUPABASE_URL:-}"
subst SUPABASE_SERVICE_ROLE_KEY "${SUPABASE_SERVICE_ROLE_KEY:-}"
subst UPLOAD_BUCKET             "$UPLOAD_BUCKET"
subst AWS_REGION                "$REGION"

echo "[stage] user-data rendered ($(wc -c <"$UD") bytes)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "[dry-run] All preflight + staging done. NOT calling run-instances."
  echo "[dry-run] would launch with hard cap $HARD_CAP_MINUTES min,"
  echo "[dry-run] spot max-price \$$ON_DEMAND_PRICE, ISB=terminate."
  rm -f "$UD"
  exit 0
fi

# ---- 7. Launch (self-terminating) -----------------------------------------
# Worst-case is already priced at on-demand, so on-demand stays within the
# same per-iteration ceiling while removing reclaim risk.
if [[ "$USE_ONDEMAND" -eq 1 ]]; then
  MARKET_OPTS=()
  echo "[launch] requesting ON-DEMAND $AWS_GPU_INSTANCE_TYPE (no reclaim) ..."
else
  MARKET_OPTS=(--instance-market-options
    "MarketType=spot,SpotOptions={MaxPrice=$ON_DEMAND_PRICE,SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}")
  echo "[launch] requesting SPOT $AWS_GPU_INSTANCE_TYPE (reclaim-risk) ..."
fi
INSTANCE_ID=$(awsbin ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$AWS_GPU_INSTANCE_TYPE" \
  --security-group-ids "$SEC_GROUP" \
  --associate-public-ip-address \
  --iam-instance-profile "Name=$AWS_INSTANCE_PROFILE" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3}" \
  ${MARKET_OPTS[@]+"${MARKET_OPTS[@]}"} \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Project,Value=dual-fusion-v2},{Key=Phase,Value=$PHASE},{Key=RunId,Value=$RUN_ID}]" \
  --user-data "file://$UD" \
  --query 'Instances[0].InstanceId' --output text)

echo "[launch] instance $INSTANCE_ID launched ($([[ $USE_ONDEMAND -eq 1 ]] && echo on-demand || echo spot), hard cap ${HARD_CAP_MINUTES}m)."
echo "[launch] monitor: aws ec2 describe-instances --filters Name=tag:RunId,Values=$RUN_ID"
echo "[launch] progress: aws s3 ls $S3_WORK_PREFIX/progress/$PHASE/"
rm -f "$UD"
