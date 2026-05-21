#!/usr/bin/env python3
"""
P1c rim-region crop batch driver.

Runs extract_crops for a list of games SEQUENTIALLY and records progress,
mirroring run_netmotion_batch.py exactly:

  * If SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set, writes a
    dual_fusion_v2.iterations row (phase=P1c_crops, status running->
    done/failed) and updates dual_fusion_v2.games
    (crops_extracted_at / crops_s3_key) via the PostgREST API.
  * Tracking is best-effort: a missing service key, a missing
    dual_fusion_v2 schema, or any PostgREST error must NOT fail the
    job. In that case progress is written to S3 instead.

Idempotent: a game whose crops_meta.json already exists is skipped
(unless --force).

Usage:
  python run_crops_batch.py --split train|val|test|all
  python run_crops_batch.py --games <id1> <id2> ...
  python run_crops_batch.py --split all --redetect-conf 0.10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

from common import (
    Game, eprint, games_for_split, git_sha, load_manifest, run as _run,
    supabase_available,
)
from extract_crops import extract_game

SCHEMA = "dual_fusion_v2"
PHASE = "P1c_crops"
# Conservative spot price for cost bookkeeping when the real price is
# unknown on the box (laptop launcher passes the true number via env).
DEFAULT_SPOT_USD_PER_HR = float(os.environ.get("P1_SPOT_USD_PER_HR", "0.24"))


def _progress_s3_uri(run_id: str) -> str:
    base = os.environ.get(
        "S3_WORK_PREFIX", "s3://uball-cv-results/cv-results/dual-fusion-v2"
    ).rstrip("/")
    return f"{base}/progress/{PHASE}/{run_id}.json"


def _write_progress_s3(run_id: str, payload: dict) -> None:
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            json.dump(payload, fh, indent=2)
            tmp = fh.name
        _run(["aws", "s3", "cp", tmp, _progress_s3_uri(run_id)])
        os.unlink(tmp)
    except Exception as e:  # never fail the job on a progress write
        eprint(f"[batch] WARNING: could not write progress to S3: {e}")


# ---------------------------------------------------------------------------
# PostgREST helpers (schema dual_fusion_v2). Best-effort only.
# ---------------------------------------------------------------------------

def _pgrest(method: str, path: str, body: Optional[dict] = None,
            prefer: Optional[str] = None):
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{url}/rest/v1/{path}", data=data, method=method,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Profile": SCHEMA,
            "Accept-Profile": SCHEMA,
            **({"Prefer": prefer} if prefer else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw.strip() else None


def _tracking_enabled() -> bool:
    return supabase_available()


def _safe(fn, what: str):
    """Run a tracking call, swallow + log any error (best-effort)."""
    try:
        return fn()
    except urllib.error.HTTPError as e:
        eprint(f"[batch] tracking '{what}' HTTP {e.code}: "
               f"{e.read().decode()[:300]} (continuing)")
    except Exception as e:
        eprint(f"[batch] tracking '{what}' failed: {e} (continuing)")
    return None


def _iteration_create(run_id: str, games: List[str]):
    return _safe(lambda: _pgrest(
        "POST", "iterations",
        {
            "phase": PHASE,
            "status": "running",
            "git_sha": git_sha(),
            "run_id": run_id,
            "games": games,
            "cost_usd": 0.0,
            "s3_artifacts": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        prefer="return=representation",
    ), "iteration_create")


def _iteration_finish(run_id: str, status: str, cost_usd: float,
                      artifacts: dict):
    return _safe(lambda: _pgrest(
        "PATCH", f"iterations?run_id=eq.{run_id}&phase=eq.{PHASE}",
        {
            "status": status,
            "cost_usd": round(cost_usd, 4),
            "s3_artifacts": artifacts,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        prefer="return=minimal",
    ), "iteration_finish")


def _game_mark_extracted(game_id: str, crops_key: str):
    return _safe(lambda: _pgrest(
        "PATCH", f"games?id=eq.{game_id}",
        {
            "crops_extracted_at": datetime.now(timezone.utc).isoformat(),
            "crops_s3_key": crops_key,
        },
        prefer="return=minimal",
    ), "game_mark_extracted")


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _resolve_games(args) -> List[Game]:
    if args.games:
        by_id = {g.game_id: g for g in load_manifest()}
        missing = [g for g in args.games if g not in by_id]
        if missing:
            raise SystemExit(f"unknown game ids: {missing}")
        return [by_id[g] for g in args.games]
    return games_for_split(args.split)


def main(argv: Optional[List[str]] = None) -> int:
    from common import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="P1c batch rim-region crop extraction"
    )
    ap.add_argument("--split", choices=["train", "val", "test", "all"])
    ap.add_argument("--games", nargs="*", help="explicit game ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--redetect-conf", type=float, default=None,
        help="if set, also write denser ball boxes per game (best-effort)",
    )
    ap.add_argument(
        "--run-id",
        default=os.environ.get("P1_RUN_ID")
        or datetime.now(timezone.utc).strftime("p1c-%Y%m%dT%H%M%SZ"),
    )
    ap.add_argument("--spot-usd-per-hr", type=float,
                    default=DEFAULT_SPOT_USD_PER_HR)
    args = ap.parse_args(argv)

    if not args.games and not args.split:
        raise SystemExit("provide --split or --games")

    games = _resolve_games(args)
    run_id = args.run_id
    started = datetime.now(timezone.utc)
    eprint(f"[batch] run {run_id}: {len(games)} game(s), tracking="
           f"{'supabase' if _tracking_enabled() else 's3-only'}")

    tracking = _tracking_enabled()
    if tracking:
        _iteration_create(run_id, [g.game_id for g in games])

    results: List[dict] = []
    artifacts: Dict[str, str] = {}
    overall = "done"

    for i, g in enumerate(games, 1):
        eprint(f"[batch] ({i}/{len(games)}) game {g.game_id}")
        rec = {"game_id": g.game_id, "split": g.split}
        try:
            res = extract_game(
                g, force=args.force, redetect_conf=args.redetect_conf)
            rec.update(status=res["status"])
            if res.get("crops_s3_key"):
                artifacts[g.game_id] = res["crops_s3_key"]
                rec["crops_s3_key"] = res["crops_s3_key"]
                if tracking and res["status"] == "done":
                    _game_mark_extracted(g.game_id, res["crops_s3_key"])
        except Exception as e:
            overall = "failed"
            rec.update(status="failed", error=str(e)[:500])
            eprint(f"[batch] game {g.game_id} FAILED: {e}")
        results.append(rec)
        # Persist incremental progress to S3 every game (cheap, resumable).
        _write_progress_s3(run_id, {
            "run_id": run_id, "phase": PHASE, "git_sha": git_sha(),
            "started_at": started.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "running", "results": results,
        })

    elapsed_hr = (
        datetime.now(timezone.utc) - started
    ).total_seconds() / 3600.0
    cost_usd = round(elapsed_hr * args.spot_usd_per_hr, 4)

    final = {
        "run_id": run_id, "phase": PHASE, "git_sha": git_sha(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": round(elapsed_hr, 4),
        "spot_usd_per_hr": args.spot_usd_per_hr,
        "cost_usd": cost_usd,
        "status": overall, "results": results, "s3_artifacts": artifacts,
    }
    _write_progress_s3(run_id, final)
    if tracking:
        _iteration_finish(run_id, overall, cost_usd, artifacts)

    print(json.dumps(final, indent=2))
    return 0 if overall == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
