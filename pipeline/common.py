"""
Shared helpers for the P1 track-extraction pipeline.

No secrets are stored here; everything sensitive is read from the
environment / a gitignored .env at call time. Pure helpers only — no
network or AWS side effects on import.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Frozen detector bundle (immutable input — never mutated).
FROZEN_BUNDLE_S3 = (
    "s3://uball-cv-results/cv-results/dual-fusion-v2/"
    "frozen_detector_v16/frozen_detector_v16.tar.gz"
)
FROZEN_BUNDLE_SHA_S3 = FROZEN_BUNDLE_S3 + ".sha256"

# Frozen ground-truth windows, pre-exported to S3 (immutable input). The
# GPU box has NO Supabase credentials, so this is the primary GT source.
# Overridable via the GT_WINDOWS_S3 env var. A sibling ".sha256" object
# (first whitespace-delimited field = hex digest) pins the content.
GT_WINDOWS_S3_DEFAULT = (
    "s3://uball-cv-results/cv-results/dual-fusion-v2/gt/gt_windows.json"
)


def gt_windows_s3_uri() -> str:
    return os.environ.get("GT_WINDOWS_S3", GT_WINDOWS_S3_DEFAULT).strip()


def gt_windows_sha_s3_uri() -> str:
    return gt_windows_s3_uri() + ".sha256"

# Video filename angle codes -> which frozen detector weights to use.
# N* (near-left / near-right) use the near model, F* the far model.
ANGLES = ("FL", "FR", "NL", "NR")
ANGLE_TO_DETECTOR = {"NL": "near", "NR": "near", "FL": "far", "FR": "far"}

# v1's EXACT scope (verbatim from v1 config.py). v1 has a quirk: it omits
# "4PT_MAKE" from every list, so the v1 frozen pipeline silently dropped all
# 4PT_MAKE shots. Keep this set ONLY to reproduce the v1 baseline on the
# anchor game apples-to-apples vs 85.7/79.5/89.2.
V1_MAKE_CLASSES = ["3PT_MAKE", "FG_MAKE", "FREE_THROW_MAKE"]
V1_MISS_CLASSES = ["3PT_MISS", "FG_MISS", "FREE_THROW_MISS", "4PT_MISS"]
V1_ALL_SHOT_CLASSES = V1_MAKE_CLASSES + V1_MISS_CLASSES

# DB-confirmed: 4PT_MAKE exists (138 shots, all with valid windows). EXTRACT
# the union (lossless; same videos, no extra cost). classification is stored
# verbatim in the tracks so downstream derives make/miss and a v1_in_scope
# flag without re-running extraction.
EXTRA_MAKE_CLASSES = ["4PT_MAKE"]
SHOT_MAKE_CLASSES = V1_MAKE_CLASSES + EXTRA_MAKE_CLASSES
SHOT_MISS_CLASSES = V1_MISS_CLASSES
ALL_SHOT_CLASSES = SHOT_MAKE_CLASSES + SHOT_MISS_CLASSES  # = extraction scope

# Buffer around GT timestamps (v1 used 2.0 s). Immutable default.
TIMESTAMP_BUFFER_SECONDS = 2.0


def load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal .env loader. Only sets keys that are not already in the
    environment (real env always wins). Never logs values."""
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return os.environ.get("GIT_SHA", "unknown")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    """Run a subprocess, raising with a clear message on failure."""
    proc = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


# ---------------------------------------------------------------------------
# S3 helpers (thin wrappers over the aws CLI so the box needs no boto on it
# beyond what the bundle ships; the laptop only uses these for staging).
# ---------------------------------------------------------------------------

def s3_exists(uri: str) -> bool:
    proc = subprocess.run(
        ["aws", "s3", "ls", uri], text=True, capture_output=True
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def s3_cp(src: str, dst: str, recursive: bool = False) -> None:
    cmd = ["aws", "s3", "cp", src, dst]
    if recursive:
        cmd.append("--recursive")
    run(cmd)


def s3_cat(uri: str) -> str:
    proc = run(["aws", "s3", "cp", uri, "-"])
    return proc.stdout


def parse_s3(uri: str):
    m = re.match(r"^s3://([^/]+)/(.+)$", uri)
    if not m:
        raise ValueError(f"not an s3 uri: {uri}")
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Game:
    game_id: str
    s3_prefix: str
    date: str
    shots: int
    split: str

    @property
    def game23(self) -> str:
        return self.game_id[:23]


def load_manifest(path: Optional[Path] = None) -> List[Game]:
    path = path or (REPO_ROOT / "data" / "games_manifest.json")
    data = json.loads(Path(path).read_text())
    games = []
    for g in data["games"]:
        games.append(Game(
            game_id=g["game_id"],
            s3_prefix=g["s3_prefix"],
            date=g["date"],
            shots=g.get("shots", 0),
            split=g.get("split", "train"),
        ))
    return games


def games_for_split(split: str, path: Optional[Path] = None) -> List[Game]:
    games = load_manifest(path)
    if split == "all":
        return games
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r} (train|val|test|all)")
    return [g for g in games if g.split == split]


def video_s3_uri(game: Game, angle: str, bucket: str) -> str:
    if angle not in ANGLES:
        raise ValueError(f"bad angle {angle!r}")
    fname = f"{game.date}_{game.game23}_{angle}.mp4"
    return f"s3://{bucket}/{game.s3_prefix}{fname}"


# ---------------------------------------------------------------------------
# Ground truth.
#
# PRIMARY: a frozen gt_windows.json pre-exported to S3 (sha256-pinned). The
# GPU box has no Supabase creds, so this path must NOT require them.
# FALLBACK: Supabase PostgREST (public.plays), used ONLY when the S3 GT
# object is absent AND service-role creds exist. Either way the semantics
# mirror v1 gt_loader.py: shot classifications only (ALL_SHOT_CLASSES),
# must have start_timestamp + end_timestamp, sorted by start_timestamp.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GTShot:
    play_id: str
    classification: str
    start_timestamp: float
    end_timestamp: float
    angle: Optional[str]

    def buffered_window(self, buf: float = TIMESTAMP_BUFFER_SECONDS):
        return (max(0.0, self.start_timestamp - buf),
                self.end_timestamp + buf)


def supabase_available() -> bool:
    return bool(os.environ.get("SUPABASE_URL")
                and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))


def _supabase_get(path_qs: str) -> Any:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    req = urllib.request.Request(
        f"{url}/rest/v1/{path_qs}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _gt_cache_path() -> Path:
    """Per-(uri,sha) cache file under the work dir. Keyed by the sha so a
    changed frozen export never serves a stale cached copy."""
    base = Path(os.environ.get("P1_WORK_DIR", tempfile.gettempdir()))
    digest = hashlib.sha256(gt_windows_s3_uri().encode()).hexdigest()[:16]
    return base / f"gt_windows_{digest}.json"


def _filter_rows_to_shots(rows: List[dict]) -> List[GTShot]:
    """v1 gt_loader.py semantics: keep ALL_SHOT_CLASSES only, require both
    timestamps, classification stored verbatim, sorted by start."""
    shots: List[GTShot] = []
    for r in rows:
        cls = r.get("classification")
        if cls not in ALL_SHOT_CLASSES:
            continue
        st = r.get("start_timestamp")
        et = r.get("end_timestamp")
        if st is None or et is None:
            continue
        shots.append(GTShot(
            play_id=str(r.get("play_id", r.get("id"))),
            classification=cls,
            start_timestamp=float(st),
            end_timestamp=float(et),
            angle=r.get("angle"),
        ))
    shots.sort(key=lambda s: s.start_timestamp)
    return shots


def _download_and_verify_gt() -> dict:
    """Download the frozen gt_windows.json (cached by sha), verify its
    sha256 against the sibling .sha256 sidecar, fail hard on mismatch.

    Returns the parsed JSON document. Raises FileNotFoundError if the S3
    object is absent (lets the caller decide on the Supabase fallback),
    RuntimeError on a sha mismatch or malformed sidecar.
    """
    gt_uri = gt_windows_s3_uri()
    sha_uri = gt_windows_sha_s3_uri()
    if not s3_exists(gt_uri):
        raise FileNotFoundError(gt_uri)

    expected = s3_cat(sha_uri).split()[0].strip().lower()
    if not expected or len(expected) != 64:
        raise RuntimeError(
            f"bad sha256 sidecar from {sha_uri!r}: {expected!r}"
        )

    cache = _gt_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or sha256_file(cache) != expected:
        eprint(f"[gt] downloading {gt_uri}")
        s3_cp(gt_uri, str(cache))

    actual = sha256_file(cache)
    if actual != expected:
        raise RuntimeError(
            f"gt_windows.json sha256 mismatch: expected {expected} "
            f"got {actual}. Refusing to run with unverified frozen GT."
        )
    eprint(f"[gt] sha256 OK {actual}")
    return json.loads(cache.read_text())


def _load_gt_shots_supabase(game_id: str) -> List[GTShot]:
    """Fallback: load shot plays from public.plays via PostgREST. Only
    reachable when the frozen S3 GT is absent AND creds exist."""
    rows = _supabase_get(
        "plays?select=id,classification,start_timestamp,end_timestamp,angle"
        f"&game_id=eq.{game_id}"
    )
    return _filter_rows_to_shots(rows)


def load_gt_shots(game_id: str) -> List[GTShot]:
    """Load shot windows for a game.

    1. PRIMARY: frozen gt_windows.json from S3 (GT_WINDOWS_S3 override,
       else GT_WINDOWS_S3_DEFAULT), sha256-verified against its sidecar.
       Does NOT require Supabase credentials.
    2. FALLBACK: Supabase PostgREST, ONLY if the S3 GT object is absent
       AND SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are set.
    3. Otherwise raise a clear error.

    Returned shots use the GTShot dataclass; classification is stored
    verbatim and buffered_window() keeps v1's max(0,start-2)/end+2.
    """
    try:
        doc = _download_and_verify_gt()
    except FileNotFoundError as miss:
        if supabase_available():
            eprint(
                f"[gt] frozen GT absent ({miss}); falling back to Supabase"
            )
            return _load_gt_shots_supabase(game_id)
        raise RuntimeError(
            f"frozen GT not found at {miss} and no Supabase fallback "
            "available (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY unset). "
            "Set GT_WINDOWS_S3 or export gt_windows.json to the default "
            "path."
        ) from miss

    games = doc.get("games")
    if not isinstance(games, dict):
        raise RuntimeError(
            "frozen gt_windows.json missing a 'games' object"
        )
    rows = games.get(game_id)
    if rows is None:
        raise RuntimeError(
            f"game {game_id} not present in frozen gt_windows.json "
            f"({gt_windows_s3_uri()})"
        )
    return _filter_rows_to_shots(rows)


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)
    sys.stderr.flush()
