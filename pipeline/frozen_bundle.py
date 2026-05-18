"""
Fetch + verify + load the frozen v1 detector bundle.

The bundle is an immutable input. We download it once, verify its sha256
against the sibling .sha256 S3 object, unpack it to a fixed local dir, and
put it on sys.path so the v1 detection code can be imported AS-IS. We never
reimplement YOLO or mutate the bundle.
"""
from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path
from typing import Tuple

from common import (
    FROZEN_BUNDLE_S3, FROZEN_BUNDLE_SHA_S3, eprint, s3_cat, s3_cp,
    sha256_file,
)

WORK_DIR = Path(os.environ.get("P1_WORK_DIR", "/tmp/p1_work"))
BUNDLE_TGZ = WORK_DIR / "frozen_detector_v16.tar.gz"
BUNDLE_DIR = WORK_DIR / "frozen_detector_v16"


def fetch_and_verify_bundle() -> Tuple[Path, str]:
    """Download (idempotent), verify sha256, unpack. Returns (dir, sha256)."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    expected = s3_cat(FROZEN_BUNDLE_SHA_S3).split()[0].strip().lower()
    if not expected or len(expected) != 64:
        raise RuntimeError(f"bad sha256 from {FROZEN_BUNDLE_SHA_S3!r}: {expected!r}")

    if not BUNDLE_TGZ.exists() or sha256_file(BUNDLE_TGZ) != expected:
        eprint(f"[bundle] downloading {FROZEN_BUNDLE_S3}")
        s3_cp(FROZEN_BUNDLE_S3, str(BUNDLE_TGZ))

    actual = sha256_file(BUNDLE_TGZ)
    if actual != expected:
        raise RuntimeError(
            f"bundle sha256 mismatch: expected {expected} got {actual}. "
            "Refusing to run with an unverified detector."
        )
    eprint(f"[bundle] sha256 OK {actual}")

    if not BUNDLE_DIR.exists():
        tmp = WORK_DIR / "_unpack"
        tmp.mkdir(exist_ok=True)
        with tarfile.open(BUNDLE_TGZ, "r:gz") as tf:
            tf.extractall(tmp)  # noqa: S202 - trusted, sha-verified artifact
        root = _find_bundle_root(tmp)
        root.rename(BUNDLE_DIR)
    return BUNDLE_DIR, actual


def _find_bundle_root(extracted: Path) -> Path:
    """Locate the dir that contains enhanced_shot_detector.py."""
    if (extracted / "enhanced_shot_detector.py").exists():
        return extracted
    for p in extracted.rglob("enhanced_shot_detector.py"):
        return p.parent
    raise RuntimeError("enhanced_shot_detector.py not found in bundle")


def import_v1(bundle_dir: Path):
    """Put the bundle on sys.path and import the REAL v1 modules we reuse,
    AS-IS. Returns (v1_config, VideoProcessor, EnhancedShotDetector).

    No os.chdir: every v1 path lookup that matters is __file__-relative
    (config.py uses Path(__file__).parent for INPUT/WEIGHTS/etc., and
    WeightedFeatureScorer loads weights_config/ via
    Path(__file__).parent / "weights_config" / ...). Changing the process
    cwd would be a hidden global side effect with no upside, so we don't.
    """
    bd = str(bundle_dir.resolve())
    if not (bundle_dir / "enhanced_shot_detector.py").exists():
        raise RuntimeError(
            f"bundle dir {bd} does not contain enhanced_shot_detector.py"
        )
    if bd not in sys.path:
        sys.path.insert(0, bd)

    import importlib

    v1_config = importlib.import_module("config")
    video_processor = importlib.import_module("video_processor")
    esd = importlib.import_module("enhanced_shot_detector")

    # Real v1 names — fail loudly here rather than at first use if the
    # bundle ever drifts from the expected v1 API.
    try:
        VideoProcessor = video_processor.VideoProcessor
        EnhancedShotDetector = esd.EnhancedShotDetector
    except AttributeError as e:
        raise RuntimeError(
            f"frozen bundle does not expose the expected v1 API: {e}"
        ) from e
    return v1_config, VideoProcessor, EnhancedShotDetector


def weight_paths(bundle_dir: Path) -> dict:
    near = bundle_dir / "weights/near_angle_weights/basketball_yolo11n3/weights/best.pt"
    far = bundle_dir / "weights/far_angle_weights/basketball_yolo11n2/weights/best.pt"
    for p in (near, far):
        if not p.exists():
            raise RuntimeError(f"missing frozen weight: {p}")
    return {"near": str(near), "far": str(far)}
