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
    """Put the bundle on sys.path and import the v1 modules we reuse."""
    bd = str(bundle_dir)
    if bd not in sys.path:
        sys.path.insert(0, bd)
    # Run from the bundle dir so v1's relative weights_config/ lookups work.
    os.chdir(bundle_dir)
    import config as v1_config  # noqa: E402
    from video_processor import VideoProcessor  # noqa: E402
    from enhanced_shot_detector import EnhancedShotDetector  # noqa: E402
    return v1_config, VideoProcessor, EnhancedShotDetector


def weight_paths(bundle_dir: Path) -> dict:
    near = bundle_dir / "weights/near_angle_weights/basketball_yolo11n3/weights/best.pt"
    far = bundle_dir / "weights/far_angle_weights/basketball_yolo11n2/weights/best.pt"
    for p in (near, far):
        if not p.exists():
            raise RuntimeError(f"missing frozen weight: {p}")
    return {"near": str(near), "far": str(far)}
