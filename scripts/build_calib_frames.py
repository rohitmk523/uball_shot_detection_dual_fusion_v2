#!/usr/bin/env python3
"""Build a numbered NEAR|FAR side-by-side calibration set for ONE made free throw.

Free throw = fixed, known geometry (15 ft line, centered; rim 18 in @ 10 ft;
men's size-7 ball = 9.39 in / 0.2385 m). User annotates real-world measurements
per frame; we tie each frame number to the detector's pixel boxes (detections.csv)
to calibrate the near camera (intrinsics + pose -> true 3D depth).

Outputs to data/client_report/calib_freethrow/:
  calib_freethrow_sidebyside.mp4   scrubbable, frame counter burned in
  frames/frame_000.jpg .. 149.jpg  numbered NEAR|FAR stills
  detections.csv                   clip_frame -> timestamp + near/far ball+rim px
  README.md                        how to annotate
Frames pulled from S3 via presigned-URL range seeks (no full download). CPU.
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "client_report" / "calib_freethrow"
FRAMES = OUT / "frames"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"

# chosen calibration shot
GID = "b3c1f62c-1a02-47c9-8d2a-4e05a27dc14d"
PLAY = "954c1708-1eb5-4e44-a879-6022969bfbb2"
SIDE = "RIGHT"
NEAR, FAR = "NR", "FR"
START = 203.8                  # plays.start_timestamp
FPS = 30000 / 1001            # 29.97
NF = 150
DUR = NF / FPS                 # ~5.005 s
PW, PH = 1280, 720            # per-panel size


def uri(cam):
    g = next(x for x in json.loads((ROOT / "data" / "games_manifest.json").read_text())["games"]
             if x["game_id"] == GID)
    return f"s3://uball-videos-production/{g['s3_prefix']}{g['date']}_{GID[:23]}_{cam}.mp4"


def presign(u):
    return subprocess.run(["aws", "s3", "presign", u, "--expires-in", "10800"],
                          capture_output=True, text=True, check=True).stdout.strip()


def filtergraph():
    title = (f"FREE THROW MAKE  game {GID[:8]}  {SIDE} side   "
             f"ball=men size7 (9.39in)  rim=18in @ 10ft")
    return (
        f"[0:v]scale={PW}:{PH},drawtext=fontfile={FONT}:text='NEAR {NEAR}':"
        f"x=18:y=14:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[a];"
        f"[1:v]scale={PW}:{PH},drawtext=fontfile={FONT}:text='FAR {FAR}':"
        f"x=18:y=14:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[b];"
        f"[a][b]hstack=inputs=2,pad=iw:ih+64:0:64:black[s];"
        f"[s]drawtext=fontfile={FONT}:text='{title}':x=18:y=18:fontsize=26:"
        f"fontcolor=white,"
        f"drawtext=fontfile={FONT}:text='FRAME %{{n}} / {NF-1}':"
        f"x=w-380:y=14:fontsize=34:fontcolor=lime:box=1:boxcolor=black@0.6[v]"
    )


def detections_csv():
    df = pd.read_parquet(Path("/tmp/p1tracks_fresh") / f"{GID}.parquet")
    df = df[df.play_id == PLAY]
    def slice_cam(cam):
        s = df[df.angle == cam].sort_values("timestamp")
        return s
    sn, sf = slice_cam(NEAR), slice_cam(FAR)
    rows = []
    for n in range(NF):
        t = START + n / FPS
        rec = {"clip_frame": n, "video_time_s": round(t, 4)}
        for tag, s in (("near", sn), ("far", sf)):
            if len(s):
                i = (s.timestamp - t).abs().idxmin()
                r = s.loc[i]
                if abs(float(r.timestamp) - t) < 0.05:      # within ~1 frame
                    for col in ("ball_x", "ball_y", "ball_w", "ball_h", "ball_conf",
                                "rim_x", "rim_y", "rim_w", "rim_h"):
                        v = r[col]
                        rec[f"{tag}_{col}"] = (round(float(v), 1)
                                               if pd.notna(v) else "")
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "detections.csv", index=False)


def main():
    FRAMES.mkdir(parents=True, exist_ok=True)
    nu, fu = presign(uri(NEAR)), presign(uri(FAR))
    fg = filtergraph()
    common = ["-ss", f"{START:.3f}", "-i", nu, "-ss", f"{START:.3f}", "-i", fu,
              "-t", f"{DUR:.3f}", "-filter_complex", fg, "-map", "[v]"]
    print("[calib] rendering side-by-side video ...")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *common,
                    "-r", f"{FPS}", "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", str(OUT / "calib_freethrow_sidebyside.mp4")],
                   check=True)
    print("[calib] writing 150 numbered frames ...")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *common,
                    "-r", f"{FPS}", "-frames:v", str(NF), "-start_number", "0",
                    "-q:v", "2", str(FRAMES / "frame_%03d.jpg")], check=True)
    print("[calib] writing detections.csv ...")
    detections_csv()
    (OUT / "README.md").write_text(README)
    n = len(list(FRAMES.glob('*.jpg')))
    print(f"[calib] done: {n} frames + video + detections.csv in {OUT}")


README = f"""# Free-throw calibration set — annotate these to calibrate the NEAR camera

Shot: made free throw, game {GID[:8]}, {SIDE} side. NEAR={NEAR}, FAR={FAR}.
150 frames @ {FPS:.2f} fps (clip frame 0 = video time {START}s). Each side-by-side
frame is numbered (top-right). Left = NEAR camera, right = FAR camera.

## Known constants (please confirm)
- Ball: men's size 7 = 9.39 in (0.2385 m) diameter.
- Rim: 18 in (0.4572 m) inner diameter, top at 10 ft (3.048 m).
- Free-throw line: 15 ft (4.572 m) from the backboard.

## What helps most (only you can give)
For the NEAR camera, the mounting geometry:
  1. height of the NEAR camera above the floor (ft or m)
  2. horizontal distance from the camera to the hoop (ft or m)
  3. downward tilt angle of the camera (degrees), if known
Plus the GoPro model + recording mode (e.g., HERO12, Linear, 1080/4K, 30fps).

## How to annotate (any frames you can)
Pick a handful of clear frames and note, per frame number:
  - ball's real height above the floor (your best estimate), and/or
  - any frame where the ball is exactly AT the rim level (passing the hoop),
  - the release frame and the through-net frame.
`detections.csv` already lists, per clip_frame, the detector's ball & rim pixel
boxes (x,y,w,h) for both cameras — so your frame-referenced notes map straight
to pixels and we can fit pixel-size <-> real-distance/height for this camera.
"""


if __name__ == "__main__":
    main()
