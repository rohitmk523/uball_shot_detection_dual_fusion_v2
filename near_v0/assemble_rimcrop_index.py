#!/usr/bin/env python3
"""Phase 1 final step: merge per-game rim-crop metas into one dataset index.

- Frozen-band scan per game/angle: a run of >=3 consecutive (by t1) clips with
  peak_energy < 3.0 marks a camera fault band -> unusable. Isolated lows are
  kept and flagged low_energy (no_event candidates -- taxonomy section 4).
- Emits data/near_rimcrop/dataset_index.json with per-clip records and
  whole-game LOGO fold list. Frozen test games are absent by construction.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RC = REPO / "data/near_rimcrop"
LOW_E = 3.0
BAND_MIN = 3


def main():
    records = []
    for meta_path in sorted(RC.glob("*/meta.json")):
        ga = meta_path.parent.name
        clips = json.loads(meta_path.read_text())
        clips.sort(key=lambda c: c["t1_game"])
        # frozen-band scan
        low_flags = [c.get("peak_energy", 99) < LOW_E for c in clips]
        in_band = [False] * len(clips)
        i = 0
        while i < len(clips):
            if low_flags[i]:
                j = i
                while j < len(clips) and low_flags[j]:
                    j += 1
                if j - i >= BAND_MIN:
                    for k in range(i, j):
                        in_band[k] = True
                i = j
            else:
                i += 1
        for c, band in zip(clips, in_band):
            mp4 = meta_path.parent / f"{c['name']}.mp4"
            usable = mp4.exists() and not band and not c.get("unusable")
            flags = []
            if band or c.get("unusable"):
                flags.append("camera_fault_band")
            elif c.get("peak_energy", 99) < LOW_E:
                flags.append("low_energy_no_event_candidate")
            records.append({
                "path": str(mp4.relative_to(REPO)), "ga": ga,
                "game": c["gid8"], "cam": c["cam"], "pid8": c["pid8"],
                "gt": c["gt"], "make": c["make"], "usable": usable,
                "flags": flags, "peak_energy": c.get("peak_energy"),
                "n_frames": c.get("n_frames"), "t1_game": c["t1_game"],
            })

    usable = [r for r in records if r["usable"]]
    games = sorted({r["game"] for r in usable})
    index = {"records": records, "logo_games": games,
             "frozen_test_pool": ["c2a354fe", "6d601c99", "ee8745f1",
                                  "0fa23810", "49b3873e", "e74164e6"]}
    (RC / "dataset_index.json").write_text(json.dumps(index, indent=0))

    print(f"total={len(records)} usable={len(usable)} "
          f"games={len(games)} (LOGO folds)")
    print("class balance (usable):", dict(Counter(
        "MAKE" if r["make"] else "MISS" for r in usable)))
    per_game = defaultdict(Counter)
    for r in records:
        per_game[r["game"]][("ok" if r["usable"] else "bad")] += 1
    for g in sorted(per_game):
        c = per_game[g]
        print(f"  {g}: usable={c['ok']} unusable={c['bad']}")


if __name__ == "__main__":
    main()
