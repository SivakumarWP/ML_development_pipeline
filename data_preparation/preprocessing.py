#!/usr/bin/env python3
"""
cardsense_2 data preparation (simple version)

Assumes folder structure:
  blank/blank_0.png, blank_1.png, ...
  written/written_0_C0.png, written_1_C2.png, ...
  digital/0.gcode, 1.gcode, ... (digital images optional)

Outputs:
  data/processed/ML_processed_image/{idx}.png   (diff image)
  data/processed/metadata.csv                   (metadata table)
"""

from pathlib import Path
import cv2, csv, re
import numpy as np

# ====== CONFIGURE THESE PATHS ======
BLANK_DIR   = Path("data/raw/blank")
WRITTEN_DIR = Path("data/raw/written")
DIGITAL_DIR = Path("data/raw/digital")

OUT_IMAGES  = Path("data/processed/ML_processed_image")
OUT_META    = Path("data/processed/metadata.csv")

ROUND       = "Round 1 Complete"
JOB_GROUP   = "Job 3"
JOB_NAME    = "Job_3_C5_envelopes_black_perma"
# ===================================

OUT_IMAGES.mkdir(parents=True, exist_ok=True)

# Regex patterns
WRITTEN_RE = re.compile(r".*?_(\d+)_C(\d+)\.png$", re.IGNORECASE)
BLANK_RE   = re.compile(r".*?_(\d+)\.png$", re.IGNORECASE)
GCODE_RE   = re.compile(r"(\d+)\.gcode$", re.IGNORECASE)

def read_gray(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

def compute_diff(blank_gray, written_gray):
    H, W = blank_gray.shape
    wr = cv2.resize(written_gray, (W, H))
    diff = cv2.absdiff(blank_gray, wr)
    _, diff_bin = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    return diff_bin

def main():
    # Collect files
    blank_map = {m.group(1): p for p in BLANK_DIR.glob("*.png")
                 if (m := BLANK_RE.match(p.name))}
    written_map = {m.group(1): (p, f"C{m.group(2)}")
                   for p in WRITTEN_DIR.glob("*.png")
                   if (m := WRITTEN_RE.match(p.name))}
    gcode_map = {m.group(1): p for p in DIGITAL_DIR.glob("*.gcode")
                 if (m := GCODE_RE.match(p.name))}

    with OUT_META.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "idx","Round","JobGroup","JobName","n",
            "written_path","blank_path","gcode_path",
            "output_filename","labels"
        ])

        out_idx = 0
        for idx, (w_path, label) in sorted(written_map.items(), key=lambda kv: int(kv[0])):
            b_path = blank_map.get(idx)
            g_path = gcode_map.get(idx)

            if not b_path:
                print(f"Skipping idx={idx}, no blank found")
                continue

            blank_gray   = read_gray(b_path)
            written_gray = read_gray(w_path)
            if blank_gray is None or written_gray is None:
                print(f"Skipping idx={idx}, cannot read images")
                continue

            diff = compute_diff(blank_gray, written_gray)
            out_name = f"{out_idx}.png"
            cv2.imwrite(str(OUT_IMAGES / out_name), diff)

            writer.writerow([
                out_idx,
                ROUND,
                JOB_GROUP,
                JOB_NAME,
                idx,
                str(w_path),
                str(b_path),
                str(g_path) if g_path else "",
                out_name,
                f"['{label}']"
            ])
            out_idx += 1

    print(f"Done. Saved {out_idx} processed images to {OUT_IMAGES}")
    print(f"Metadata saved to {OUT_META}")

if __name__ == "__main__":
    main()
