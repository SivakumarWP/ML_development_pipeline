# python build_ml_processing_dataset.py /absolute/path/to/ROOT_DIR
# or edit ROOT_DIR at the top, then:
# python build_ml_processing_dataset.py

import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================
# Configure: default root directory to scan
# (you can also pass it on the CLI)
# ============================================
ROOT_DIR = Path("/path/to/your/data/root")   # <-- change me or pass as argv[1]


# ============================================
# Geometry / pipeline params
# ============================================
CARD_HEIGHT_MM = 222.5
DIGITAL_TOP_CROP_MM = 7.0


# ============================================
# Output helpers
# ============================================
def get_out_root(root_dir: Path) -> Path:
    out_root = root_dir / "ML_PROCESSING_DATA"
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root

num_png_re = re.compile(r"^(\d+)\.png$", re.IGNORECASE)

def next_sequential_number(out_root: Path) -> int:
    """
    Look for N.png files in OUT_ROOT and return the next integer after the max.
    Starts at 1 if none exist.
    """
    max_n = 0
    for p in out_root.glob("*.png"):
        m = num_png_re.match(p.name)
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except ValueError:
                pass
    return max_n + 1


# ============================================
# Image / vision utils
# ============================================
def detect_card_bbox(image_bgr):
    """Find a large, single card-like blob and return x,y,w,h."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)  # (x, y, w, h)


def crop_to_bbox(image, bbox):
    x, y, w, h = bbox
    return image[y:y + h, x:x + w]


def diff1_subtraction(blank_bgr, written_bgr, bbox):
    """Return |written - blank| in grayscale inside the detected card bbox."""
    blank_crop = crop_to_bbox(blank_bgr, bbox)
    written_crop = crop_to_bbox(written_bgr, bbox)
    b_gray = cv2.cvtColor(blank_crop, cv2.COLOR_BGR2GRAY)
    w_gray = cv2.cvtColor(written_crop, cv2.COLOR_BGR2GRAY)
    return cv2.absdiff(b_gray, w_gray)


def crop_digital_top_mm(digital_gray, card_height_mm=CARD_HEIGHT_MM, top_mm=DIGITAL_TOP_CROP_MM):
    """Crop a fixed mm from the top of the digital image (to remove headers/margins)."""
    h, _ = digital_gray.shape[:2]
    px_per_mm = h / float(card_height_mm)
    offset_px = int(round(top_mm * px_per_mm))
    offset_px = max(0, min(offset_px, max(0, h - 1)))
    return digital_gray[offset_px:, :]


def align_digital_to_diff1_orb(digital_gray, diff1_gray):
    """
    Align digital (as grayscale) to diff1 using ORB features + homography.
    Returns a version of digital resized to diff1 and warped if homography found.
    """
    Ht, Wt = diff1_gray.shape[:2]

    # mild denoise + invert digital for better feature similarity
    diff1_proc = cv2.GaussianBlur(diff1_gray, (3, 3), 0)
    digital_inv = 255 - digital_gray
    digital_proc = cv2.GaussianBlur(digital_inv, (3, 3), 0)

    _, diff1_bin = cv2.threshold(diff1_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, digital_bin = cv2.threshold(digital_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    digital_bin = cv2.resize(digital_bin, (Wt, Ht))
    digital_gray_rs = cv2.resize(digital_gray, (Wt, Ht))

    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(diff1_bin, None)
    k2, d2 = orb.detectAndCompute(digital_bin, None)

    aligned = digital_gray_rs
    if d1 is not None and d2 is not None and len(k1) >= 4 and len(k2) >= 4:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if matches:
            matches = sorted(matches, key=lambda m: m.distance)
            good = matches[:max(30, int(0.25 * len(matches)))]
            if len(good) >= 4:
                src = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # digital
                dst = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # diff1
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None:
                    aligned = cv2.warpPerspective(digital_gray_rs, H, (Wt, Ht), flags=cv2.INTER_LINEAR)
    return aligned


# ============================================
# Filename patterns and discovery
# ============================================
re_blank   = re.compile(r"^blank_(\d+)\.png$", re.IGNORECASE)
re_digital = re.compile(r"^digital_(\d+)\.png$", re.IGNORECASE)
# written_*_classname.png -> capture index and label (label is last underscore token)
re_written = re.compile(r"^written_(\d+)_([A-Za-z0-9\-]+)\.png$", re.IGNORECASE)

def index_of_blank(p: Path):
    m = re_blank.match(p.name)
    return m.group(1) if m else None

def index_and_label_of_written(p: Path):
    m = re_written.match(p.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def index_of_digital(p: Path):
    m = re_digital.match(p.name)
    return m.group(1) if m else None


def find_triplets_in_dir(dir_path: Path):
    """
    Returns tuples: (blank_path, written_path, digital_path, index, label)
    Only includes triplets where all three indices match.
    If multiple written files share the index but different labels, yields all.
    """
    blanks = {}
    digitals = {}
    writtens_by_index = {}

    for p in sorted(dir_path.glob("*.png")):
        idx = index_of_blank(p)
        if idx is not None:
            blanks[idx] = p
            continue

        idx = index_of_digital(p)
        if idx is not None:
            digitals[idx] = p
            continue

        idx, label = index_and_label_of_written(p)
        if idx is not None and label is not None:
            writtens_by_index.setdefault(idx, []).append((p, label))

    triplets = []
    common = set(blanks.keys()) & set(digitals.keys()) & set(writtens_by_index.keys())
    for idx in sorted(common, key=lambda x: int(x)):
        for w_path, label in sorted(writtens_by_index[idx], key=lambda t: t[0].name.lower()):
            triplets.append((blanks[idx], w_path, digitals[idx], idx, label))
    return triplets


# ============================================
# Processing a single triplet
# ============================================
def process_triplet(blank_path: Path, written_path: Path, digital_path: Path, idx: str, label: str):
    """Return diff2 image (numpy array)."""
    blank_bgr = cv2.imread(str(blank_path))
    written_bgr = cv2.imread(str(written_path))
    digital_bgr = cv2.imread(str(digital_path))

    if blank_bgr is None or written_bgr is None or digital_bgr is None:
        raise RuntimeError("Failed to read one of the images.")

    bbox = detect_card_bbox(blank_bgr)
    if bbox is None:
        raise RuntimeError("Could not detect card bbox in blank image.")

    # diff1
    d1 = diff1_subtraction(blank_bgr, written_bgr, bbox)

    # prepare + align digital
    digital_gray = cv2.cvtColor(digital_bgr, cv2.COLOR_BGR2GRAY)
    digital_gray = crop_digital_top_mm(digital_gray, CARD_HEIGHT_MM, DIGITAL_TOP_CROP_MM)
    digital_aligned = align_digital_to_diff1_orb(digital_gray, d1)

    # diff2
    d2 = cv2.absdiff(d1, digital_aligned)
    return d2


# ============================================
# Main
# ============================================
def main():
    # CLI override: python build_ml_processing_dataset.py /path/to/root
    root_dir = ROOT_DIR
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        root_dir = root_dir.expanduser().resolve()

    out_root = get_out_root(root_dir)
    csv_path = out_root / "metadata.csv"

    # Determine starting number for N.png
    n = next_sequential_number(out_root)

    # Prepare CSV
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["blank_path", "written_path", "digital_path", "diff2_path", "label"])

        # Walk all subfolders (sorted for determinism)
        for dir_path in sorted(root_dir.rglob("*")):
            if not dir_path.is_dir():
                continue

            triplets = find_triplets_in_dir(dir_path)
            if not triplets:
                continue

            for blank_path, written_path, digital_path, idx, label in triplets:
                try:
                    d2 = process_triplet(blank_path, written_path, digital_path, idx, label)

                    # Save as N.png (sequential)
                    diff2_file = out_root / f"{n}.png"
                    ok = cv2.imwrite(str(diff2_file), d2)
                    if not ok:
                        raise RuntimeError("Failed to write diff2 image.")

                    # Write metadata (absolute paths for consistency)
                    writer.writerow([
                        str(blank_path.resolve()),
                        str(written_path.resolve()),
                        str(digital_path.resolve()),
                        str(diff2_file.resolve()),
                        label
                    ])

                    print(f"OK  #{n}  idx={idx} label={label}")
                    n += 1

                except Exception as e:
                    print(f"FAIL idx={idx} label={label} in {dir_path}: {e}")

    print(f"Done. Outputs in: {out_root}")
    print(f"Metadata CSV: {csv_path}")


if __name__ == "__main__":
    main()
