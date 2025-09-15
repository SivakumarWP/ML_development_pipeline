from pathlib import Path
import cv2, csv, re, numpy as np
from configs.config import CONFIG

def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def parse_label_from_written(written_path: Path) -> str:
    m = re.search(r"_C(\d+)\b", written_path.stem, flags=re.IGNORECASE)
    return f"C{m.group(1)}" if m else "C0"

def main():
    blank_p, written_p = Path(CONFIG.BLANK_PATH), Path(CONFIG.WRITTEN_PATH)
    digital_p = Path(CONFIG.DIGITAL_IMG)
    gcode_p = Path(CONFIG.GCODE_PATH) if CONFIG.GCODE_PATH else None

    out_dir = Path(CONFIG.OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_idx = int(CONFIG.OUT_IDX)
    out_name = f"{out_idx}.png"
    out_path = out_dir / out_name

    # load grayscale
    blank, written, digital = read_gray(blank_p), read_gray(written_p), read_gray(digital_p)

    # resize to blank size (no alignment)
    H, W = blank.shape[:2]
    written_r = cv2.resize(written, (W, H), interpolation=cv2.INTER_LINEAR)
    digital_r = cv2.resize(digital, (W, H), interpolation=cv2.INTER_LINEAR)

    # light blur
    k = CONFIG.GAUSS_K if CONFIG.GAUSS_K % 2 == 1 else CONFIG.GAUSS_K + 1
    b_blur = cv2.GaussianBlur(blank, (k, k), 0)
    w_blur = cv2.GaussianBlur(written_r, (k, k), 0)
    d_blur = cv2.GaussianBlur(digital_r, (k, k), 0)

    # diffs
    diff1 = cv2.absdiff(b_blur, w_blur)
    raw_diff2 = cv2.absdiff(diff1, d_blur)

    # binarize & clean
    _, diff2_bin = cv2.threshold(raw_diff2, CONFIG.THRESH_DIFF, 255, cv2.THRESH_BINARY)
    diff2_bin = cv2.morphologyEx(
        diff2_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1
    )

    # save processed
    cv2.imwrite(str(out_path), diff2_bin)

    # append metadata (TSV like your sample)
    meta_path = Path(CONFIG.META_TSV)
    is_new = not meta_path.exists()
    with meta_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if is_new:
            w.writerow(["idx","Round","JobGroup","JobName","n",
                        "written_path","blank_path","gcode_path",
                        "output_filename","labels"])
        label = parse_label_from_written(written_p)
        w.writerow([
            out_idx,
            CONFIG.ROUND,
            CONFIG.JOB_GROUP,
            CONFIG.JOB_NAME,
            CONFIG.N_INDEX,
            str(written_p).replace("\\", "/"),
            str(blank_p).replace("\\", "/"),
            (str(gcode_p).replace("\\", "/") if gcode_p else ""),
            out_name,
            f"['{label}']",
        ])

    print(f"Saved diff2 → {out_path}")
    print(f"Appended row → {meta_path}")

if __name__ == "__main__":
    main()
