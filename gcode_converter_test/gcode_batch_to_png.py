#!/usr/bin/env python3
# gcode_batch_to_png.py — Batch G-code → PNG using your converter

from __future__ import annotations
from pathlib import Path
import sys
import inspect

# ==== EDIT THESE ====
SRC_DIR = Path("/Users/sivakumarvaradharajan/Downloads/gcode_by_section/job from Campaign - DL_PLUS_ENVELOPE - 2030 11 September 2025")
DST_DIR = Path("/Users/sivakumarvaradharajan/Downloads/gcode_by_section/outputcheck")
# ====================

# If needed, add the folder containing your modules:
# sys.path.append("/absolute/path/to/your/module/folder")

try:
    from gcode_converter_test.gcode_imager_services import render_gcode_to_png
except ImportError:
    try:
        from gcode_converter_test.gcode_imager_services import render_gcode_to_png
    except ImportError as e:
        print("Import error: could not import render_gcode_to_png.")
        print(e)
        sys.exit(1)

def ask_canvas_mm():
    try:
        cw = input("Canvas width in mm (default 297): ").strip()
        ch = input("Canvas height in mm (default 210): ").strip()
        return (float(cw) if cw else 297.0, float(ch) if ch else 210.0)
    except Exception:
        print("Invalid input. Using defaults 297×210 mm.")
        return 297.0, 210.0

def call_converter_adaptive(func, gcode_path: Path, out_png: Path, w_mm: float, h_mm: float):
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    in_keys  = ["gcode_path", "gcode", "path", "infile", "input_path"]
    # ✅ include your name "output_png_path"
    out_keys = ["output_png_path", "out_png", "out_path", "output_path", "png_path", "save_path", "outfile"]
    w_keys   = ["canvas_width_mm", "canvas_w_mm", "width_mm", "w_mm", "canvas_width"]
    h_keys   = ["canvas_height_mm", "canvas_h_mm", "height_mm", "h_mm", "canvas_height"]

    def pick(cands):
        for k in cands:
            if k in params:
                return k
        return None

    ik, ok, wk, hk = pick(in_keys), pick(out_keys), pick(w_keys), pick(h_keys)

    kwargs = {}
    if ik: kwargs[ik] = str(gcode_path)
    if ok: kwargs[ok] = str(out_png)
    if wk: kwargs[wk] = float(w_mm)
    if hk: kwargs[hk] = float(h_mm)

    if kwargs:
        return func(**kwargs)

    # Fallback positional guess
    return func(str(gcode_path), str(out_png), float(w_mm), float(h_mm))

def main():
    print("=== Batch G-code → PNG (using your converter) ===")
    print(f"SRC_DIR: {SRC_DIR.resolve()}")
    print(f"DST_DIR: {DST_DIR.resolve()}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    w_mm, h_mm = ask_canvas_mm()

    files = sorted(SRC_DIR.glob("*.gcode"))
    if not files:
        print("No .gcode files found.")
        return

    ok = 0
    fail = 0
    for gfile in files:
        out_png = DST_DIR / (gfile.stem + ".png")
        try:
            call_converter_adaptive(render_gcode_to_png, gfile, out_png, w_mm, h_mm)
            print(f"✓ {gfile.name} → {out_png.name}")
            ok += 1
        except Exception as e:
            print(f"✗ {gfile.name}: {e}")
            fail += 1

    print(f"Done. Converted {ok} file(s). Failed: {fail}.")

if __name__ == "__main__":
    main()
