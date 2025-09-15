# !/usr/bin/env python3
# gcode_batch_to_png.py — Batch G-code → PNG using your converter
#
# from __future__ import annotations
# from pathlib import Path
# import sys
# import inspect
#
# # ==== EDIT THESE ====
# SRC_DIR = Path("/Users/sivakumarvaradharajan/Downloads/gcode_by_section/job from Campaign - DL_PLUS_ENVELOPE - 2030 11 September 2025")
# DST_DIR = Path("/Users/sivakumarvaradharajan/Downloads/gcode_by_section/outputcheck")
# # ====================
#
# # If needed, add the folder containing your modules:
# # sys.path.append("/absolute/path/to/your/module/folder")
#
# try:
#     from gcode_converter_test.gcode_imager_services import render_gcode_to_png
# except ImportError:
#     try:
#         from gcode_converter_test.gcode_imager_services import render_gcode_to_png
#     except ImportError as e:
#         print("Import error: could not import render_gcode_to_png.")
#         print(e)
#         sys.exit(1)
#
# def ask_canvas_mm():
#     try:
#         cw = input("Canvas width in mm (default 297): ").strip()
#         ch = input("Canvas height in mm (default 210): ").strip()
#         return (float(cw) if cw else 297.0, float(ch) if ch else 210.0)
#     except Exception:
#         print("Invalid input. Using defaults 297×210 mm.")
#         return 297.0, 210.0
#
# def call_converter_adaptive(func, gcode_path: Path, out_png: Path, w_mm: float, h_mm: float):
#     sig = inspect.signature(func)
#     params = list(sig.parameters.keys())
#
#     in_keys  = ["gcode_path", "gcode", "path", "infile", "input_path"]
#     # ✅ include your name "output_png_path"
#     out_keys = ["output_png_path", "out_png", "out_path", "output_path", "png_path", "save_path", "outfile"]
#     w_keys   = ["canvas_width_mm", "canvas_w_mm", "width_mm", "w_mm", "canvas_width"]
#     h_keys   = ["canvas_height_mm", "canvas_h_mm", "height_mm", "h_mm", "canvas_height"]
#
#     def pick(cands):
#         for k in cands:
#             if k in params:
#                 return k
#         return None
#
#     ik, ok, wk, hk = pick(in_keys), pick(out_keys), pick(w_keys), pick(h_keys)
#
#     kwargs = {}
#     if ik: kwargs[ik] = str(gcode_path)
#     if ok: kwargs[ok] = str(out_png)
#     if wk: kwargs[wk] = float(w_mm)
#     if hk: kwargs[hk] = float(h_mm)
#
#     if kwargs:
#         return func(**kwargs)
#
#     # Fallback positional guess
#     return func(str(gcode_path), str(out_png), float(w_mm), float(h_mm))
#
# def main():
#     print("=== Batch G-code → PNG (using your converter) ===")
#     print(f"SRC_DIR: {SRC_DIR.resolve()}")
#     print(f"DST_DIR: {DST_DIR.resolve()}")
#     DST_DIR.mkdir(parents=True, exist_ok=True)
#
#     w_mm, h_mm = ask_canvas_mm()
#
#     files = sorted(SRC_DIR.glob("*.gcode"))
#     if not files:
#         print("No .gcode files found.")
#         return
#
#     ok = 0
#     fail = 0
#     for gfile in files:
#         out_png = DST_DIR / (gfile.stem + ".png")
#         try:
#             call_converter_adaptive(render_gcode_to_png, gfile, out_png, w_mm, h_mm)
#             print(f"✓ {gfile.name} → {out_png.name}")
#             ok += 1
#         except Exception as e:
#             print(f"✗ {gfile.name}: {e}")
#             fail += 1
#
#     print(f"Done. Converted {ok} file(s). Failed: {fail}.")
#
# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# # gcode_batch_to_png.py — batch convert *.gcode → *.png, then crop 7 mm from the top.
# # This version AUTO-loads your local renderer from the same folder (gcode_converter_test)
# # and also works if you point to a directory or a single .gcode file.
#
# from __future__ import annotations
# from pathlib import Path
# import sys
# import inspect
# import importlib
# import importlib.util
# import cv2
# import matplotlib
# matplotlib.use("Agg")  # safe headless backend
#
#
# # ================== EDIT THESE ==================
# # SRC_PATH can be a folder (batch) or a single .gcode file.
# SRC_PATH = Path("/Users/sivakumarvaradharajan/Desktop/gcode_testing/gcode_parsing")
# DST_DIR  = Path("/Users/sivakumarvaradharajan/Desktop/gcode_testing/gcode_parsing/out_png")
#
# # Canvas prompt defaults
# DEFAULT_W_MM = 297.0
# DEFAULT_H_MM = 210.0
#
# # DPI must match the renderer's DPI for correct crop conversion
# DPI = 300
#
# # Crop config (7 mm top)
# CROP_TOP_MM = 7.0
# # ================================================
#
# MM_PER_INCH = 25.4
#
#
# def ask_canvas_mm() -> tuple[float, float]:
#     try:
#         cw = input(f"Canvas width in mm (default {DEFAULT_W_MM:.0f}): ").strip()
#         ch = input(f"Canvas height in mm (default {DEFAULT_H_MM:.0f}): ").strip()
#         cw_mm = float(cw) if cw else DEFAULT_W_MM
#         ch_mm = float(ch) if ch else DEFAULT_H_MM
#     except Exception:
#         print(f"Invalid input. Using defaults {DEFAULT_W_MM:.0f}×{DEFAULT_H_MM:.0f} mm.")
#         cw_mm, ch_mm = DEFAULT_W_MM, DEFAULT_H_MM
#     return cw_mm, ch_mm
#
#
# def _load_from_file(module_name: str, file_path: Path):
#     spec = importlib.util.spec_from_file_location(module_name, str(file_path))
#     if spec and spec.loader:
#         mod = importlib.util.module_from_spec(spec)
#         sys.modules[module_name] = mod
#         spec.loader.exec_module(mod)  # type: ignore[attr-defined]
#         return mod
#     raise ImportError(f"Could not load {module_name} from {file_path}")
#
#
# def import_converter():
#     try:
#         import gcode_imager_services  # lives in the same folder
#         return gcode_imager_services.render_gcode_to_png
#     except Exception as e:
#         print("Failed to import render_gcode_to_png from gcode_imager_services.py")
#         print("Error:", e)
#         sys.exit(1)
#
# def call_converter_adaptive(
#     func,
#     gcode_path: Path,
#     out_png: Path,
#     w_mm: float,
#     h_mm: float,
#     dpi: int = DPI,
#     line_w: float = 1.0,
#     color: str = "0.1,0.2,0.5",
# ):
#     """
#     Call render_gcode_to_png regardless of parameter naming differences.
#     Tries common names for: input path, output path, width/height (mm), dpi, line width/color.
#     """
#     sig = inspect.signature(func)
#     params = list(sig.parameters.keys())
#
#     def pick(cands):
#         for k in cands:
#             if k in params:
#                 return k
#         return None
#
#     in_keys  = ["gcode_path", "gcode", "path", "infile", "input_path"]
#     out_keys = ["output_png_path", "out_png", "out_path", "output_path", "png_path", "save_path", "outfile"]
#     w_keys   = ["width_mm", "canvas_width_mm", "canvas_w_mm", "width", "w_mm", "canvas_width"]
#     h_keys   = ["height_mm", "canvas_height_mm", "canvas_h_mm", "height", "h_mm", "canvas_height"]
#     dpi_keys = ["dpi"]
#     lw_keys  = ["line_w", "line_width", "linewidth"]
#     col_keys = ["line_color", "color"]
#
#     kwargs = {}
#     ik = pick(in_keys)
#     ok = pick(out_keys)
#     wk = pick(w_keys)
#     hk = pick(h_keys)
#     dk = pick(dpi_keys)
#     lk = pick(lw_keys)
#     ck = pick(col_keys)
#
#     if ik: kwargs[ik] = str(gcode_path)
#     if ok: kwargs[ok] = str(out_png)
#     if wk: kwargs[wk] = float(w_mm)
#     if hk: kwargs[hk] = float(h_mm)
#     if dk: kwargs[dk] = int(dpi)
#     if lk: kwargs[lk] = float(line_w)
#     if ck: kwargs[ck] = str(color)
#
#     if kwargs:
#         return func(**kwargs)
#
#     # Fallback positional order guess
#     return func(str(gcode_path), str(out_png), float(w_mm), float(h_mm))
#
#
# def crop_top_mm(png_path: Path, dpi: int = DPI, crop_mm: float = CROP_TOP_MM):
#     crop_px = int(round((crop_mm / MM_PER_INCH) * dpi))
#     img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise RuntimeError(f"Could not read PNG for cropping: {png_path}")
#     h, w = img.shape[:2]
#     crop_px = max(0, min(crop_px, h - 1))
#     if crop_px > 0:
#         img = img[crop_px:, :]
#         cv2.imwrite(str(png_path), img)
#
#
# def gather_gcode_files(src_path: Path) -> list[Path]:
#     if src_path.is_file() and src_path.suffix.lower() == ".gcode":
#         return [src_path]
#     if src_path.is_dir():
#         return sorted([p for p in src_path.glob("*.gcode")])
#     raise FileNotFoundError(f"SRC_PATH not found or not a .gcode/.dir: {src_path}")
#
#
# def main():
#     print("=== G-code → PNG batch (with 7 mm top crop) ===")
#     print(f"SRC_PATH: {SRC_PATH.resolve()}")
#     print(f"DST_DIR : {DST_DIR.resolve()}")
#
#     render_gcode_to_png = import_converter()
#     w_mm, h_mm = ask_canvas_mm()
#
#     DST_DIR.mkdir(parents=True, exist_ok=True)
#
#     gcode_files = gather_gcode_files(SRC_PATH)
#     if not gcode_files:
#         print("No .gcode files found. Done.")
#         return
#
#     ok = 0
#     failed = 0
#     for gpath in gcode_files:
#         try:
#             out_png = DST_DIR / (gpath.stem + ".png")
#
#             # Render (ensuring DPI matches crop DPI)
#             call_converter_adaptive(
#                 render_gcode_to_png,
#                 gcode_path=gpath,
#                 out_png=out_png,
#                 w_mm=w_mm,
#                 h_mm=h_mm,
#                 dpi=DPI,
#                 line_w=1.0,
#                 color="0.1,0.2,0.5",
#             )
#
#             # Crop 7 mm on top
#             crop_top_mm(out_png, dpi=DPI, crop_mm=CROP_TOP_MM)
#
#             print(f"✓ {gpath.name} → {out_png.name} (cropped top {CROP_TOP_MM} mm)")
#             ok += 1
#         except Exception as e:
#             print(f"✗ {gpath.name}: {e}")
#             failed += 1
#
#     print(f"Done. Converted {ok} file(s). Failed: {failed}.")
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# gcode_batch_to_png.py — batch convert *.gcode → *.png, then crop 7 mm from the top.
# Fix: pass a proper matplotlib color (tuple) instead of a CSV string.

from __future__ import annotations
from pathlib import Path
import sys
import inspect
import cv2

# ================== EDIT THESE ==================
# SRC_PATH can be a folder (batch) or a single .gcode file.
SRC_PATH = Path("/Users/sivakumarvaradharajan/Desktop/gcode_testing/gcode_parsing")
DST_DIR  = Path("/Users/sivakumarvaradharajan/Desktop/gcode_testing/gcode_parsing/out_png")

# Canvas prompt defaults
DEFAULT_W_MM = 297.0
DEFAULT_H_MM = 210.0

# DPI must match the renderer's DPI for correct crop conversion
DPI = 300

# Crop config (7 mm top)
CROP_TOP_MM = 7.0

# Use a proper matplotlib color tuple (r,g,b) in 0..1
LINE_COLOR = (0.1, 0.2, 0.5)
LINE_WIDTH = 1.0
# ================================================

MM_PER_INCH = 25.4


def ask_canvas_mm() -> tuple[float, float]:
    try:
        cw = input(f"Canvas width in mm (default {DEFAULT_W_MM:.0f}): ").strip()
        ch = input(f"Canvas height in mm (default {DEFAULT_H_MM:.0f}): ").strip()
        cw_mm = float(cw) if cw else DEFAULT_W_MM
        ch_mm = float(ch) if ch else DEFAULT_H_MM
    except Exception:
        print(f"Invalid input. Using defaults {DEFAULT_W_MM:.0f}×{DEFAULT_H_MM:.0f} mm.")
        cw_mm, ch_mm = DEFAULT_W_MM, DEFAULT_H_MM
    return cw_mm, ch_mm


def import_converter():
    """
    Import render_gcode_to_png from the local file gcode_imager_services.py
    (same folder as this script).
    """
    try:
        import gcode_imager_services  # lives alongside this script
        return gcode_imager_services.render_gcode_to_png
    except Exception as e:
        print("Failed to import render_gcode_to_png from gcode_imager_services.py")
        print("Error:", e)
        sys.exit(1)


def call_converter_adaptive(
    func,
    gcode_path: Path,
    out_png: Path,
    w_mm: float,
    h_mm: float,
    dpi: int = DPI,
    line_w: float = LINE_WIDTH,
    color = LINE_COLOR,  # pass as tuple, not string
):
    """
    Call render_gcode_to_png regardless of parameter naming differences.
    Tries common names for: input path, output path, width/height (mm), dpi, line width/color.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    def pick(cands):
        for k in cands:
            if k in params:
                return k
        return None

    in_keys  = ["gcode_path", "gcode", "path", "infile", "input_path"]
    out_keys = ["output_png_path", "out_png", "out_path", "output_path", "png_path", "save_path", "outfile"]
    w_keys   = ["width_mm", "canvas_width_mm", "canvas_w_mm", "width", "w_mm", "canvas_width"]
    h_keys   = ["height_mm", "canvas_height_mm", "canvas_h_mm", "height", "h_mm", "canvas_height"]
    dpi_keys = ["dpi"]
    lw_keys  = ["line_w", "line_width", "linewidth"]
    col_keys = ["line_color", "color"]

    kwargs = {}
    ik = pick(in_keys)
    ok = pick(out_keys)
    wk = pick(w_keys)
    hk = pick(h_keys)
    dk = pick(dpi_keys)
    lk = pick(lw_keys)
    ck = pick(col_keys)

    if ik: kwargs[ik] = str(gcode_path)
    if ok: kwargs[ok] = str(out_png)
    if wk: kwargs[wk] = float(w_mm)
    if hk: kwargs[hk] = float(h_mm)
    if dk: kwargs[dk] = int(dpi)
    if lk: kwargs[lk] = float(line_w)
    if ck: kwargs[ck] = color  # <-- keep as tuple (r,g,b) for matplotlib

    if kwargs:
        return func(**kwargs)

    # Fallback positional order guess
    return func(str(gcode_path), str(out_png), float(w_mm), float(h_mm))


def crop_top_mm(png_path: Path, dpi: int = DPI, crop_mm: float = CROP_TOP_MM):
    crop_px = int(round((crop_mm / MM_PER_INCH) * dpi))
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read PNG for cropping: {png_path}")
    h, w = img.shape[:2]
    crop_px = max(0, min(crop_px, h - 1))
    if crop_px > 0:
        img = img[crop_px:, :]
        cv2.imwrite(str(png_path), img)


def gather_gcode_files(src_path: Path) -> list[Path]:
    if src_path.is_file() and src_path.suffix.lower() == ".gcode":
        return [src_path]
    if src_path.is_dir():
        return sorted([p for p in src_path.glob("*.gcode")])
    raise FileNotFoundError(f"SRC_PATH not found or not a .gcode/.dir: {src_path}")


def main():
    print("=== G-code → PNG batch (with 7 mm top crop) ===")
    print(f"SRC_PATH: {SRC_PATH.resolve()}")
    print(f"DST_DIR : {DST_DIR.resolve()}")

    render_gcode_to_png = import_converter()
    w_mm, h_mm = ask_canvas_mm()

    DST_DIR.mkdir(parents=True, exist_ok=True)

    gcode_files = gather_gcode_files(SRC_PATH)
    if not gcode_files:
        print("No .gcode files found. Done.")
        return

    ok = 0
    failed = 0
    for gpath in gcode_files:
        try:
            out_png = DST_DIR / (gpath.stem + ".png")

            # Render (ensuring DPI matches crop DPI)
            call_converter_adaptive(
                render_gcode_to_png,
                gcode_path=gpath,
                out_png=out_png,
                w_mm=w_mm,
                h_mm=h_mm,
                dpi=DPI,
                line_w=LINE_WIDTH,
                color=LINE_COLOR,
            )

            # Crop 7 mm on top
            crop_top_mm(out_png, dpi=DPI, crop_mm=CROP_TOP_MM)

            print(f"✓ {gpath.name} → {out_png.name} (cropped top {CROP_TOP_MM} mm)")
            ok += 1
        except Exception as e:
            print(f"✗ {gpath.name}: {e}")
            failed += 1

    print(f"Done. Converted {ok} file(s). Failed: {failed}.")


if __name__ == "__main__":
    main()
