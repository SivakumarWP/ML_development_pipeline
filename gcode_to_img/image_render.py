#!/usr/bin/env python3
import os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog

DEBUG = True

def dprint(*args):
    if DEBUG: print(*args, flush=True)

def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))

# ---------------------------
# 1) Parse Z to detect "pen down"
# ---------------------------
def detect_pen_down_z(gcode_path: str) -> float:
    abs_pos = True
    z = 0.0
    z_levels = []
    with open(gcode_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):  # comments / blanks
                continue
            line = re.sub(r"\(.*?\)", "", line)   # remove ( ... ) comments
            if not line: continue

            if line.startswith("G90"): abs_pos = True;  continue
            if line.startswith("G91"): abs_pos = False; continue

            if line.startswith(("G0","G00","G1","G01")):
                mz = re.search(r"[Zz](-?\d+\.?\d*)", line)
                if mz:
                    vz = float(mz.group(1))
                    z = (z + vz) if not abs_pos else vz
                    z_levels.append(round(z, 3))
    pdz = min(z_levels) if z_levels else 0.0
    dprint(f"[detect_pen_down_z] {os.path.basename(gcode_path)} -> pen_down_z={pdz} (from {len(z_levels)} Z samples)")
    return pdz

# ---------------------------
# 2) Build line segments when Z==pen_down_z
# ---------------------------
def collect_segments(gcode_path: str, width_mm: float, height_mm: float, pen_down_z: float):
    abs_pos, units_mm = True, True
    x = y = z = 0.0
    segs = []
    with open(gcode_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"): continue
            line = re.sub(r"\(.*?\)", "", line)
            if not line: continue

            if line.startswith("G90"): abs_pos = True;  continue
            if line.startswith("G91"): abs_pos = False; continue
            if line.startswith("G20"): units_mm = False; continue  # inches
            if line.startswith("G21"): units_mm = True;  continue  # mm

            if line.startswith(("G0","G00","G1","G01")):
                mx = re.search(r"[Xx](-?\d+\.?\d*)", line)
                my = re.search(r"[Yy](-?\d+\.?\d*)", line)
                mz = re.search(r"[Zz](-?\d+\.?\d*)", line)

                nx, ny, nz = x, y, z
                if mx: nx = (x + float(mx.group(1))) if not abs_pos else float(mx.group(1))
                if my: ny = (y + float(my.group(1))) if not abs_pos else float(my.group(1))
                if mz: nz = (z + float(mz.group(1))) if not abs_pos else float(mz.group(1))

                if not units_mm:  # convert inches -> mm
                    nx *= 25.4; ny *= 25.4; nz *= 25.4

                # draw only when pen is down
                if abs(z - pen_down_z) < 0.01 and ((nx != x) or (ny != y)):
                    segs.append(((x, y), (nx, ny)))

                x, y, z = nx, ny, nz

    # flip Y so origin is bottom-left
    def flip_y(pt): return (pt[0], height_mm - pt[1])
    segs = [(flip_y(a), flip_y(b)) for a, b in segs]

    # debug: print drawing bbox & count
    if segs:
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        dprint(f"[collect_segments] segs={len(segs)} bbox_mm=(xmin={bbox[0]:.2f}, ymin={bbox[1]:.2f}, xmax={bbox[2]:.2f}, ymax={bbox[3]:.2f})")
    else:
        dprint("[collect_segments] segs=0 (no pen-down moves)")

    return segs

# ---------------------------
# 3) Render to an exact-size PNG
# ---------------------------
def render_segments(segments, width_mm, height_mm, dpi=300, out_path="preview.png"):
    px_w = mm_to_px(width_mm, dpi)
    px_h = mm_to_px(height_mm, dpi)

    # Build figure that EXACTLY matches target pixels
    fig = plt.figure(figsize=(px_w / dpi, px_h / dpi), dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])  # fill entire canvas

    # Coordinate system is in mm to match your inputs
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.axis("off")

    for (p0, p1) in segments:
        (x0, y0), (x1, y1) = p0, p1
        ax.plot([x0, x1], [y0, y1], linewidth=0.7, color="black")

    # Save WITHOUT cropping or padding — preserves exact px size
    fig.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    # Verify on disk
    with Image.open(out_path) as im:
        saved_px = im.size

    dprint(f"[render_segments] target_px=({px_w},{px_h}) fig_inches={fig.get_size_inches()} dpi={fig.dpi} -> saved_px={saved_px}")
    if saved_px != (px_w, px_h):
        dprint("  [WARN] Saved pixel size does not match target. Check for external edits or older function in use.")
    return out_path

# ---------------------------
# 4) Batch runner + names
# ---------------------------
def process_folder(folder, width_mm, height_mm, dpi=300):
    print(f"[process_folder] using script: {os.path.abspath(sys.argv[0])}")
    print(f"[process_folder] width_mm={width_mm}, height_mm={height_mm}, dpi={dpi} "
          f"-> target_px=({mm_to_px(width_mm,dpi)}, {mm_to_px(height_mm,dpi)})")

    gcode_files = [f for f in os.listdir(folder) if f.lower().endswith(".gcode")]
    gcode_files.sort()
    print(f"[process_folder] found {len(gcode_files)} .gcode files")

    for fname in gcode_files:
        path = os.path.join(folder, fname)
        z = detect_pen_down_z(path)
        segs = collect_segments(path, width_mm, height_mm, z)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(folder, f"{base}.png")
        render_segments(segs, width_mm, height_mm, dpi=dpi, out_path=out_path)
        print(f"[DONE] {fname} -> {out_path}")

# ---------------------------
# 5) CLI
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(title="Select folder with .gcode files")
    if not folder:
        print("No folder selected."); sys.exit(1)
    try:
        width_mm  = float(input("Enter width in mm: "))
        height_mm = float(input("Enter height in mm: "))
    except ValueError:
        print("Invalid width/height"); sys.exit(1)

    process_folder(folder, width_mm, height_mm, dpi=300)
    print("All files processed.")
