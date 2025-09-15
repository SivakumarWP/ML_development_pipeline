#!/usr/bin/env python3
# gcode_img_endpoint.py — endpoint-friendly G-code → PNG renderer
# - Reuses parser + canvas size from gcode_imager_tk.py (no Tkinter)
# - Usage:
#     python gcode_img_endpoint.py --input /path/to/file.gcode --tmp_path /tmp
#   (saves to /tmp/file.png and prints absolute path)
# - Optional:
#     --output /path/to/output.png  (overrides tmp_path behavior)
#     --dpi 300
#     --line_width 1.0
#     --color 0.1,0.2,0.5   (RGB in 0..1)

import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gcode_converter_test.gcode_imager_tk import (
    read_gcode_file,
    parse_gcode_lines,
    canvas_width_mm,
    canvas_height_mm,
)

def _normalize_pen_comments(lines):
    out = []
    for ln in lines:
        ln2 = re.sub(r'#?\b[Pp]en[_\s-]?Up\b', 'Pen Up', ln)
        ln2 = re.sub(r'#?\b[Pp]en[_\s-]?Down\b', 'Pen Down', ln2)
        out.append(ln2)
    return out

def _detect_units_scale(lines):
    is_inches = any(re.search(r'^\s*G20\b', ln) for ln in lines)
    return 25.4 if is_inches else 1.0

def _parse_color(s: str):
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError("--color expects 'r,g,b' with values 0..1")
        return tuple(float(p) for p in parts)
    return s  # matplotlib named color

def render_gcode_to_png(
    gcode_path: str,
    output_png_path: str,
    line_color=(0.1, 0.2, 0.5),
    line_w: float = 1.0,
    dpi: int = 300,
) -> str:
    lines = read_gcode_file(gcode_path)
    lines = _normalize_pen_comments(lines)

    x_values, y_values = parse_gcode_lines(lines)
    if not any(x_values):
        raise RuntimeError(f"No pen-down paths parsed from: {gcode_path}")

    xs = np.concatenate([np.array(seg) for seg in x_values if seg])
    ys = np.concatenate([np.array(seg) for seg in y_values if seg])
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    scale = _detect_units_scale(lines)

    # Use canvas size defined in gcode_imager_tk.py
    Wmm = float(canvas_width_mm)
    Hmm = float(canvas_height_mm)

    shift_x = -min_x * scale
    shift_y = -min_y * scale

    plt.figure(figsize=(Wmm / 25.4, Hmm / 25.4))
    ax = plt.gca()

    for seg_x, seg_y in zip(x_values, y_values):
        if not seg_x:
            continue
        sx = (np.array(seg_x) * scale) + shift_x
        sy = (np.array(seg_y) * scale) + shift_y
        ax.plot(sx, sy, c=line_color, linewidth=line_w)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, Wmm)
    ax.set_ylim(Hmm, 0)  # top-left origin
    ax.axis('off')

    os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)
    plt.savefig(output_png_path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
    plt.close('all')
    return os.path.abspath(output_png_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .gcode file")
    ap.add_argument("--tmp_path", default="", help="Directory to save output (used if --output not set)")
    ap.add_argument("--output", default="", help="Explicit output .png path (overrides --tmp_path)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--line_width", type=float, default=1.0)
    ap.add_argument("--color", default="0.1,0.2,0.5")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0] + ".png"
        out_dir = args.tmp_path or os.getcwd()
        out_path = os.path.join(out_dir, base)

    try:
        color = _parse_color(args.color)
        saved = render_gcode_to_png(
            args.input,
            out_path,
            line_color=color,
            line_w=args.line_width,
            dpi=args.dpi,
        )
        print(saved)
        sys.exit(0)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
