#!/usr/bin/env python3
"""
gcode_imager_tk.py

- Includes your original functions (unchanged) for compatibility.
- Adds a robust CLI renderer that:
  * normalizes pen up/down markers,
  * uses your parse_gcode_lines() to draw only pen-down strokes,
  * detects units (G20 inches / G21 mm),
  * auto-fits to nearest standard paper (A4, A5, DL, DL-rot),
  * renders to PNG with correct physical dimensions.

USAGE:
    # GUI (your original visualize_gcode)
    python gcode_imager_tk.py

    # CLI render to PNG (recommended)
    python gcode_imager_tk.py input.gcode output.png
"""

# -----------------------------
# Your original script (as-is)
# -----------------------------
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

MM_TO_PIXEL = 3.779527559055

# Define dimensions in millimeters
canvas_width_mm = 297
canvas_height_mm = 210

# Convert dimensions to pixels

# Declare x_values and y_values as global variables
x_values = []
y_values = []
z_values = []

# Initialize global variables for zoom_factor, line_width, and delay
zoom_factor = 10
line_width = 2
delay = 3

# Variables to store the previous zoom factor and mouse position
prev_zoom_factor = zoom_factor
prev_mouse_x = None
prev_mouse_y = None


def read_gcode_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines


def parse_gcode_lines(lines):
    """Reads in the gcode lines and returns the movement of the pen.

    :param lines: A list of strings. Output of read_gcode_file function

    :return: x_values and y_values. Pen up/down movements break the values into sublists.
    """
    x_values = []
    y_values = []
    current_letter_x = []
    current_letter_y = []
    current_position_x = 0
    current_position_y = 0
    pen_up = True

    for line in lines:
        # Only look at the linear movements

        if line.startswith("G1"):

            parts = line.split()

            # If it is a pen up movement then append current_letter to x_values, y_values
            # and set pen_up variable to be True
            if all(x in parts for x in ["Pen", "Up"]):
                x_values.append(current_letter_x)
                y_values.append(current_letter_y)
                pen_up = True

            # If it is a pen down movement, create a new current_letter
            # and append previous xy movement. Set pen_up to False
            if all(x in parts for x in ["Pen", "Down"]):
                current_letter_x = [current_position_x]
                current_letter_y = [current_position_y]
                pen_up = False

            # If it is an xy movement, append the values to current letter
            if (
                len(parts) == 3
                and parts[1].startswith("X")
                and parts[2].startswith("Y")
            ):
                # Only count the line if the X and Y values actually have a value
                if len(parts[1][1:]) > 0 and len(parts[2][1:]) > 0:
                    current_position_x = float(parts[1][1:])
                    current_position_y = float(parts[2][1:])

                    # If the pen is down, append the current position to the current letter
                    if pen_up == False:
                        current_letter_x.append(current_position_x)
                        current_letter_y.append(current_position_y)

    return x_values, y_values


def export_parsed_gcode_as_png(
    x_values,
    y_values,
    output_file_path,
    line_width=1.0,
    line_color=(0.1, 0.2, 0.5),
    face_color="white",
):
    """Plots the lines from the parsed gcode into an image, and then saves as .png.

    :param line_color: [R,G,B] with each R,G,B in range [0,1]
    """

    plt.figure(
        figsize=(10, 5), facecolor=face_color
    )  # Adjust the figure size as needed
    for x, y in zip(x_values, y_values):
        plt.plot(x, y, c=line_color, linewidth=line_width)

    plt.xlim(0, 200)
    plt.ylim(100, 0)
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig(output_file_path)
    plt.show()


def save_parsed_gcode_as_png(
    x_values,
    y_values,
    output_file_path,
    line_width=1.0,
    line_color=(0.1, 0.2, 0.5),
    face_color="white",
):
    """Plots the lines from the parsed gcode into an image, and then saves as .png.

    :param line_color: [R,G,B] with each R,G,B in range [0,1]
    """

    plt.figure(
        figsize=(10, 5), facecolor=face_color
    )  # Adjust the figure size as needed
    for x, y in zip(x_values, y_values):
        plt.plot(x, y, c=line_color, linewidth=line_width)

    plt.xlim(0, 200)
    plt.ylim(100, 0)
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig(output_file_path)

    plt.close("all")


def parse_gcode_lines_deprecated(lines):
    # Rectangle
    lines.insert(0, "G1 X2 Y2")
    lines.insert(1, "G1 X2 Y2")
    lines.insert(2, "G1 X206 Y2")
    lines.insert(3, "G1 X206 Y101")
    lines.insert(4, "G1 X2 Y101")
    lines.insert(5, "G1 X2 Y2")
    # Pen Down
    lines.insert(6, "G1 Z2.5 F25000 ; Pen Down")
    global x_values, y_values, z_values
    x_values = []
    y_values = []
    z_values = []
    for line in lines:
        # skip empty
        if line[0] == "":
            continue
        elif line[0] == " ":
            continue
        # skip comment
        elif line[0] == ";":
            continue
        elif line.startswith("G1"):
            parts = line.split()
            for part in parts:
                if part.startswith("X"):
                    if part[1:] == "":
                        continue
                    x_values.append(float(part[1:]))
                elif part.startswith("Y"):
                    if part[1:] == "":
                        continue
                    y_values.append(float(part[1:]))
                elif part.startswith("Z"):
                    x_values.append(float(0.00))
                    y_values.append(float(0.00))

    return x_values, y_values, z_values


# def visualize_gcode():
#     """Uses the output from parse_gcode_lines_deprecated"""
#     global zoom_factor, line_width, delay, prev_zoom_factor, prev_mouse_x, prev_mouse_y
#
#     root = tk.Tk()
#
#     # Initialize canvas size with default values
#     canvas_width = int(canvas_width_mm * MM_TO_PIXEL)  # Default width for the canvas
#     canvas_height = int(canvas_height_mm * MM_TO_PIXEL)
#
#     canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
#     canvas.pack(expand=tk.YES, fill=tk.BOTH)  # Expanding the canvas to fill the window
#     # Set canvas background color to white
#     canvas.configure(bg="white")
#
#     # Entry widgets for user input
#     line_width_var = tk.StringVar(value=str(line_width))
#     zoom_factor_var = tk.StringVar(value=str(zoom_factor))
#     delay_var = tk.StringVar(value=str(delay))
#
#     line_width_label = tk.Label(root, text="Line Width:")
#     line_width_label.pack()
#     line_width_entry = tk.Entry(root, textvariable=line_width_var)
#     line_width_entry.pack()
#
#     zoom_factor_label = tk.Label(root, text="Zoom Factor:")
#     zoom_factor_label.pack()
#     zoom_factor_entry = tk.Entry(root, textvariable=zoom_factor_var)
#     zoom_factor_entry.pack()
#
#     delay_label = tk.Label(root, text="Delay (ms):")
#     delay_label.pack()
#     delay_entry = tk.Entry(root, textvariable=delay_var)
#     delay_entry.pack()
#
#     apply_button = tk.Button(
#         root,
#         text="Apply Settings",
#         command=lambda: apply_settings(
#             line_width_var.get(), zoom_factor_var.get(), delay_var.get()
#         ),
#     )
#     apply_button.pack()
#
#     # Set the initial values for global variables
#     line_width = int(line_width_var.get())
#     zoom_factor = float(zoom_factor_var.get())
#     delay = int(delay_var.get())
#
#     canvas.create_rectangle(
#         10, 10, canvas_width - 10, canvas_height - 10, outline="black"
#     )
#
#     def apply_settings(new_line_width, new_zoom_factor, new_delay):
#         global line_width, zoom_factor, delay
#         line_width = int(new_line_width)
#         zoom_factor = float(new_zoom_factor)
#         delay = int(new_delay)
#         zoom_label.config(text=f"Zoom Factor: {zoom_factor:.2f}")
#         redraw()
#
#     def on_mouse_wheel(event):
#         global zoom_factor, prev_zoom_factor, prev_mouse_x, prev_mouse_y
#
#         if prev_mouse_x is not None and prev_mouse_y is not None:
#             canvas.xview_scroll(int((prev_mouse_x - event.x) / 2), "units")
#             canvas.yview_scroll(int((prev_mouse_y - event.y) / 2), "units")
#
#         if event.delta > 0:
#             zoom_factor *= 1.1  # Zoom in
#         else:
#             zoom_factor /= 1.1  # Zoom out
#
#         canvas.scale(
#             "all", 0, 0, zoom_factor / prev_zoom_factor, zoom_factor / prev_zoom_factor
#         )
#         prev_zoom_factor = zoom_factor
#
#         zoom_label.config(text=f"Zoom Factor: {zoom_factor:.2f}")
#         prev_mouse_x = event.x
#         prev_mouse_y = event.y
#
#         redraw()
#
#     def redraw(index=0):
#         if index < len(x_values) - 1:
#             x1, y1 = x_values[index] * MM_TO_PIXEL, y_values[index] * MM_TO_PIXEL
#             x2, y2 = (
#                 x_values[index + 1] * MM_TO_PIXEL,
#                 y_values[index + 1] * MM_TO_PIXEL,
#             )
#
#             # canvas.create_line(x1, y1, x2, y2, fill='blue', width=line_width, tag="my_line")
#             if (
#                 x_values[index] == 0
#                 or x_values[index - 1] == 0
#                 or x_values[index + 1] == 0
#             ):
#                 canvas.create_line(x2, y2, x2, y2, fill="white", width=line_width)
#             else:
#                 canvas.create_line(x1, y1, x2, y2, fill="blue", width=line_width)
#
#             root.after(delay, redraw, index + 1)
#
#     # def open_file():
#     #     global x_values, y_values, prev_zoom_factor, prev_mouse_x, prev_mouse_y
#     #
#     #     x_values = []
#     #     y_values = []
#     #     prev_zoom_factor = zoom_factor
#     #     prev_mouse_x = None
#     #     prev_mouse_y = None
#     #
#     #     canvas.delete("all")  # Clear the canvas
#     #     file_path = filedialog.askopenfilename(
#     #         title="Select Gcode file", filetypes=[("Gcode files", "*.gcode")]
#     #     )
#     #     if file_path:
#     #         gcode_lines = read_gcode_file(file_path)
#     #         x_values, y_values, z_values = parse_gcode_lines_deprecated(gcode_lines)
#     #         # Update canvas size based on loaded data
#     #         nonlocal canvas_width, canvas_height
#     #         canvas_width = max(x_values) * zoom_factor
#     #         canvas.config(width=canvas_width)
#     #         # This is a comment in Python code
#     #
#     #         redraw()
#
#     def open_file():
#         global x_values, y_values, prev_zoom_factor, prev_mouse_x, prev_mouse_y
#
#         x_values = []
#         y_values = []
#         prev_zoom_factor = zoom_factor
#         prev_mouse_x = None
#         prev_mouse_y = None
#
#         canvas.delete("all")  # Clear the canvas
#         file_path = filedialog.askopenfilename(
#             title="Select Gcode file", filetypes=[("Gcode files", "*.gcode")]
#         )
#         if file_path:
#             gcode_lines = read_gcode_file(file_path)
#             x_values, y_values, z_values = parse_gcode_lines_deprecated(gcode_lines)
#
#             # 🔽 Add this block to auto-save PNG after loading
#             output_png = file_path + ".png"  # save next to the gcode
#             save_parsed_gcode_as_png(
#                 x_values, y_values, output_png,
#                 line_width=2.0, line_color=(0.1, 0.2, 0.5)
#             )
#             print(f"Saved render as {output_png}")
#
#             # Update canvas size based on loaded data
#             nonlocal canvas_width, canvas_height
#             canvas_width = max(x_values) * zoom_factor
#             canvas.config(width=canvas_width)
#
#             redraw()
#
#     open_button = tk.Button(root, text="Open Gcode File", command=open_file)
#     open_button.pack()
#
#     zoom_label = tk.Label(root, text=f"Zoom Factor: {zoom_factor:.2f}")
#     zoom_label.pack()
#
#     canvas.bind("<MouseWheel>", on_mouse_wheel)
#
#     # Set the window size relative to the canvas size
#     root.geometry(
#         "%dx%d" % (canvas_width, canvas_height + 50)
#     )  # Increased height for button and label
#
#     root.mainloop()

def visualize_gcode():
    import os
    global zoom_factor, line_width, delay, prev_zoom_factor, prev_mouse_x, prev_mouse_y

    root = tk.Tk()

    canvas_width = int(canvas_width_mm * MM_TO_PIXEL)
    canvas_height = int(canvas_height_mm * MM_TO_PIXEL)

    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack(expand=tk.YES, fill=tk.BOTH)

    line_width_var = tk.StringVar(value=str(line_width))
    zoom_factor_var = tk.StringVar(value=str(zoom_factor))
    delay_var = tk.StringVar(value=str(delay))

    tk.Label(root, text="Line Width:").pack()
    tk.Entry(root, textvariable=line_width_var).pack()
    tk.Label(root, text="Zoom Factor:").pack()
    tk.Entry(root, textvariable=zoom_factor_var).pack()
    tk.Label(root, text="Delay (ms):").pack()
    tk.Entry(root, textvariable=delay_var).pack()

    def apply_settings(new_line_width, new_zoom_factor, new_delay):
        global line_width, zoom_factor, delay
        line_width = int(new_line_width)
        zoom_factor = float(new_zoom_factor)
        delay = int(new_delay)
        zoom_label.config(text=f"Zoom Factor: {zoom_factor:.2f}")
        redraw()

    tk.Button(
        root,
        text="Apply Settings",
        command=lambda: apply_settings(
            line_width_var.get(), zoom_factor_var.get(), delay_var.get()
        ),
    ).pack()

    line_width = int(line_width_var.get())
    zoom_factor = float(zoom_factor_var.get())
    delay = int(delay_var.get())

    canvas.create_rectangle(10, 10, canvas_width - 10, canvas_height - 10, outline="black")

    prev_zoom_factor = zoom_factor
    prev_mouse_x = None
    prev_mouse_y = None

    save_path = {"path": None}  # mutable holder

    def on_mouse_wheel(event):
        global zoom_factor, prev_zoom_factor, prev_mouse_x, prev_mouse_y
        if prev_mouse_x is not None and prev_mouse_y is not None:
            canvas.xview_scroll(int((prev_mouse_x - event.x) / 2), "units")
            canvas.yview_scroll(int((prev_mouse_y - event.y) / 2), "units")
        if event.delta > 0:
            zoom_factor *= 1.1
        else:
            zoom_factor /= 1.1
        canvas.scale("all", 0, 0, zoom_factor / prev_zoom_factor, zoom_factor / prev_zoom_factor)
        prev_zoom_factor = zoom_factor
        zoom_label.config(text=f"Zoom Factor: {zoom_factor:.2f}")
        prev_mouse_x = event.x
        prev_mouse_y = event.y
        redraw()

    def redraw(index=0):
        # draw animated lines
        if index < len(x_values) - 1:
            x1, y1 = x_values[index] * MM_TO_PIXEL, y_values[index] * MM_TO_PIXEL
            x2, y2 = x_values[index + 1] * MM_TO_PIXEL, y_values[index + 1] * MM_TO_PIXEL

            if x_values[index] == 0 or x_values[index - 1] == 0 or x_values[index + 1] == 0:
                canvas.create_line(x2, y2, x2, y2, fill="white", width=line_width)
            else:
                canvas.create_line(x1, y1, x2, y2, fill="blue", width=line_width)

            # schedule next frame
            root.after(delay, redraw, index + 1)
        else:
            # === SAVE AFTER RENDER COMPLETES ===
            if save_path["path"]:
                # use your existing saver; this saves a clean PNG (not a canvas screenshot)
                save_parsed_gcode_as_png(
                    x_values,
                    y_values,
                    save_path["path"],
                    line_width=2.0,
                    line_color=(0.1, 0.2, 0.5),
                    face_color="white",
                )
                print(f"Saved render as {os.path.abspath(save_path['path'])}")

    def open_file():
        global x_values, y_values, prev_zoom_factor, prev_mouse_x, prev_mouse_y
        x_values.clear()
        y_values.clear()
        prev_zoom_factor = zoom_factor
        prev_mouse_x = None
        prev_mouse_y = None

        canvas.delete("all")
        file_path = filedialog.askopenfilename(
            title="Select Gcode file", filetypes=[("Gcode files", "*.gcode")]
        )
        if file_path:
            gcode_lines = read_gcode_file(file_path)
            # you were using the deprecated parser for GUI; keep that for the animation
            xs, ys, _ = parse_gcode_lines_deprecated(gcode_lines)
            x_values[:] = xs
            y_values[:] = ys

            nonlocal canvas_width, canvas_height
            canvas_width = max(x_values) * zoom_factor if x_values else canvas_width
            canvas.config(width=canvas_width)

            # set where to save; will be saved AFTER the animation completes
            save_path["path"] = file_path + ".png"

            # start animation
            redraw(0)

    tk.Button(root, text="Open Gcode File", command=open_file).pack()
    zoom_label = tk.Label(root, text=f"Zoom Factor: {zoom_factor:.2f}")
    zoom_label.pack()

    canvas.bind("<MouseWheel>", on_mouse_wheel)
    root.geometry("%dx%d" % (canvas_width, canvas_height + 50))
    root.mainloop()


def main():
    visualize_gcode()


# -----------------------------
# Added: robust CLI renderer
# -----------------------------
import re
import sys
import numpy as np

def _normalize_pen_comments(lines):
    """Normalize pen up/down markers so parse_gcode_lines can recognize them."""
    out = []
    for ln in lines:
        ln2 = re.sub(r'#?\b[Pp]en[_\s-]?Up\b', 'Pen Up', ln)
        ln2 = re.sub(r'#?\b[Pp]en[_\s-]?Down\b', 'Pen Down', ln2)
        out.append(ln2)
    return out

def _detect_units_scale(lines):
    """Return scale to mm. G20 => inches (25.4), else mm (1.0)."""
    is_inches = any(re.search(r'^\s*G20\b', ln) for ln in lines)
    return 25.4 if is_inches else 1.0

def _best_paper_fit(width_mm, height_mm):
    """Pick nearest standard by aspect + closeness."""
    standards = [
        ("A4", (210.0, 297.0)),
        ("A5", (148.0, 210.0)),
        ("DL", (210.0, 99.0)),
        ("DL-rot", (99.0, 210.0)),
    ]
    ar = (width_mm / height_mm) if height_mm else 0.0
    best = None
    best_score = 1e9
    for name, (W, H) in standards:
        ar_c = W / H
        score = abs(ar - ar_c) + 0.01 * abs(W - width_mm) + 0.01 * abs(H - height_mm)
        if score < best_score:
            best_score = score
            best = (name, W, H)
    return best  # (name, W, H)

# def render_gcode_to_png(gcode_path: str, output_png_path: str,
#                         line_color=(0.1, 0.2, 0.5), line_w=1.0):
#     """
#     Robust renderer:
#       - normalizes pen markers,
#       - uses your parse_gcode_lines() (draws pen-down only),
#       - detects inches/mm,
#       - fits to nearest standard paper and renders to PNG with true dimensions.
#     """
#     lines = read_gcode_file(gcode_path)
#     lines = _normalize_pen_comments(lines)
#
#     # Parse with your function (relies on "Pen Up/Down" markers)
#     x_values, y_values = parse_gcode_lines(lines)
#
#     # Flatten to compute extents
#     xs = np.concatenate([np.array(seg) for seg in x_values if len(seg)])
#     ys = np.concatenate([np.array(seg) for seg in y_values if len(seg)])
#
#     min_x, max_x = float(xs.min()), float(xs.max())
#     min_y, max_y = float(ys.min()), float(ys.max())
#     width_raw, height_raw = (max_x - min_x), (max_y - min_y)
#
#     # Units -> mm
#     scale = _detect_units_scale(lines)
#     width_mm, height_mm = width_raw * scale, height_raw * scale
#
#     # Pick paper
#     name, Wmm, Hmm = _best_paper_fit(width_mm, height_mm)
#
#     # Shift so bottom-left of extents is (0,0); no scaling distortion
#     shift_x = -min_x * scale
#     shift_y = -min_y * scale
#
#     # Render at true physical size (inches = mm/25.4)
#     plt.figure(figsize=(Wmm / 25.4, Hmm / 25.4))
#     ax = plt.gca()
#     for seg_x, seg_y in zip(x_values, y_values):
#         if not seg_x:
#             continue
#         sx = (np.array(seg_x) * scale) + shift_x
#         sy = (np.array(seg_y) * scale) + shift_y
#         ax.plot(sx, sy, c=line_color, linewidth=line_w)
#
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlim(0, Wmm)
#     ax.set_ylim(Hmm, 0)  # invert Y for printer-like view
#     ax.axis('off')
#
#     plt.savefig(output_png_path, dpi=300, bbox_inches=None, pad_inches=0.0)
#     plt.close('all')
#     print(f"Saved {output_png_path} ({name}, {Wmm}×{Hmm} mm)")

def render_gcode_to_png(gcode_path: str, output_png_path: str,
                        line_color=(0.1, 0.2, 0.5), line_w=1.0):
    lines = read_gcode_file(gcode_path)
    lines = _normalize_pen_comments(lines)

    x_values, y_values = parse_gcode_lines(lines)
    if not any(x_values):
        print("No pen-down data to render.")
        return

    xs = np.concatenate([np.array(seg) for seg in x_values if seg])
    ys = np.concatenate([np.array(seg) for seg in y_values if seg])
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    width_raw, height_raw = (max_x - min_x), (max_y - min_y)

    scale = _detect_units_scale(lines)
    width_mm, height_mm = width_raw * scale, height_raw * scale
    name, Wmm, Hmm = _best_paper_fit(width_mm, height_mm)

    shift_x = -min_x * scale
    shift_y = -min_y * scale

    Wrot, Hrot = Wmm, Hmm
    def transform(sx, sy):
        return sx, sy

    plt.figure(figsize=(Wrot / 25.4, Hrot / 25.4))
    ax = plt.gca()
    for seg_x, seg_y in zip(x_values, y_values):
        if not seg_x:
            continue
        sx = (np.array(seg_x) * scale) + shift_x
        sy = (np.array(seg_y) * scale) + shift_y
        tx, ty = transform(sx, sy)
        ax.plot(tx, ty, c=line_color, linewidth=line_w)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, Wrot)
    ax.set_ylim(Hrot, 0)
    ax.axis('off')

    plt.savefig(output_png_path, dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close('all')
    #print(f"Saved {output_png_path} [{name} {Wmm}×{Hmm} mm]")
    import os
    plt.savefig(output_png_path, dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close('all')
    print(f"Saved {os.path.abspath(output_png_path)} [{name} {Wmm}×{Hmm} mm]")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # If called with args: run CLI renderer
    if len(sys.argv) >= 3:
        in_gcode = sys.argv[1]
        out_png = sys.argv[2]
        render_gcode_to_png(in_gcode, out_png)
    else:
        # No args: run your original GUI
        main()
