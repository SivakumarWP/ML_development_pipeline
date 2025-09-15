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


def visualize_gcode():
    """Uses the output from parse_gcode_lines_deprecated"""
    global zoom_factor, line_width, delay, prev_zoom_factor, prev_mouse_x, prev_mouse_y

    root = tk.Tk()

    # Initialize canvas size with default values
    canvas_width = int(canvas_width_mm * MM_TO_PIXEL)  # Default width for the canvas
    canvas_height = int(canvas_height_mm * MM_TO_PIXEL)

    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack(expand=tk.YES, fill=tk.BOTH)  # Expanding the canvas to fill the window
    # Set canvas background color to white
    canvas.configure(bg="white")

    # Entry widgets for user input
    line_width_var = tk.StringVar(value=str(line_width))
    zoom_factor_var = tk.StringVar(value=str(zoom_factor))
    delay_var = tk.StringVar(value=str(delay))

    line_width_label = tk.Label(root, text="Line Width:")
    line_width_label.pack()
    line_width_entry = tk.Entry(root, textvariable=line_width_var)
    line_width_entry.pack()

    zoom_factor_label = tk.Label(root, text="Zoom Factor:")
    zoom_factor_label.pack()
    zoom_factor_entry = tk.Entry(root, textvariable=zoom_factor_var)
    zoom_factor_entry.pack()

    delay_label = tk.Label(root, text="Delay (ms):")
    delay_label.pack()
    delay_entry = tk.Entry(root, textvariable=delay_var)
    delay_entry.pack()

    apply_button = tk.Button(
        root,
        text="Apply Settings",
        command=lambda: apply_settings(
            line_width_var.get(), zoom_factor_var.get(), delay_var.get()
        ),
    )
    apply_button.pack()

    # Set the initial values for global variables
    line_width = int(line_width_var.get())
    zoom_factor = float(zoom_factor_var.get())
    delay = int(delay_var.get())

    canvas.create_rectangle(
        10, 10, canvas_width - 10, canvas_height - 10, outline="black"
    )

    def apply_settings(new_line_width, new_zoom_factor, new_delay):
        global line_width, zoom_factor, delay
        line_width = int(new_line_width)
        zoom_factor = float(new_zoom_factor)
        delay = int(new_delay)
        zoom_label.config(text=f"Zoom Factor: {zoom_factor:.2f}")
        redraw()

    def on_mouse_wheel(event):
        global zoom_factor, prev_zoom_factor, prev_mouse_x, prev_mouse_y

        if prev_mouse_x is not None and prev_mouse_y is not None:
            canvas.xview_scroll(int((prev_mouse_x - event.x) / 2), "units")
            canvas.yview_scroll(int((prev_mouse_y - event.y) / 2), "units")

        if event.delta > 0:
            zoom_factor *= 1.1  # Zoom in
        else:
            zoom_factor /= 1.1  # Zoom out

        canvas.scale(
            "all", 0, 0, zoom_factor / prev_zoom_factor, zoom_factor / prev_zoom_factor
        )
        prev_zoom_factor = zoom_factor

        zoom_label.config(text=f"Zoom Factor: {zoom_factor:.2f}")
        prev_mouse_x = event.x
        prev_mouse_y = event.y

        redraw()

    def redraw(index=0):
        if index < len(x_values) - 1:
            x1, y1 = x_values[index] * MM_TO_PIXEL, y_values[index] * MM_TO_PIXEL
            x2, y2 = (
                x_values[index + 1] * MM_TO_PIXEL,
                y_values[index + 1] * MM_TO_PIXEL,
            )

            # canvas.create_line(x1, y1, x2, y2, fill='blue', width=line_width, tag="my_line")
            if (
                x_values[index] == 0
                or x_values[index - 1] == 0
                or x_values[index + 1] == 0
            ):
                canvas.create_line(x2, y2, x2, y2, fill="white", width=line_width)
            else:
                canvas.create_line(x1, y1, x2, y2, fill="blue", width=line_width)

            root.after(delay, redraw, index + 1)

    def open_file():
        global x_values, y_values, prev_zoom_factor, prev_mouse_x, prev_mouse_y

        x_values = []
        y_values = []
        prev_zoom_factor = zoom_factor
        prev_mouse_x = None
        prev_mouse_y = None

        canvas.delete("all")  # Clear the canvas
        file_path = filedialog.askopenfilename(
            title="Select Gcode file", filetypes=[("Gcode files", "*.gcode")]
        )
        if file_path:
            gcode_lines = read_gcode_file(file_path)
            x_values, y_values, z_values = parse_gcode_lines_deprecated(gcode_lines)
            # Update canvas size based on loaded data
            nonlocal canvas_width, canvas_height
            canvas_width = max(x_values) * zoom_factor
            canvas.config(width=canvas_width)
            # This is a comment in Python code

            redraw()

    open_button = tk.Button(root, text="Open Gcode File", command=open_file)
    open_button.pack()

    zoom_label = tk.Label(root, text=f"Zoom Factor: {zoom_factor:.2f}")
    zoom_label.pack()

    canvas.bind("<MouseWheel>", on_mouse_wheel)

    # Set the window size relative to the canvas size
    root.geometry(
        "%dx%d" % (canvas_width, canvas_height + 50)
    )  # Increased height for button and label

    root.mainloop()


def main():
    visualize_gcode()


if __name__ == "__main__":
    main()
