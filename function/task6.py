import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def adjust_brightness(value):
    global img_original
    global img_display

    brightness = int(value)
    modified_image = cv2.add(img_original, brightness)
    modified_image = np.clip(modified_image, 0, 255)

    img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)))
    canvas.itemconfig(display_img, image=img_display)

def adjust_color(color):
    global img_original
    global img_display

    modified_image = img_original.copy()

    if color == 'Red':
        modified_image[:, :, 0] = np.clip(modified_image[:, :, 0] + 50, 0, 255)
    elif color == 'Green':
        modified_image[:, :, 1] = np.clip(modified_image[:, :, 1] + 50, 0, 255)
    elif color == 'Blue':
        modified_image[:, :, 2] = np.clip(modified_image[:, :, 2] + 50, 0, 255)
    # Similarly, you can add adjustments for other colors

    img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)))
    canvas.itemconfig(display_img, image=img_display)

def open_image():
    global img_original
    global img_display

    file_path = filedialog.askopenfilename()
    if file_path:
        img_original = cv2.imread(file_path)
        img_original = cv2.resize(img_original, (300, 300))

        img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)))
        canvas.itemconfig(display_img, image=img_display)

def plot_histogram():
    global img_original

    if img_original is not None:
        plt.figure(figsize=(8, 4))

        if img_original.ndim == 2:
            # Grayscale image histogram
            hist = cv2.calcHist([img_original], [0], None, [256], [0, 256])
            plt.plot(hist)
            plt.title('Grayscale Image Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
        elif img_original.ndim == 3:
            # BGR image histogram for each channel
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img_original], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
            plt.title('BGR Image Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')

        plt.show()

# Create a Tkinter window
root = tk.Tk()
root.title("Image Adjustments")

# Create a canvas to display the image
canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

# Buttons for color adjustments
color_buttons = tk.Frame(root)
color_buttons.pack()

color_labels = ['Red', 'Green', 'Blue'  ]
for label in color_labels:
    button = tk.Button(color_buttons, text=label, command=lambda label=label: adjust_color(label))
    button.pack(side='left', padx=5, pady=5)

# Slider for brightness adjustment
brightness_slider = tk.Scale(root, from_=0, to=100, orient='horizontal', label='Brightness',
                             command=adjust_brightness)
brightness_slider.pack()

# Buttons for image operations
operation_buttons = tk.Frame(root)
operation_buttons.pack()

open_button = tk.Button(operation_buttons, text="Open Image", command=open_image)
open_button.pack(side='left', padx=5, pady=5)

histogram_button = tk.Button(operation_buttons, text="Plot Histogram", command=plot_histogram)
histogram_button.pack(side='left', padx=5, pady=5)

# Placeholder for displayed image
img_display = ImageTk.PhotoImage(Image.new('RGB', (300, 300)))
display_img = canvas.create_image(0, 0, anchor='nw', image=img_display)

img_original = None

root.mainloop()
