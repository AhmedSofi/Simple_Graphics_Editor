import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        self.img_original = None
        self.img_display = ImageTk.PhotoImage(Image.new('RGB', (300, 300)))
        self.display_img = canvas.create_image(0, 0, anchor='nw', image=self.img_display)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img_original = cv2.imread(file_path)
            self.img_original = cv2.resize(self.img_original, (300, 300))

            self.update_display()

    def adjust_brightness(self, value):
        brightness = int(value)
        modified_image = cv2.add(self.img_original, brightness)
        modified_image = np.clip(modified_image, 0, 255)

        self.img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)))
        canvas.itemconfig(self.display_img, image=self.img_display)

    def adjust_color(self, color):
        modified_image = self.img_original.copy()

        if color == 'Red':
            modified_image[:, :, 0] = np.clip(modified_image[:, :, 0] + 50, 0, 255)
        elif color == 'Green':
            modified_image[:, :, 1] = np.clip(modified_image[:, :, 1] + 50, 0, 255)
        elif color == 'Blue':
            modified_image[:, :, 2] = np.clip(modified_image[:, :, 2] + 50, 0, 255)

        self.img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)))
        canvas.itemconfig(self.display_img, image=self.img_display)

    def eliminate_color(self, color):
        modified_image = self.img_original.copy()

        # Set image to grayscale and convert it back to BGR
        grayscale_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
        modified_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

        # Define color channel elimination based on the desired color
        if color == "red":
            modified_image[:, :, 2] = 0  # Eliminate red channel
        elif color == "green":
            modified_image[:, :, 1] = 0  # Eliminate green channel
        elif color == "blue":
            modified_image[:, :, 0] = 0  # Eliminate blue channel
        elif color == "yellow":
            modified_image[:, :, 2] = 0  # Eliminate red channel
            modified_image[:, :, 1] = 0  # Eliminate green channel
        elif color == "cyan":
            modified_image[:, :, 1] = 0  # Eliminate green channel
            modified_image[:, :, 0] = 0  # Eliminate blue channel
        elif color == "purple":
            modified_image[:, :, 0] = 0  # Eliminate blue channel
            modified_image[:, :, 1] = 0  # Eliminate green channel
        else:
            print("Invalid color option.")
            return

        self.img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)))
        canvas.itemconfig(self.display_img, image=self.img_display)

    def eliminate_cyan(self):
        self.eliminate_color('cyan')

    def eliminate_purple(self):
        self.eliminate_color("purple")

    def eliminate_yellow(self):
        self.eliminate_color('yellow')

    def eliminate_blue(self):
        self.eliminate_color('blue')

    def eliminate_red(self):
        self.eliminate_color('red')

    def eliminate_green(self):
        self.eliminate_color('green')

    def color_channel_swap(self, channel_order):
        if channel_order == 'RGB':
            if not np.array_equal(self.img_original, self.img_display_to_array()):
                self.img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)))
                canvas.itemconfig(self.display_img, image=self.img_display)
            else:
                print("No swapping needed for RGB order.")
            return

        img_rgb = self.img_original.copy()

        if channel_order == 'GRB':
            img_rgb[:, :, [0, 1, 2]] = img_rgb[:, :, [1, 0, 2]]  # Swap intensity between R and G channels (GRB)
        elif channel_order == 'RBG':
            img_rgb[:, :, [0, 1, 2]] = img_rgb[:, :, [2, 1, 0]]  # Swap intensity between G and B channels (RBG)
        elif channel_order == 'BGR':
            img_rgb[:, :, [0, 1, 2]] = img_rgb[:, :, [2, 0, 1]]  # Swap intensity between B and R channels (BGR)
        else:
            print("Invalid channel order specified.")
            return

        self.img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)))
        canvas.itemconfig(self.display_img, image=self.img_display)

    def img_display_to_array(self):
        return np.array(self.img_display)

    def update_display(self):
        self.img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)))
        canvas.itemconfig(self.display_img, image=self.img_display)

def plot_histogram():
    global image_processor

    if image_processor.img_original is not None:
        plt.figure(figsize=(8, 4))

        if image_processor.img_original.ndim == 2:
            # Grayscale image histogram
            hist = cv2.calcHist([image_processor.img_original], [0], None, [256], [0, 256])
            plt.plot(hist)
            plt.title('Grayscale Image Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
        elif image_processor.img_original.ndim == 3:
            # BGR image histogram for each channel
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([image_processor.img_original], [i], None, [256], [0, 256])
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

# Create an instance of ImageProcessor
image_processor = ImageProcessor()

# Buttons for color adjustments
color_buttons = tk.Frame(root)
color_buttons.pack()

color_labels = ['Red', 'Green', 'Blue']
for label in color_labels:
    button = tk.Button(color_buttons, text=label, command=lambda label=label: image_processor.adjust_color(label))
    button.pack(side='left', padx=5, pady=5)

# Buttons for color elimination
elimination_buttons = tk.Frame(root)
elimination_buttons.pack()

elimination_labels = ['Cyan', 'purple', 'Yellow', 'Blue', 'Red', 'Green']
elimination_functions = ['eliminate_cyan', 'eliminate_purple', 'eliminate_yellow', 'eliminate_blue', 'eliminate_red', 'eliminate_green']

for label, function_name in zip(elimination_labels, elimination_functions):
    button = tk.Button(elimination_buttons, text=f'Eliminate {label}', command=lambda func=function_name: getattr(image_processor, func)())
    button.pack(side='left', padx=5, pady=5)

# Buttons for color channel swap
swap_buttons = tk.Frame(root)
swap_buttons.pack()

swap_labels = ['RGB', 'GRB', 'RBG', 'BGR']
for label in swap_labels:
    button = tk.Button(swap_buttons, text=f'Channel Swap ({label})', command=lambda label=label: image_processor.color_channel_swap(label))
    button.pack(side='left', padx=5, pady=5)

# Slider for brightness adjustment
brightness_slider = tk.Scale(root, from_=0, to=100, orient='horizontal', label='Brightness',
                             command=image_processor.adjust_brightness)
brightness_slider.pack()

# Buttons for image operations
operation_buttons = tk.Frame(root)
operation_buttons.pack()

open_button = tk.Button(operation_buttons, text="Open Image", command=image_processor.open_image)
open_button.pack(side='left', padx=5, pady=5)

histogram_button = tk.Button(operation_buttons, text="Plot Histogram", command=plot_histogram)
histogram_button.pack(side='left', padx=5, pady=5)

root.mainloop()




