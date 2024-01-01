import tkinter as tk
from tkinter import Scale, filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import *


class GUI:
    def __init__(self):
        self.form = None
        self.root = tk.Tk()
        self.root.title("Image Processing")
        self.root.geometry("800x600")
        self.root.resizable(width=True, height=True)
        self.root.configure(bg='gray')
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.frame = tk.Frame(self.root)
        self.frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=0)
        self.frame.config(bg='gray')
        self.image_label = None
        self.editor = None
        self.create_widgets()
        self.root.mainloop()

    def open_and_display_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.editor = Editor(file_path)
            img = Image.open(file_path)
            
               
            

            img_tk = ImageTk.PhotoImage(img)

            if self.image_label:
                self.image_label.destroy()

            self.image_label = tk.Label(self.frame, image=img_tk)
            self.image_label.image = img_tk
            self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.N)
            if img.width > 800 or img.height > 600:
                img = self.create_resize_form()

    def original_image(self):
        self.editor.image = self.editor.image_copy
        self.update_image_label()

    def apply_operation(self, operation):
        if self.editor:
            if operation == 'Show Image':
                self.editor.show_image()
            elif operation == 'Convert to Grayscale':
                self.editor.convert2grayscale()
                self.update_image_label()
            elif operation == 'Color Elimination':
                self.create_color_elimination_form()
            elif operation == 'Histogram Equalization':
                self.editor.histogram_equalization()
                self.update_image_label()
            elif operation == 'Resize':
                self.create_resize_form()
            elif operation == 'Color Channel Swap':
                self.create_channel_swap_form()
            elif operation == 'Complement Image':
                self.editor.complement_image()
                self.update_image_label()
            elif operation == 'Solarize Image':
                self.create_solarize_form()
            elif operation == 'Darken Image':
                self.create_darken_form()
            elif operation == 'Brighten Image':
                self.create_brighten_form()
            elif operation == 'Adjust Brightness':
                self.create_adjust_brightness_form()
            elif operation == 'Adjust Color':
                self.create_adjust_color_form()
            elif operation == 'Plot Histogram':
                self.create_plot_histogram_form()
            elif operation == 'Increase Contrast':
                self.editor.increase_contrast()
                self.update_image_label()
            elif operation == 'Apply Median Filter':
                self.create_median_filter_form()
            elif operation == 'Apply Min Filter':
                self.create_min_filter_form()
            elif operation == 'Apply Max Filter':
                self.create_max_filter_form()
            elif operation == 'Sharpen with Range Filter':
                self.create_sharpen_with_range_filter_form()
            elif operation == 'Apply Dilation':
                self.create_apply_dilation_form()
            elif operation == 'Apply Erosion':
                self.create_apply_erosion_form()
            elif operation == 'Apply Opening':
                self.create_apply_opening_form()
            elif operation == 'Apply Closing':
                self.create_closing_form()
            elif operation == 'Apply Global Thresholding':
                self.create_global_thresholding_form()
            elif operation == 'Apply Adaptive Thresholding':
                self.create_adaptive_thresholding_form()

    def create_adaptive_thresholding_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Adaptive Thresholding")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        block_size_label = tk.Label(self.form, text="Block Size:")
        block_size_label.grid(row=0, column=0, padx=10, pady=5)

        self.block_size_var = tk.IntVar(self.form)
        self.block_size_var.set(11)

        block_size_slider = tk.Scale(self.form, from_=3, to=21, orient='horizontal', variable=self.block_size_var,
                                     tickinterval=2)
        block_size_slider.grid(row=0, column=1, padx=10, pady=5)

        constant_label = tk.Label(self.form, text="Constant Value:")
        constant_label.grid(row=1, column=0, padx=10, pady=5)

        self.constant_var = tk.IntVar(self.form)
        self.constant_var.set(2)

        constant_slider = tk.Scale(self.form, from_=1, to=10, orient='horizontal', variable=self.constant_var)
        constant_slider.grid(row=1, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Adaptive Thresholding",
                                  command=self.apply_adaptive_thresholding_filter)
        submit_button.grid(row=2, column=1, padx=10, pady=5)

    def apply_adaptive_thresholding_filter(self):
        block_size = self.block_size_var.get()
        constant = self.constant_var.get()
        if self.editor:
            self.editor.apply_adaptive_thresholding(block_size, constant)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_global_thresholding_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Global Thresholding")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        threshold_label = tk.Label(self.form, text="Threshold Value:")
        threshold_label.grid(row=0, column=0, padx=10, pady=5)

        self.threshold_var = tk.IntVar(self.form)
        self.threshold_var.set(128)

        threshold_slider = tk.Scale(self.form, from_=0, to=255, orient='horizontal', variable=self.threshold_var)
        threshold_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Thresholding",
                                  command=self.apply_global_thresholding_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_global_thresholding_filter(self):
        threshold_value = self.threshold_var.get()
        if self.editor:
            self.editor.apply_global_thresholding(threshold_value)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_closing_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Closing")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        kernel_label = tk.Label(self.form, text="Kernel Size:")
        kernel_label.grid(row=0, column=0, padx=10, pady=5)

        self.kernel_var = tk.IntVar(self.form)
        self.kernel_var.set(1)

        kernel_slider = tk.Scale(self.form, from_=1, to=15, orient='horizontal', variable=self.kernel_var)
        kernel_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Closing", command=self.apply_closing_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_closing_filter(self):
        kernel_size = self.kernel_var.get()
        if self.editor:
            self.editor.apply_closing(kernel_size)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_apply_opening_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Opening")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        kernel_label = tk.Label(self.form, text="Kernel Size:")
        kernel_label.grid(row=0, column=0, padx=10, pady=5)

        self.kernel_var = tk.IntVar(self.form)
        self.kernel_var.set(1)

        kernel_slider = tk.Scale(self.form, from_=1, to=15, orient='horizontal', variable=self.kernel_var)
        kernel_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Opening", command=self.apply_opening_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_opening_filter(self):
        kernel_size = self.kernel_var.get()
        if self.editor:
            self.editor.apply_opening(kernel_size)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_apply_erosion_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Erosion")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        kernel_label = tk.Label(self.form, text="Kernel Size:")
        kernel_label.grid(row=0, column=0, padx=10, pady=5)

        self.kernel_var = tk.IntVar(self.form)
        self.kernel_var.set(1)

        kernel_slider = tk.Scale(self.form, from_=1, to=15, orient='horizontal', variable=self.kernel_var)
        kernel_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Erosion", command=self.apply_erosion_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_erosion_filter(self):
        kernel_size = self.kernel_var.get()
        if self.editor:
            self.editor.apply_erosion(kernel_size)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_apply_dilation_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Dilation")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        kernel_label = tk.Label(self.form, text="Kernel Size:")
        kernel_label.grid(row=0, column=0, padx=10, pady=5)

        self.kernel_var = tk.IntVar(self.form)
        self.kernel_var.set(1)

        kernel_slider = tk.Scale(self.form, from_=1, to=15, orient='horizontal', variable=self.kernel_var)
        kernel_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Dilation", command=self.apply_dilation_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_dilation_filter(self):
        kernel_size = self.kernel_var.get()
        if self.editor:
            self.editor.apply_dilation(kernel_size)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_sharpen_with_range_filter_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Sharpen with Range Filter")
        self.form.geometry("300x200")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        kernel_label = tk.Label(self.form, text="Kernel Size:")
        kernel_label.grid(row=0, column=0, padx=10, pady=5)

        self.kernel_var = tk.IntVar(self.form)
        self.kernel_var.set(1)

        kernel_slider = tk.Scale(self.form, from_=1, to=15, orient='horizontal', variable=self.kernel_var)
        kernel_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Sharpen Filter", command=self.apply_sharpen_with_range_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_sharpen_with_range_filter(self):

        kernel_size = self.kernel_var.get()
        if self.editor:
            self.editor.sharpen_with_range_filter(int(kernel_size))
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_max_filter_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Max Filter")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        kernel_label = tk.Label(self.form, text="Kernel Size:")
        kernel_label.grid(row=0, column=0, padx=10, pady=5)

        self.kernel_var = tk.IntVar(self.form)
        self.kernel_var.set(1)

        kernel_slider = tk.Scale(self.form, from_=1, to=15, orient='horizontal', variable=self.kernel_var)
        kernel_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Max Filter", command=self.apply_max_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_max_filter(self):
        kernel_size = self.kernel_var.get()
        if self.editor:
            self.editor.apply_max_filter(kernel_size)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_min_filter_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Min Filter")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        kernel_label = tk.Label(self.form, text="Kernel Size:")
        kernel_label.grid(row=0, column=0, padx=10, pady=5)

        self.kernel_var = tk.IntVar(self.form)
        self.kernel_var.set(1)

        kernel_slider = tk.Scale(self.form, from_=1, to=15, orient='horizontal', variable=self.kernel_var)
        kernel_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply min Filter", command=self.apply_min_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_min_filter(self):
        kernel_size = self.kernel_var.get()
        if self.editor:
            self.editor.apply_min_filter(kernel_size)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_median_filter_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Apply Median Filter")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        kernel_label = tk.Label(self.form, text="Kernel Size:")
        kernel_label.grid(row=0, column=0, padx=10, pady=5)

        self.kernel_var = tk.IntVar(self.form)
        self.kernel_var.set(1)  # Default value

        kernel_slider = tk.Scale(self.form, from_=1, to=15, orient='horizontal', variable=self.kernel_var)
        kernel_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Median Filter", command=self.apply_median_filter)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_median_filter(self):
        kernel_size = self.kernel_var.get()
        if self.editor:
            self.editor.apply_median_filter(kernel_size)
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_plot_histogram_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Plot Histogram")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        channel_label = tk.Label(self.form, text="Select Channel:")
        channel_label.grid(row=0, column=0, padx=10, pady=5)

        self.channel_var = tk.StringVar(self.form)
        channels = ['BGR', 'Grayscale', 'Red', 'Green', 'Blue']
        self.channel_var.set(channels[0])  # Default value

        channel_dropdown = tk.OptionMenu(self.form, self.channel_var, *channels)
        channel_dropdown.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Plot Histogram", command=self.plot_histogram)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def plot_histogram(self):
        channel = self.channel_var.get()
        if self.editor:
            self.editor.plot_histogram(channel)
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def create_adjust_color_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Adjust Color")
        self.form.geometry("300x200")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        color_label = tk.Label(self.form, text="Select Color:")
        color_label.grid(row=0, column=0, padx=10, pady=5)

        self.color_var = tk.StringVar(self.form)
        colors = ['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Cyan']
        self.color_var.set(colors[0])  # Default value

        color_dropdown = tk.OptionMenu(self.form, self.color_var, *colors)
        color_dropdown.grid(row=0, column=1, padx=10, pady=5)

        value_label = tk.Label(self.form, text="Adjustment Value:")
        value_label.grid(row=1, column=0, padx=10, pady=5)

        self.value_slider = tk.Scale(self.form, from_=-100, to=100, orient='horizontal')
        self.value_slider.set(0)  # Default value
        self.value_slider.grid(row=1, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Color Adjustment", command=self.apply_color_adjustment)
        submit_button.grid(row=2, column=1, padx=10, pady=5)

    def apply_color_adjustment(self):
        color = self.color_var.get()
        adjustment_value = int(self.value_slider.get())
        if self.editor:
            modified_image = self.editor.adjust_color(color, adjustment_value)
            img = Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
            img = img.resize((img.width // 3, img.height // 3))
            img_tk = ImageTk.PhotoImage(img)

            if self.image_label:
                self.image_label.destroy()

            self.image_label = tk.Label(self.frame, image=img_tk)
            self.image_label.image = img_tk
            self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.N)
            self.form.destroy()

    def create_adjust_brightness_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Adjust Brightness")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        value_label = tk.Label(self.form, text="Brightness Value:")
        value_label.grid(row=0, column=0, padx=10, pady=5)

        self.value_slider = tk.Scale(self.form, from_=-100, to=100, orient='horizontal')
        self.value_slider.set(0)  # Default value
        self.value_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Brightness", command=self.apply_brightness)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_brightness(self):
        brightness_value = int(self.value_slider.get())
        if self.editor:
            modified_image = self.editor.adjust_brightness(brightness_value)
            img = Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
            img = img.resize((img.width // 3, img.height // 3))
            img_tk = ImageTk.PhotoImage(img)

            if self.image_label:
                self.image_label.destroy()

            self.image_label = tk.Label(self.frame, image=img_tk)
            self.image_label.image = img_tk
            self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.N)
            self.form.destroy()

    def create_brighten_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Brighten Image")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        factor_label = tk.Label(self.form, text="Brightening Factor:")
        factor_label.grid(row=0, column=0, padx=10, pady=5)

        self.factor_slider = tk.Scale(self.form, from_=0, to=100, orient='horizontal')
        self.factor_slider.set(50)
        self.factor_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Brighten", command=self.apply_brighten)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_brighten(self):
        brightening_factor = int(self.factor_slider.get())
        if self.editor:
            self.editor.brighten_image(brightening_factor)
            self.update_image_label()
            self.form.destroy()

    def create_darken_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Darken Image")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        factor_label = tk.Label(self.form, text="Darkening Factor:")
        factor_label.grid(row=0, column=0, padx=10, pady=5)

        self.factor_slider = Scale(self.form, from_=0, to=1, resolution=0.1, orient='horizontal')
        self.factor_slider.set(0.5)  # Default value
        self.factor_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Darken", command=self.apply_darken)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_darken(self):
        darkening_factor = float(self.factor_slider.get())
        if self.editor:
            self.editor.darken_image(darkening_factor)
            self.update_image_label()
            self.form.destroy()

    def create_solarize_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Solarize Image")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        threshold_label = tk.Label(self.form, text="Threshold:")
        threshold_label.grid(row=0, column=0, padx=10, pady=5)

        self.threshold_slider = Scale(self.form, from_=0, to=255, orient='horizontal')
        self.threshold_slider.set(120)  # Default value
        self.threshold_slider.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Apply Solarize", command=self.apply_solarize)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def apply_solarize(self):
        threshold_value = int(self.threshold_slider.get())
        if self.editor:
            self.editor.solarize_image(threshold_value)
            self.update_image_label()
            self.form.destroy()

    def create_channel_swap_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Color Channel Swap")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        channel_label = tk.Label(self.form, text="Channel Order:")
        channel_label.grid(row=0, column=0, padx=10, pady=5)

        self.channel_entry = tk.Entry(self.form)
        self.channel_entry.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Swap Channels", command=self.swap_channels)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def swap_channels(self):
        channel_order = self.channel_entry.get()
        result = self.editor.colour_channel_swap(channel_order.upper())
        if result is not None:
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Invalid channel order specified." + channel_order.upper())

    def create_color_elimination_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Color Elimination")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        color_label = tk.Label(self.form, text="Color:")
        color_label.grid(row=0, column=0, padx=10, pady=5)

        self.color_entry = tk.Entry(self.form)
        self.color_entry.grid(row=0, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Eliminate Color", command=self.eliminate_color)
        submit_button.grid(row=1, column=1, padx=10, pady=5)

    def eliminate_color(self):
        color = self.color_entry.get()
        if color.lower() in ['red', 'green', 'blue', 'yellow', 'cyan', 'purple']:
            self.editor.color_elimination(color.lower())
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Invalid color option. Please enter a valid color.")

    def create_resize_form(self):
        self.form = tk.Toplevel(self.root)
        self.form.title("Resize Image")
        self.form.geometry("300x150")
        self.form.resizable(width=False, height=False)
        self.form.grid_rowconfigure(0, weight=1)
        self.form.grid_columnconfigure(0, weight=1)

        width_label = tk.Label(self.form, text="Width:")
        width_label.grid(row=0, column=0, padx=10, pady=5)
        self.width_entry = tk.Entry(self.form)
        self.width_entry.grid(row=0, column=1, padx=10, pady=5)

        height_label = tk.Label(self.form, text="Height:")
        height_label.grid(row=1, column=0, padx=10, pady=5)
        self.height_entry = tk.Entry(self.form)
        self.height_entry.grid(row=1, column=1, padx=10, pady=5)

        submit_button = tk.Button(self.form, text="Resize", command=self.resize_image)
        submit_button.grid(row=2, column=1, padx=10, pady=5)

    def resize_image(self):
        width = self.width_entry.get()
        height = self.height_entry.get()
        if width.isdigit() and height.isdigit():
            self.editor.resize_image(int(width), int(height))
            self.update_image_label()
            self.form.destroy()
        else:
            messagebox.showerror("Error", "Please enter valid numeric values for width and height.")

    def update_image_label(self):
        img = self.editor.image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # img = img.resize((img.width // 3, img.height // 3))
        img_tk = ImageTk.PhotoImage(img)

        if self.image_label:
            self.image_label.destroy()

        self.image_label = tk.Label(self.frame, image=img_tk)
        self.image_label.image = img_tk
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.N)

    def create_widgets(self):
        import_button = tk.Button(self.frame, text="Import Image", command=self.open_and_display_image, bg='#545454'
                                  , width=20, height=2, font=13)
        import_button.grid(row=1, column=0, padx=2, pady=10)
        import_button = tk.Button(self.frame, text="Original Image", command=self.original_image, bg='#545454'
                                  , width=20, height=2, font=13)
        import_button.grid(row=2, column=0, padx=10, pady=10)
        self.type_Var = tk.StringVar()
        operations = ['Show Image', 'Convert to Grayscale', 'Color Elimination', 'Resize', 'Color Channel Swap',
                      'Complement Image', 'Solarize Image', 'Darken Image', 'Brighten Image', 'Adjust Brightness',
                      'Adjust Color', 'Plot Histogram', 'Increase Contrast', 'Apply Median Filter', 'Apply Min Filter',
                      'Apply Max Filter', 'Sharpen with Range Filter', 'Apply Dilation', 'Apply Erosion',
                      'Apply Opening', 'Apply Closing', 'Apply Global Thresholding', 'Apply Adaptive Thresholding']
        type_dropdown = ttk.Combobox(self.frame, textvariable=self.type_Var, values=operations, state='readonly',
                                     font=15)
        type_dropdown.grid(row=2, column=1, padx=10, pady=10)
        import_button = tk.Button(self.frame, text="apply", command=lambda: self.apply_operation(self.type_Var.get())
                                  , bg='#545454', width=20, height=2, font=13)
        import_button.grid(row=1, column=1, padx=10, pady=10)


if __name__ == "__main__":
    GUI()
