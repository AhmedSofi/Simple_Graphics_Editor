import cv2
import numpy as np
import matplotlib.pyplot as plt

class Editor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    def show_image(self):
        cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize_image(self, width, height):
        self.image = cv2.resize(self.image, (width, height))

    def convert2grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def color_elimination(self, color):
        if len(self.image.shape) == 2:
            print("The image is already in grayscale.")
            return

        if color == "red":
            self.image[:, :, 2] = 0  # Eliminate red channel
        elif color == "green":
            self.image[:, :, 1] = 0  # Eliminate green channel
        elif color == "blue":
            self.image[:, :, 0] = 0  # Eliminate blue channel
        elif color == "yellow":
            self.image[:, :, 2] = 0  # Eliminate red channel
            self.image[:, :, 1] = 0  # Eliminate green channel
        elif color == "cyan":
            self.image[:, :, 1] = 0  # Eliminate green channel
            self.image[:, :, 0] = 0  # Eliminate blue channel
        elif color == "purple":
            self.image[:, :, 0] = 0  # Eliminate blue channel
            self.image[:, :, 2] = 0  # Eliminate green channel
        else:
            print("Invalid color option.")

    def color_channel_swap(self, channel_order):
        if self.image is None:
            print("Open an image first.")
            return None

        img_rgb = self.image.copy()

        if channel_order == 'RGB':
            return img_rgb
        elif channel_order == 'GRB':
            img_rgb[:, :, [1, 0, 2]] = img_rgb[:, :, [0, 1, 2]]  # Swap intensity between R and G channels (GRB)
        elif channel_order == 'RBG':
            img_rgb[:, :, [2, 1, 0]] = img_rgb[:, :, [0, 1, 2]]  # Swap intensity between G and B channels (RBG)
        elif channel_order == 'BGR':
            img_rgb[:, :, [2, 0, 1]] = img_rgb[:, :, [0, 1, 2]]  # Swap intensity between B and R channels (BGR)
        else:
            print("Invalid channel order specified.")

        self.image = img_rgb

    def complement_image(self):
        if len(self.image.shape) == 2:
            self.image = 255 - self.image
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = 255 - self.image

    def solarize_image(self, threshold=120):
        if len(self.image.shape) == 2:
            self.image[self.image < threshold] = self.image[self.image < threshold]
            self.image[self.image >= threshold] = 255 - self.image[self.image >= threshold]
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image[self.image < threshold] = self.image[self.image < threshold]
            self.image[self.image >= threshold] = 255 - self.image[self.image >= threshold]

    def darken_image(self, darkening_factor=0.2):
        if len(self.image.shape) == 2:
            self.image = (self.image * darkening_factor).astype(np.uint8)
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = (self.image * darkening_factor).astype(np.uint8)

    def brighten_image(self, brightening_factor=50):
        if len(self.image.shape) == 2:
            self.image = cv2.add(self.image, brightening_factor).astype(np.uint8)
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.add(self.image, brightening_factor).astype(np.uint8)

    def adjust_brightness(self, brightness_value):
        modified_image = cv2.add(self.image, brightness_value)
        modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        return modified_image

    def adjust_color(self, color, adjustment_value):
        modified_image = self.image.copy()

        if color == 'Red':
            modified_image[:, :, 2] = cv2.add(modified_image[:, :, 2], adjustment_value)
        elif color == 'Green':
            modified_image[:, :, 1] = cv2.add(modified_image[:, :, 1], adjustment_value)
        elif color == 'Blue':
            modified_image[:, :, 0] = cv2.add(modified_image[:, :, 0], adjustment_value)
        elif color == 'Yellow':
            modified_image[:, :, 2] = cv2.add(modified_image[:, :, 2], adjustment_value)
            modified_image[:, :, 1] = cv2.add(modified_image[:, :, 1], adjustment_value)
        elif color == 'Purple':
            modified_image[:, :, 2] = cv2.add(modified_image[:, :, 2], adjustment_value)
            modified_image[:, :, 0] = cv2.add(modified_image[:, :, 0], adjustment_value)
        elif color == 'Cyan':
            modified_image[:, :, 1] = cv2.add(modified_image[:, :, 1], adjustment_value)
            modified_image[:, :, 0] = cv2.add(modified_image[:, :, 0], adjustment_value)
        else:
            print("Invalid color option.")

        modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        return modified_image

    def plot_histogram(self, channels='BGR'):
        image = self.image.copy()
        if channels == 'BGR':
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(histogram, color=col)
                plt.xlim([0, 256])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title('BGR Image Histogram')
        elif channels == 'Grayscale':
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(histogram, color='black')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title('Grayscale Image Histogram')
        elif channels in ['Red', 'Green', 'Blue']:
            channel_idx = {'Red': 2, 'Green': 1, 'Blue': 0}
            idx = channel_idx[channels]
            histogram = cv2.calcHist([image], [idx], None, [256], [0, 256])
            plt.plot(histogram, color=channels.lower())
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title(f'{channels} Channel Histogram')
        else:
            print("Invalid channel option.")

        plt.show()
def increase_contrast(img_path):
    # Read the image in grayscale
    img = cv2.imread(img_path, 0)  # Read the image as grayscale
    
    # Check if the image is successfully loaded
    if img is None:
        print("Could not read the image.")
        return
    
    # Apply histogram equalization to increase contrast
    equalized_img = cv2.equalizeHist(img)
    
    # Display the original and equalized images
    cv2.imshow('Original Image', img)
    cv2.imshow('Equalized Image', equalized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_median_filter(image_path, kernel_size=3):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if img is None:
        print("Could not read the image.")
        return

    # Apply the median filter
    smoothed = cv2.medianBlur(img, kernel_size)

    # Display the original and smoothed images
    cv2.imshow('Original Image', img)
    cv2.imshow('Smoothed Image (Median Filter)', smoothed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to apply average filter for smoothing
def apply_average_filter(image_path, kernel_size=3):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if img is None:
        print("Could not read the image.")
        return

    # Apply the average filter
    smoothed = cv2.blur(img, (kernel_size, kernel_size))

    # Display the original and smoothed images
    cv2.imshow('Original Image', img)
    cv2.imshow('Smoothed Image (Average Filter)', smoothed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to apply Min filter for smoothing
def apply_min_filter(image_path, kernel_size=3):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if img is None:
        print("Could not read the image.")
        return

    # Define the kernel for the min filter
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply the min filter (erosion)
    smoothed = cv2.erode(img, kernel, iterations=1)

    # Display the original and smoothed images
    cv2.imshow('Original Image', img)
    cv2.imshow('Smoothed Image (Min Filter)', smoothed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:


def apply_max_filter(image_path, kernel_size=3):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if img is None:
        print("Could not read the image.")
        return

    # Define the kernel for the max filter
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply the max filter (dilation)
    smoothed = cv2.dilate(img, kernel, iterations=1)

    # Display the original and smoothed images
    cv2.imshow('Original Image', img)
    cv2.imshow('Smoothed Image (Max Filter)', smoothed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sharpen_with_range_filter(image_path, kernel):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the range filter
    sharpened_image = cv2.filter2D(gray_image, -1, kernel)

    # Display the original and sharpened images
    cv2.imshow('Original Image', gray_image)
    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Dilation
def apply_dilation(image_path, kernel_size):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(gray_image, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Erosion
def apply_erosion(image_path, kernel_size):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(gray_image, kernel, iterations=1)
    cv2.imshow('Eroded Image', erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Opening
def apply_opening(image_path, kernel_size):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Opened Image', opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Closing
def apply_closing(image_path, kernel_size):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Closed Image', closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_global_thresholding(image_path, threshold_value):
    image = cv2.imread(image_path, 0)  # Read the image in grayscale
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow('Global Thresholding', thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Adaptive Thresholding Segmentation
def apply_adaptive_thresholding(image_path):
    image = cv2.imread(image_path, 0)  # Read the image in grayscale
    adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('Adaptive Thresholding', adaptive_threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




        
        
