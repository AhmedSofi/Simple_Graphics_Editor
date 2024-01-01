import cv2
import numpy as np
import matplotlib.pyplot as plt


class Editor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image_copy = cv2.imread(image_path)

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

    def colour_channel_swap(self, channel_order):
        if self.image is None:
            print("Open an image first.")
            return None

        img_rgb = self.image.copy()

        if channel_order == 'RGB':
            return img_rgb
        elif channel_order == 'GRB':
            img_rgb[:, :, [1, 0, 2]] = img_rgb[:, :, [0, 1, 2]]  # Swap intensity between R and G channels (GRB)
        elif channel_order == 'RBG':
            img_rgb[:, :, [0, 2, 1]] = img_rgb[:, :, [0, 1, 2]]  # Swap intensity between G and B channels (RBG)
        elif channel_order == 'BGR':
            img_rgb[:, :, [2, 0, 1]] = img_rgb[:, :, [0, 1, 2]]  # Swap intensity between B and R channels (BGR)
        elif channel_order == 'BRG':
            img_rgb[:, :, [2, 1, 0]] = img_rgb[:, :, [0, 1, 2]]  # Swap intensity between R and B channels (BRG)
        elif channel_order == 'GBR':
            img_rgb[:, :, [0, 2, 1]] = img_rgb[:, :, [0, 1, 2]]  # Swap intensity between B and G channels (GBR)
        else:
            print("Invalid channel order specified.")

        self.image = img_rgb
        return 1

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

    def darken_image(self, darkening_factor):
        self.image = self.image_copy.copy()
        if len(self.image.shape) == 2:
            self.image = (self.image * darkening_factor).astype(np.uint8)
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = (self.image * darkening_factor).astype(np.uint8)

    def brighten_image(self, brightening_factor):
        self.image = self.image_copy.copy()
        if len(self.image.shape) == 2:
            self.image = cv2.add(self.image, brightening_factor).astype(np.uint8)
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.add(self.image, brightening_factor).astype(np.uint8)

    def adjust_brightness(self, brightness_value):
        self.image = self.image_copy.copy()
        modified_image = cv2.add(self.image, brightness_value)
        modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        self.image = modified_image

    def adjust_color(self, color, adjustment_value):
        self.image = self.image_copy.copy()
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
        self.image = modified_image

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

    def increase_contrast(self):
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.equalizeHist(gray_image)
        elif len(self.image.shape) == 2:
            self.image = cv2.equalizeHist(self.image)
        else:
            print("Unsupported image format for contrast adjustment.")

    def apply_median_filter(self, kernel_size):
        self.image = self.image_copy.copy()
        if self.image is None:
            print("Could not read the image.")
            return None
        self.image = cv2.medianBlur(self.image, kernel_size)

    def apply_min_filter(self, kernel_size=3):
        self.image = self.image_copy.copy()
        if self.image is None:
            print("Could not read the image.")
            return None

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        self.image = cv2.erode(self.image, kernel, iterations=1)

    def apply_max_filter(self, kernel_size=3):
        self.image = self.image_copy.copy()
        if self.image is None:
            print("Could not read the image.")
            return None

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        self.image = cv2.dilate(self.image, kernel, iterations=1)

    def sharpen_with_range_filter(self, kernel):
        self.image = self.image_copy.copy()
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.image = cv2.filter2D(gray_image, -1, kernel)

    def apply_dilation(self, kernel_size):
        self.image = self.image_copy.copy()
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.dilate(gray_image, kernel, iterations=1)

    def apply_erosion(self, kernel_size):
        self.image = self.image_copy.copy()
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.erode(gray_image, kernel, iterations=1)

    def apply_opening(self, kernel_size):
        self.image = self.image_copy.copy()
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

    def apply_closing(self, kernel_size):
        self.image = self.image_copy.copy()
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    def apply_global_thresholding(self, threshold_value):
        self.image = self.image_copy.copy()
        _, self.image = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY)

    def apply_adaptive_thresholding(self, block_size, constant):
        self.image = self.image_copy.copy()

        # Convert to grayscale if the image is not already in grayscale
        if len(self.image.shape) > 2 and self.image.shape[2] > 1:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Ensure the image is of type CV_8UC1
        if self.image.dtype != np.uint8:
            self.image = np.clip(self.image, 0, 255).astype(np.uint8)

        self.image = cv2.adaptiveThreshold(
            self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant
        )

