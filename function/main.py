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




        
        
