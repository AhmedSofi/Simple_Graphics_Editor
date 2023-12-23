import cv2
import numpy as np
import matplotlib as plt


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

        # Define color channel elimination based on the desired color
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

        # Return the original image for RGB order
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
            # Grayscale image complement
            self.image = 255 - self.image
        else:
            # convert from BGR to Grayscale
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Grayscale image complement
            self.image = 255 - self.image

    def solarize_image(self, threshold=120):
        if len(self.image.shape) == 2:
            # Grayscale image solarization
            self.image[self.image < threshold] = self.image[self.image < threshold]
            self.image[self.image >= threshold] = 255 - self.image[self.image >= threshold]
        else:
            # convert from BGR to Grayscale
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Grayscale image solarization
            self.image[self.image < threshold] = self.image[self.image < threshold]
            self.image[self.image >= threshold] = 255 - self.image[self.image >= threshold]

    def darken_image(self, darkening_factor=0.2):
        # Darken the image by multiplying all pixel values by the darkening factor
        if len(self.image.shape) == 2:
            self.image = (self.image * darkening_factor).astype(np.uint8)
        else:
            # convert from BGR to Grayscale
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = (self.image * darkening_factor).astype(np.uint8)

    def brighten_image(self, brightening_factor=50):
        # Brighten the image by adding all pixel values by the brightening factor
        if len(self.image.shape) == 2:
            self.image = cv2.add(self.image, brightening_factor).astype(np.uint8)

        else:
            # convert from BGR to Grayscale
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.add(self.image, brightening_factor).astype(np.uint8)

 

    def adjust_brightness(image, brightness_value):
    modified_image = cv2.add(image, brightness_value)
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    return modified_image

    def adjust_color(image, color, adjustment_value):
    modified_image = image.copy()

      if color == 'Red':
        modified_image[:, :, 2] = cv2.add(modified_image[:, :, 2], adjustment_value)  # Increase intensity of red channel
     elif color == 'Green':
        modified_image[:, :, 1] = cv2.add(modified_image[:, :, 1], adjustment_value)  # Increase intensity of green channel
    elif color == 'Blue':
        modified_image[:, :, 0] = cv2.add(modified_image[:, :, 0], adjustment_value)  # Increase intensity of blue channel
    elif color == 'Yellow':
        modified_image[:, :, 2] = cv2.add(modified_image[:, :, 2], adjustment_value)  # Increase intensity of red channel
        modified_image[:, :, 1] = cv2.add(modified_image[:, :, 1], adjustment_value)  # Increase intensity of green channel
    elif color == 'Purple':
        modified_image[:, :, 2] = cv2.add(modified_image[:, :, 2], adjustment_value)  # Increase intensity of red channel
        modified_image[:, :, 0] = cv2.add(modified_image[:, :, 0], adjustment_value)  # Increase intensity of blue channel
    elif color == 'Cyan':
        modified_image[:, :, 1] = cv2.add(modified_image[:, :, 1], adjustment_value)  # Increase intensity of green channel
        modified_image[:, :, 0] = cv2.add(modified_image[:, :, 0], adjustment_value)  # Increase intensity of blue channel

    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    return modified_image

 
  brightened_image = adjust_brightness(img_original, 50)

 
 colored_image_blue = adjust_color(img_original, 'Blue', 50)
 colored_image_green = adjust_color(img_original, 'Green', 50)
 colored_image_cyan = adjust_color(img_original, 'Cyan', 50)
 colored_image_purple = adjust_color(img_original, 'Purple', 50)
 colored_image_yellow = adjust_color(img_original, 'Yellow', 50)

 cv2.imshow('Original Image', img_original)
 cv2.imshow('Brightened Image', brightened_image)
 cv2.imshow('Colored Image (Blue)', colored_image_blue)
 cv2.imshow('Colored Image (Green)', colored_image_green)
 cv2.imshow('Colored Image (Cyan)', colored_image_cyan)
 cv2.imshow('Colored Image (Purple)', colored_image_purple)
 cv2.imshow('Colored Image (Yellow)', colored_image_yellow)
 cv2.waitKey(0)
 cv2.destroyAllWindows()



        
        
