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
    
    """
    /////////////////////////////////////////////////////////
                    Amal's functions here 👉👈
    /////////////////////////////////////////////////////////
    """

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
