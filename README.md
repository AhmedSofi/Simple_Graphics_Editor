# Image Processing Application

This is a simple image processing application built with Python, using OpenCV for image manipulation and Tkinter for the graphical user interface. The application allows users to load an image and apply various transformations and filters, such as resizing, converting to grayscale, adjusting brightness, plotting histograms, and more.

## Features

- Load and display an image
- Convert image to grayscale
- Eliminate specific colors (red, green, blue, yellow, cyan, purple)
- Resize the image
- Swap color channels
- Complement the image (invert colors)
- Solarize the image with a threshold
- Darken or brighten the image
- Adjust brightness
- Adjust specific color channels
- Plot histograms for different channels
- Increase contrast using histogram equalization
- Apply median, min, and max filters
- Sharpen the image with a range filter
- Apply morphological operations: dilation, erosion, opening, closing
- Apply global and adaptive thresholding

## Installation

To set up the application on your system, follow these steps:

1. Ensure you have **Python 3.x** installed.

2. Install the required libraries using pip:

   ```
   pip install opencv-python numpy matplotlib pillow
   ```

3. Clone or download this repository to your local machine.

## Usage

To run and use the application, follow these steps:

1. Launch the application by running the `GUI.py` script:

   ```
   python GUI.py
   ```

2. Click **"Import Image"** to load an image from your file system.

3. Select an operation from the dropdown menu and click **"apply"** to perform the operation.

4. For some operations, a new window will appear where you can adjust parameters (e.g., kernel size for filters, threshold values).

5. To revert to the original image, click **"Original Image"**.

## Dependencies

The application requires the following libraries:

- **Python 3.x**
- **OpenCV** (`opencv-python`) - For image processing
- **NumPy** - For numerical operations
- **Matplotlib** - For plotting histograms
- **Pillow** - For image handling in Tkinter

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests for enhancements or bug fixes.


## Notes

- The application supports image formats compatible with OpenCV (e.g., JPG, PNG).
- Some operations may not work as expected if the image format is incompatible (e.g., applying color elimination on a grayscale image).
- The Tkinter GUI is cross-platform, though its appearance may vary slightly across operating systems.
- Ensure an image is loaded before applying any operations.
- For operations requiring input (e.g., resize dimensions, threshold values), provide valid values as prompted.
