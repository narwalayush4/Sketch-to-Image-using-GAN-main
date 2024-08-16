import cv2
import numpy as np

def apply_xdog_filter(image):
    """
    Applies the XDoG filter to an image to create a sketch-like effect.

    Args:
        image: Input image as a numpy array.

    Returns:
        Sketch-like image as a numpy array.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur with two different sigmas
    blurred1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
    blurred2 = cv2.GaussianBlur(gray, (5, 5), 2.0)

    # Compute the difference of the blurred images
    dog = blurred1 - blurred2

    # Thresholding to create a binary sketch
    _, sketch = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return sketch

def apply_gaussian_deblurring(image, kernel_size=5):
    """
    Applies Gaussian deblurring to smooth out the edges in the sketch.

    Args:
        image: Input image as a numpy array.
        kernel_size: Size of the Gaussian kernel.

    Returns:
        Deblurred image as a numpy array.
    """
    # Apply Gaussian blur (deblurring)
    deblurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return deblurred

def generate_synthetic_sketch(image_path, output_path):
    """
    Generates a synthetic sketch from an image and saves it.

    Args:
        image_path: Path to the input image.
        output_path: Path to save the generated sketch.
    """
    # Load image
    image = cv2.imread(image_path)

    # Apply XDoG filter
    sketch = apply_xdog_filter(image)

    # Apply Gaussian deblurring
    deblurred_sketch = apply_gaussian_deblurring(sketch)

    # Save the synthetic sketch
    cv2.imwrite(output_path, deblurred_sketch)
