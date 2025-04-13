import cv2
import numpy as np

def segment_leaf(image_path):
    """
    Segments a leaf from the background.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Binary mask of the segmented leaf.
    """
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gamma correction
    gamma = 1.5
    look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, look_up)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gamma_corrected)

    # Canny edge detection
    edges = cv2.Canny(contrast, 50, 150)

    # Morphological closing
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Median blur to reduce noise
    blurred = cv2.medianBlur(closed, 5)

    # Find contours
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    return mask
