import numpy as np
import pandas as pd
import cv2
import joblib

def predict_leaf_pixels(image_path, model_path="models/best_model.pkl", mask=None, border_thickness=0):
    """
    Classify sick pixels in a leaf image using a trained model.
    Only pixels inside the mask (and outside the border) are classified.

    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained model (.pkl)
        mask (np.ndarray): Optional binary mask (1=leaf, 0=background)
        border_thickness (int): Thickness of the border (in pixels) to ignore

    Returns:
        np.ndarray: Result image with sick pixels marked in red
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image from path:", image_path)

    model = joblib.load(model_path)
    height, width, _ = image.shape

    if mask is not None and border_thickness > 0:
        kernel = np.ones((border_thickness, border_thickness), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    features = []
    positions = []

    for y in range(height):
        for x in range(width):
            if mask is not None and mask[y, x] == 0:
                continue

            b, g, r = image[y, x]
            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv

            features.append([b, g, r, h, s, v])
            positions.append((y, x))

    df_features = pd.DataFrame(features, columns=["B", "G", "R", "H", "S", "V"])
    predictions = model.predict(df_features)

    # Calculate % of sick pixels inside the mask
    total_pixels = len(predictions)
    sick_pixels = sum(1 for p in predictions if p == "s")
    sick_percentage = (sick_pixels / total_pixels) * 100

    # Copy original image to create overlay
    overlay = image.copy()

    for (y, x), pred in zip(positions, predictions):
        if pred == "s":
            overlay[y, x] = (0, 0, 255)  # red on top of original

    return overlay, sick_percentage

