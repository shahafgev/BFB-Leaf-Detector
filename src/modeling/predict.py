import numpy as np
import pandas as pd
import cv2
import joblib


def predict_leaf_pixels(image_path, model_path="models/best_model.pkl", mask=None, border_thickness=0,
                        processed_img=None):
    """
    Classify sick pixels in a leaf image using a trained model.
    Only pixels inside the mask (and outside the border) are classified.

    Returns:
        overlay (np.ndarray): Original image with sick pixels marked in red
        sick_percentage (float): % of sick pixels inside the leaf
    """
    image = processed_img if processed_img is not None else cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    display_img = cv2.imread(image_path)  # Used only for displaying final result

    if mask is None:
        raise ValueError("Mask must be provided to restrict classification to leaf region.")

    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("Mask and image dimensions do not match.")

    # Store original mask for percentage calculation
    original_mask = mask.copy()

    # Create classification mask with border thickness
    classification_mask = mask.copy()
    if border_thickness > 0:
        if border_thickness * 2 < min(classification_mask.shape):
            kernel = np.ones((border_thickness, border_thickness), np.uint8)
            classification_mask = cv2.erode(classification_mask, kernel, iterations=1)

    model = joblib.load(model_path)

    features = []
    positions = []

    height, width = classification_mask.shape
    for y in range(height):
        for x in range(width):
            if classification_mask[y, x] == 0:
                continue

            b, g, r = image[y, x]
            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv
            features.append([b, g, r, h, s, v])
            positions.append((y, x))

    df_features = pd.DataFrame(features, columns=["B", "G", "R", "H", "S", "V"])
    predictions = model.predict(df_features)

    # Calculate percentage based on all leaf pixels
    total_leaf_pixels = np.sum(original_mask == 1)
    sick_pixels = sum(1 for p in predictions if p == "s")
    sick_percentage = (sick_pixels / total_leaf_pixels) * 100 if total_leaf_pixels > 0 else 0

    # Draw predictions only on a clean copy of the original image
    overlay = display_img.copy()

    for (y, x), pred in zip(positions, predictions):
        if pred == "s":
            overlay[y, x] = (0, 0, 255)  # Red

    overlay[original_mask == 0] = display_img[original_mask == 0]  # Ensure red is only on leaf

    return overlay, sick_percentage
