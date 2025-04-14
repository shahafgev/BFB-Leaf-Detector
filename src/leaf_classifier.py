import numpy as np
from src.modeling.predict import predict_leaf_pixels


def classify_leaf(image_path, mask=None, model_path="models/best_model.pkl", border_thickness=0):
    """
    Full pipeline: classify sick areas in a leaf image using SAM mask and trained model.

    Returns:
        overlay (np.ndarray): Original image with sick areas in red
        sick_percent (float): % of sick pixels inside the leaf
    """
    if mask is None:
        mask = np.load("leaf_mask.npy")

    overlay, sick_percent = predict_leaf_pixels(
        image_path=image_path,
        mask=mask,
        model_path=model_path,
        border_thickness=border_thickness
    )
    return overlay, sick_percent
