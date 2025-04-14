import numpy as np
from src.modeling.predict import predict_leaf_pixels


def classify_leaf(image_path, mask=None, model_path="models/best_model.pkl", border_thickness=0, processed_img=None):
    if mask is None:
        mask = np.load("leaf_mask.npy")

    overlay, sick_percent = predict_leaf_pixels(
        image_path=image_path,
        mask=mask,
        model_path=model_path,
        border_thickness=border_thickness,
        processed_img=processed_img
    )
    return overlay, sick_percent
