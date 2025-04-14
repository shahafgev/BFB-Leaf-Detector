from src.modeling.predict import predict_leaf_pixels
import cv2
import numpy as np

# Paths
image_path = "data/1.jpg"
mask_path = "leaf_mask.npy"

# Load mask (0s and 1s)
mask = np.load(mask_path)

# Predict only within the mask
result, percent = predict_leaf_pixels(image_path, mask=mask, border_thickness=5)

# Show result
cv2.imshow("Sick Areas Overlay", result)
print(f"Sick Area: {percent:.2f}%")
cv2.waitKey(0)
cv2.destroyAllWindows()
