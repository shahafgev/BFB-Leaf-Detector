import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load image
image_path = "data/1.jpg"  # Change this to any image you want
image_orig = cv2.imread(image_path)
image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

# Resize for faster SAM
target_size = (512, 512)
image = cv2.resize(image_orig, target_size)

# Load SAM model
checkpoint = "checkpoints/sam_vit_b.pth"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
sam.to(device="cpu")

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,       # ↓ from default 32 (faster)
    pred_iou_thresh=0.80,     # only high-confidence masks
    stability_score_thresh=0.92,
    min_mask_region_area=1000  # skip tiny segments
)
masks = mask_generator.generate(image)

# Sort masks by area (we want the largest one = leaf)
masks = sorted(masks, key=lambda m: m['area'], reverse=True)
best_mask = masks[0]["segmentation"].astype(np.uint8)  # Convert to 0/1 integers

# Resize mask back to original image size
mask_resized = cv2.resize(best_mask, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_NEAREST)

# Show only the largest mask
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.contour(best_mask, colors='red', linewidths=1)
plt.axis('off')
plt.title("Main Leaf Mask")
plt.show()

# Save the largest mask as binary NumPy array (for classification)
np.save("leaf_mask.npy", mask_resized.astype(np.uint8))  # Save as 0/1
