import pandas as pd
import cv2
import numpy as np
import os

# Load the RGB dataset
input_path = "processed_data/pixel_dataset_rgb.csv"
output_path = "processed_data/pixel_dataset_rgbhsv.csv"

df = pd.read_csv(input_path)

# Convert each pixel's BGR to HSV
hsv_values = []

for _, row in df.iterrows():
    bgr_pixel = np.uint8([[[row["B"], row["G"], row["R"]]]])
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
    hsv_values.append(hsv_pixel)

# Add HSV columns to DataFrame
df[["H", "S", "V"]] = pd.DataFrame(hsv_values, index=df.index)

# Save to new file
df.to_csv(output_path, index=False)
print(f"Saved extended dataset with HSV to: {output_path}")
