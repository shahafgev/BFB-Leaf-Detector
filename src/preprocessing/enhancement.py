import cv2
import os


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE enhancement to a BGR image.

    Args:
        image (np.ndarray): BGR image
        clip_limit (float): CLAHE clip limit
        grid_size (tuple): CLAHE grid size

    Returns:
        np.ndarray: CLAHE-enhanced BGR image
    """
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)

    clahe_img = cv2.merge([clahe_b, clahe_g, clahe_r])
    return clahe_img


def batch_apply_clahe(input_dir, output_dir, clip_limit=1.0, grid_size=(1, 1)):
    """
    Applies CLAHE to all .jpg images in a directory and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            enhanced = apply_clahe(image, clip_limit, grid_size)

            output_filename = f"clahe_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, enhanced)

    print(f"Processed images saved to: {output_dir}")

