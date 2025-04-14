from src.leaf_classifier import classify_leaf
import cv2

print("🟡 Starting classification...")

image_path = "data/1.jpg"
try:
    overlay, sick_percent = classify_leaf(image_path, border_thickness=5)
    print(f"✅ Sick Area: {sick_percent:.2f}%")
    cv2.imshow("Classified Leaf (Overlay)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print("❌ Error:", e)
