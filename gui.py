import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from src.leaf_classifier import classify_leaf
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Set QT plugin path dynamically
import PyQt5
plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins", "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path


class LeafDiseaseGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leaf Disease Classifier")
        self.resize(1000, 600)

        self.image_path = None
        self.border_thickness = 0

        self.init_ui()
        self.load_sam()

    def init_ui(self):
        # Labels
        self.original_label = QLabel("Original Image")
        self.result_label = QLabel("Result Overlay")
        self.result_label.setFrameStyle(QFrame.Box)

        # Buttons
        upload_btn = QPushButton("Upload Leaf Image")
        upload_btn.clicked.connect(self.load_image)

        classify_btn = QPushButton("Run Classification")
        classify_btn.clicked.connect(self.run_classification)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(20)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_slider_label)

        self.slider_label = QLabel("Border Thickness: 0 px")

        # Layouts
        h_img_layout = QHBoxLayout()
        h_img_layout.addWidget(self.original_label)
        h_img_layout.addWidget(self.result_label)

        h_slider_layout = QHBoxLayout()
        h_slider_layout.addWidget(self.slider_label)
        h_slider_layout.addWidget(self.slider)

        v_main_layout = QVBoxLayout()
        v_main_layout.addLayout(h_img_layout)
        v_main_layout.addWidget(upload_btn)
        v_main_layout.addLayout(h_slider_layout)
        v_main_layout.addWidget(classify_btn)

        self.setLayout(v_main_layout)

    def update_slider_label(self):
        value = self.slider.value()
        self.border_thickness = value
        self.slider_label.setText(f"Border Thickness: {value} px")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path).scaled(400, 400, Qt.KeepAspectRatio)
            self.original_label.setPixmap(pixmap)

    def load_sam(self):
        # Load SAM model once when GUI starts
        checkpoint = "checkpoints/sam_vit_b.pth"
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        sam.to(device="cpu")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.92,
            min_mask_region_area=1000
        )

    def generate_mask(self, image_path):
        image_orig = cv2.imread(image_path)
        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image_orig, (512, 512))
        masks = self.mask_generator.generate(image)
        masks = sorted(masks, key=lambda m: m['area'], reverse=True)
        best_mask = masks[0]['segmentation'].astype(np.uint8)
        mask_resized = cv2.resize(best_mask, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_resized

    def run_classification(self):
        if not self.image_path:
            return

        try:
            mask = self.generate_mask(self.image_path)
            overlay_img, percent = classify_leaf(self.image_path, mask=mask, border_thickness=self.border_thickness)

            # Convert to QPixmap for display
            rgb_image = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img).scaled(400, 400, Qt.KeepAspectRatio)

            self.result_label.setPixmap(pixmap)
            self.result_label.setToolTip(f"Sick Area: {percent:.2f}%")

        except Exception as e:
            print("Error during classification:", e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = LeafDiseaseGUI()
    gui.show()
    sys.exit(app.exec_())