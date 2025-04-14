import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QFrame,
    QMessageBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from src.leaf_classifier import classify_leaf
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set QT plugin path dynamically
import PyQt5
plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins", "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

def apply_clahe_only_on_leaf(image, mask, clip_limit=1.0, grid_size=(1, 1)):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Apply CLAHE to the full channel
    b_enh_full = clahe.apply(b)
    g_enh_full = clahe.apply(g)
    r_enh_full = clahe.apply(r)

    # Create empty channels and copy only the leaf pixels
    b_enh = np.zeros_like(b)
    g_enh = np.zeros_like(g)
    r_enh = np.zeros_like(r)

    b_enh[mask == 1] = b_enh_full[mask == 1]
    g_enh[mask == 1] = g_enh_full[mask == 1]
    r_enh[mask == 1] = r_enh_full[mask == 1]

    return cv2.merge([b_enh, g_enh, r_enh])

class LeafDiseaseGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leaf Disease Analyzer")
        self.resize(1200, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QLabel {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #ffffff;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)

        self.image_path = None
        self.border_thickness = 0
        self.mask_generator = None
        
        # Store intermediate results
        self.current_mask = None
        self.current_clahe_img = None
        self.current_original_img = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        self.init_ui(main_layout)
        self.setLayout(main_layout)
        self.load_sam()

    def init_ui(self, main_layout):
        # Create title label
        title_label = QLabel("Leaf Disease Analysis System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px;
        """)
        main_layout.addWidget(title_label)

        # Create percentage display
        percentage_container = QWidget()
        percentage_layout = QHBoxLayout(percentage_container)
        percentage_layout.setContentsMargins(0, 0, 0, 0)
        
        self.percentage_label = QLabel("Disease Percentage: --")
        self.percentage_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            border: 2px solid #4CAF50;
        """)
        self.percentage_label.setAlignment(Qt.AlignCenter)
        percentage_layout.addWidget(self.percentage_label)
        main_layout.addWidget(percentage_container)

        # Create labels for each stage
        self.original_label = QLabel("Original Image")
        self.original_label.setFrameStyle(QFrame.Box)
        self.original_label.setMinimumSize(250, 250)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("font-weight: bold;")

        self.segmentation_label = QLabel("SAM Segmentation")
        self.segmentation_label.setFrameStyle(QFrame.Box)
        self.segmentation_label.setMinimumSize(250, 250)
        self.segmentation_label.setAlignment(Qt.AlignCenter)
        self.segmentation_label.setStyleSheet("font-weight: bold;")

        self.clahe_label = QLabel("CLAHE Enhanced")
        self.clahe_label.setFrameStyle(QFrame.Box)
        self.clahe_label.setMinimumSize(250, 250)
        self.clahe_label.setAlignment(Qt.AlignCenter)
        self.clahe_label.setStyleSheet("font-weight: bold;")

        self.result_label = QLabel("Final Result")
        self.result_label.setFrameStyle(QFrame.Box)
        self.result_label.setMinimumSize(250, 250)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold;")

        # Create buttons
        upload_btn = QPushButton("Upload Leaf Image")
        upload_btn.clicked.connect(self.load_image)
        upload_btn.setIcon(self.style().standardIcon(self.style().SP_FileIcon))

        classify_btn = QPushButton("Run Analysis")
        classify_btn.clicked.connect(self.run_classification)
        classify_btn.setIcon(self.style().standardIcon(self.style().SP_CommandLink))

        # Create slider
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        
        self.slider_label = QLabel("Border Thickness: 0 px")
        self.slider_label.setStyleSheet("font-weight: bold;")
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(20)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_slider_label)
        
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)

        # Create layouts
        images_layout = QGridLayout()
        images_layout.setSpacing(10)
        images_layout.addWidget(self.original_label, 0, 0)
        images_layout.addWidget(self.segmentation_label, 0, 1)
        images_layout.addWidget(self.clahe_label, 0, 2)
        images_layout.addWidget(self.result_label, 0, 3)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(upload_btn)
        button_layout.addWidget(classify_btn)
        button_layout.addStretch()

        # Add all layouts to main layout
        main_layout.addLayout(images_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(slider_container)

    def update_slider_label(self):
        value = self.slider.value()
        self.border_thickness = value
        self.slider_label.setText(f"Border Thickness: {value} px")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path).scaled(250, 250, Qt.KeepAspectRatio)
            self.original_label.setPixmap(pixmap)
            # Clear other images and stored results
            self.segmentation_label.clear()
            self.clahe_label.clear()
            self.result_label.clear()
            self.percentage_label.setText("Disease Percentage: --")
            self.current_mask = None
            self.current_clahe_img = None
            self.current_original_img = None

    def load_sam(self):
        try:
            checkpoint = "checkpoints/sam_vit_b.pth"
            if not os.path.exists(checkpoint):
                QMessageBox.critical(self, "Error", 
                    "SAM model checkpoint not found. Please ensure 'sam_vit_b.pth' exists in the checkpoints directory.")
                return

            sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            sam.to(device="cpu")
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=8,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.92,
                min_mask_region_area=1000
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load SAM model: {str(e)}")

    def generate_mask(self, image_path):
        if self.mask_generator is None:
            raise ValueError("SAM model not loaded. Please restart the application.")

        image_orig = cv2.imread(image_path)
        if image_orig is None:
            raise ValueError("Failed to load image. Please check if the file exists and is not corrupted.")

        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image_orig, (512, 512))
        masks = self.mask_generator.generate(image)
        
        if not masks:
            raise ValueError("No leaf mask detected. Please try a different image.")

        masks = sorted(masks, key=lambda m: m['area'], reverse=True)
        best_mask = masks[0]['segmentation'].astype(np.uint8)
        mask_resized = cv2.resize(best_mask, (image_orig.shape[1], image_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
        inverted_mask = np.where(mask_resized == 1, 0, 1).astype(np.uint8)
        return inverted_mask

    def run_classification(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return

        try:
            # Step 1: Load and validate original image (only if not already loaded)
            if self.current_original_img is None:
                self.current_original_img = cv2.imread(self.image_path)
                if self.current_original_img is None:
                    raise ValueError("Failed to load image. Please check if the file exists and is not corrupted.")

            # Step 2: Generate and display SAM mask (only if not already generated)
            if self.current_mask is None:
                try:
                    original_rgb = cv2.cvtColor(self.current_original_img, cv2.COLOR_BGR2RGB)
                    self.current_mask = self.generate_mask(self.image_path)

                    # Display segmentation mask
                    mask_display = (self.current_mask * 255).astype(np.uint8)
                    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2RGB)
                    h, w, ch = mask_display.shape
                    bytes_per_line = ch * w
                    qt_mask = QImage(mask_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    mask_pixmap = QPixmap.fromImage(qt_mask).scaled(250, 250, Qt.KeepAspectRatio)
                    self.segmentation_label.setPixmap(mask_pixmap)
                    QApplication.processEvents()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error in segmentation: {str(e)}")
                    return

            # Step 3: Apply and display CLAHE (only if not already processed)
            if self.current_clahe_img is None:
                try:
                    self.current_clahe_img = apply_clahe_only_on_leaf(self.current_original_img, self.current_mask)
                    cv2.imwrite("debug_clahe_input.jpg", self.current_clahe_img)

                    # Display CLAHE result
                    clahe_rgb = cv2.cvtColor(self.current_clahe_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = clahe_rgb.shape
                    bytes_per_line = ch * w
                    qt_clahe = QImage(clahe_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    clahe_pixmap = QPixmap.fromImage(qt_clahe).scaled(250, 250, Qt.KeepAspectRatio)
                    self.clahe_label.setPixmap(clahe_pixmap)
                    QApplication.processEvents()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error in CLAHE processing: {str(e)}")
                    return

            # Step 4: Run classification and display final result (always run this step)
            try:
                overlay_img, percent = classify_leaf(
                    self.image_path,
                    mask=self.current_mask,
                    border_thickness=self.border_thickness,
                    processed_img=self.current_clahe_img
                )

                # Display final result
                rgb_image = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                result_pixmap = QPixmap.fromImage(qt_img).scaled(250, 250, Qt.KeepAspectRatio)
                self.result_label.setPixmap(result_pixmap)
                
                # Update percentage display
                self.percentage_label.setText(f"Disease Percentage: {percent:.2f}%")
                QApplication.processEvents()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error in classification: {str(e)}")
                return

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during processing: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = LeafDiseaseGUI()
    gui.show()
    sys.exit(app.exec_())
