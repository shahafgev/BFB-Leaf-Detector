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
import datetime

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
        self.percentage_label = QLabel("Disease Percentage: --")
        self.percentage_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            background: transparent;
            border: none;
            padding: 0;
        """)
        self.percentage_label.setAlignment(Qt.AlignCenter)

        # Create labels for each stage with info icons
        # Original Image
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.setContentsMargins(0, 0, 0, 0)
        original_layout.setSpacing(5)
        
        # Header with info icon
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        
        header_label = QLabel("Original Image")
        header_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            color: #2c3e50;
            background: transparent;
            border: none;
            padding: 0;
        """)
        header_label.setAlignment(Qt.AlignCenter)
        
        original_info = QLabel("ⓘ")
        original_info.setStyleSheet("""
            font-size: 12px;
            color: #3498db;
            font-weight: bold;
            background: transparent;
            border: none;
            padding: 0;
        """)
        original_info.setToolTip("The original leaf image uploaded by the user")
        
        header_layout.addStretch()
        header_layout.addWidget(header_label)
        header_layout.addWidget(original_info)
        header_layout.addStretch()
        
        self.original_label = QLabel()
        self.original_label.setFrameStyle(QFrame.Box)
        self.original_label.setMinimumSize(250, 350)  # Increased height
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("""
            background-color: #f8f9fa;
            color: #6c757d;
        """)
        self.original_label.setText("Upload an image to begin")
        
        original_layout.addWidget(header_container)
        original_layout.addWidget(self.original_label)
        
        # SAM Segmentation
        segmentation_container = QWidget()
        segmentation_layout = QVBoxLayout(segmentation_container)
        segmentation_layout.setContentsMargins(0, 0, 0, 0)
        segmentation_layout.setSpacing(5)
        
        # Header with info icon
        seg_header_container = QWidget()
        seg_header_layout = QHBoxLayout(seg_header_container)
        seg_header_layout.setContentsMargins(0, 0, 0, 0)
        seg_header_layout.setSpacing(5)
        
        seg_header_label = QLabel("SAM Segmentation")
        seg_header_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            color: #2c3e50;
            background: transparent;
            border: none;
            padding: 0;
        """)
        seg_header_label.setAlignment(Qt.AlignCenter)
        
        segmentation_info = QLabel("ⓘ")
        segmentation_info.setStyleSheet("""
            font-size: 12px;
            color: #3498db;
            font-weight: bold;
            background: transparent;
            border: none;
            padding: 0;
        """)
        segmentation_info.setToolTip("Segmentation mask showing the leaf area. Red border indicates the excluded area based on border thickness.")
        
        seg_header_layout.addStretch()
        seg_header_layout.addWidget(seg_header_label)
        seg_header_layout.addWidget(segmentation_info)
        seg_header_layout.addStretch()
        
        self.segmentation_label = QLabel()
        self.segmentation_label.setFrameStyle(QFrame.Box)
        self.segmentation_label.setMinimumSize(250, 350)  # Increased height
        self.segmentation_label.setAlignment(Qt.AlignCenter)
        self.segmentation_label.setStyleSheet("""
            background-color: #f8f9fa;
            color: #6c757d;
        """)
        self.segmentation_label.setText("Segmentation will appear here")
        
        segmentation_layout.addWidget(seg_header_container)
        segmentation_layout.addWidget(self.segmentation_label)
        
        # CLAHE Enhanced
        clahe_container = QWidget()
        clahe_layout = QVBoxLayout(clahe_container)
        clahe_layout.setContentsMargins(0, 0, 0, 0)
        clahe_layout.setSpacing(5)
        
        # Header with info icon
        clahe_header_container = QWidget()
        clahe_header_layout = QHBoxLayout(clahe_header_container)
        clahe_header_layout.setContentsMargins(0, 0, 0, 0)
        clahe_header_layout.setSpacing(5)
        
        clahe_header_label = QLabel("CLAHE Enhanced")
        clahe_header_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            color: #2c3e50;
            background: transparent;
            border: none;
            padding: 0;
        """)
        clahe_header_label.setAlignment(Qt.AlignCenter)
        
        clahe_info = QLabel("ⓘ")
        clahe_info.setStyleSheet("""
            font-size: 12px;
            color: #3498db;
            font-weight: bold;
            background: transparent;
            border: none;
            padding: 0;
        """)
        clahe_info.setToolTip("Contrast Limited Adaptive Histogram Equalization applied to enhance the leaf image")
        
        clahe_header_layout.addStretch()
        clahe_header_layout.addWidget(clahe_header_label)
        clahe_header_layout.addWidget(clahe_info)
        clahe_header_layout.addStretch()
        
        self.clahe_label = QLabel()
        self.clahe_label.setFrameStyle(QFrame.Box)
        self.clahe_label.setMinimumSize(250, 350)  # Increased height
        self.clahe_label.setAlignment(Qt.AlignCenter)
        self.clahe_label.setStyleSheet("""
            background-color: #f8f9fa;
            color: #6c757d;
        """)
        self.clahe_label.setText("Enhanced image will appear here")
        
        clahe_layout.addWidget(clahe_header_container)
        clahe_layout.addWidget(self.clahe_label)
        
        # Final Result
        result_container = QWidget()
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(5)
        
        # Header with info icon
        result_header_container = QWidget()
        result_header_layout = QHBoxLayout(result_header_container)
        result_header_layout.setContentsMargins(0, 0, 0, 0)
        result_header_layout.setSpacing(5)
        
        result_header_label = QLabel("Final Result")
        result_header_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            color: #2c3e50;
            background: transparent;
            border: none;
            padding: 0;
        """)
        result_header_label.setAlignment(Qt.AlignCenter)
        
        result_info = QLabel("ⓘ")
        result_info.setStyleSheet("""
            font-size: 12px;
            color: #3498db;
            font-weight: bold;
            background: transparent;
            border: none;
            padding: 0;
        """)
        result_info.setToolTip("Final result with diseased areas highlighted in red")
        
        result_header_layout.addStretch()
        result_header_layout.addWidget(result_header_label)
        result_header_layout.addWidget(result_info)
        result_header_layout.addStretch()
        
        self.result_label = QLabel()
        self.result_label.setFrameStyle(QFrame.Box)
        self.result_label.setMinimumSize(250, 350)  # Increased height
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            background-color: #f8f9fa;
            color: #6c757d;
        """)
        self.result_label.setText("Final result will appear here")
        
        result_layout.addWidget(result_header_container)
        result_layout.addWidget(self.result_label)

        # Create buttons
        upload_btn = QPushButton("Upload Leaf Image")
        upload_btn.clicked.connect(self.load_image)
        upload_btn.setIcon(self.style().standardIcon(self.style().SP_FileIcon))

        classify_btn = QPushButton("Run Analysis")
        classify_btn.clicked.connect(self.run_classification)
        classify_btn.setIcon(self.style().standardIcon(self.style().SP_CommandLink))
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_analysis)
        reset_btn.setIcon(self.style().standardIcon(self.style().SP_BrowserReload))
        
        save_btn = QPushButton("Save Results")
        save_btn.clicked.connect(self.save_results)
        save_btn.setIcon(self.style().standardIcon(self.style().SP_DialogSaveButton))

        # Create slider
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(5)
        
        # Header with info icon
        slider_header_container = QWidget()
        slider_header_layout = QHBoxLayout(slider_header_container)
        slider_header_layout.setContentsMargins(0, 0, 0, 0)
        slider_header_layout.setSpacing(5)
        
        self.slider_label = QLabel("Border Thickness: 0 px")
        self.slider_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            background: transparent;
            border: none;
            padding: 0;
        """)
        
        slider_info = QLabel("ⓘ")
        slider_info.setStyleSheet("""
            font-size: 12px;
            color: #3498db;
            font-weight: bold;
            background: transparent;
            border: none;
            padding: 0;
        """)
        slider_info.setToolTip("Adjust the border thickness to exclude pixels from the edge of the leaf. This helps avoid misclassification of leaf edges.")
        
        slider_header_layout.addStretch()
        slider_header_layout.addWidget(self.slider_label)
        slider_header_layout.addWidget(slider_info)
        slider_header_layout.addStretch()
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(20)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_slider_label)
        self.slider.setFixedWidth(600)  # Make the slider wider
        
        # Create percentage display with more prominent styling
        percentage_container = QWidget()
        percentage_layout = QHBoxLayout(percentage_container)
        percentage_layout.setContentsMargins(0, 0, 0, 0)
        
        self.percentage_label = QLabel("Disease Percentage: --")
        self.percentage_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            background-color: #e8f5e9;
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid #4CAF50;
        """)
        self.percentage_label.setAlignment(Qt.AlignCenter)
        percentage_layout.addWidget(self.percentage_label)
        
        slider_layout.addStretch()
        slider_layout.addWidget(slider_header_container)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(percentage_container)
        slider_layout.addStretch()

        # Create layouts
        images_layout = QGridLayout()
        images_layout.setSpacing(10)
        images_layout.addWidget(original_container, 0, 0)
        images_layout.addWidget(segmentation_container, 0, 1)
        images_layout.addWidget(clahe_container, 0, 2)
        images_layout.addWidget(result_container, 0, 3)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(upload_btn)
        button_layout.addWidget(classify_btn)
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(save_btn)
        button_layout.addStretch()

        # Add all layouts to main layout
        main_layout.addLayout(images_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(slider_container)

    def update_slider_label(self):
        value = self.slider.value()
        self.border_thickness = value
        self.slider_label.setText(f"Border Thickness: {value} px")
        
        # If we have a mask and the user is changing the border thickness,
        # update the visualization immediately
        if self.current_mask is not None:
            if value > 0:
                # Create a copy of the mask for visualization
                border_vis_mask = self.current_mask.copy()
                
                # Create the eroded mask
                kernel = np.ones((value, value), np.uint8)
                eroded_mask = cv2.erode(border_vis_mask, kernel, iterations=1)
                
                # Create a visualization where the border area is highlighted
                border_area = border_vis_mask - eroded_mask
                
                # Convert to RGB for visualization
                vis_mask = np.zeros((border_vis_mask.shape[0], border_vis_mask.shape[1], 3), dtype=np.uint8)
                vis_mask[border_vis_mask == 1] = [255, 255, 255]  # White for leaf area
                vis_mask[border_area == 1] = [255, 0, 0]  # Red for border area
                
                # Display the visualization
                h, w, ch = vis_mask.shape
                bytes_per_line = ch * w
                qt_vis = QImage(vis_mask.data, w, h, bytes_per_line, QImage.Format_RGB888)
                vis_pixmap = QPixmap.fromImage(qt_vis).scaled(250, 250, Qt.KeepAspectRatio)
                self.segmentation_label.setPixmap(vis_pixmap)
            else:
                # If border thickness is 0, show the original mask without any red border
                mask_display = (self.current_mask * 255).astype(np.uint8)
                mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2RGB)
                h, w, ch = mask_display.shape
                bytes_per_line = ch * w
                qt_mask = QImage(mask_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
                mask_pixmap = QPixmap.fromImage(qt_mask).scaled(250, 250, Qt.KeepAspectRatio)
                self.segmentation_label.setPixmap(mask_pixmap)
        
        QApplication.processEvents()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path).scaled(250, 350, Qt.KeepAspectRatio)  # Updated height
            self.original_label.setPixmap(pixmap)
            # Only clear the original image text, keep other text labels
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
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please upload an image first.")
            msg.setWindowTitle("Warning")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #f0f0f0;
                }
                QLabel {
                    background: transparent;
                    border: none;
                    padding: 0;
                    color: #2c3e50;
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
            """)
            msg.exec_()
            return

        try:
            # Step 1: Load and validate original image (only if not already loaded)
            if self.current_original_img is None:
                self.current_original_img = cv2.imread(self.image_path)
                if self.current_original_img is None:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setText("Failed to load image. Please check if the file exists and is not corrupted.")
                    msg.setWindowTitle("Error")
                    msg.setStyleSheet("""
                        QMessageBox {
                            background-color: #f0f0f0;
                        }
                        QLabel {
                            background: transparent;
                            border: none;
                            padding: 0;
                            color: #2c3e50;
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
                    """)
                    msg.exec_()
                    return

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
                    mask_pixmap = QPixmap.fromImage(qt_mask).scaled(250, 350, Qt.KeepAspectRatio)  # Updated height
                    self.segmentation_label.setPixmap(mask_pixmap)
                    QApplication.processEvents()
                except Exception as e:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setText(f"Error in segmentation: {str(e)}")
                    msg.setWindowTitle("Error")
                    msg.setStyleSheet("""
                        QMessageBox {
                            background-color: #f0f0f0;
                        }
                        QLabel {
                            background: transparent;
                            border: none;
                            padding: 0;
                            color: #2c3e50;
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
                    """)
                    msg.exec_()
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
                    clahe_pixmap = QPixmap.fromImage(qt_clahe).scaled(250, 350, Qt.KeepAspectRatio)  # Updated height
                    self.clahe_label.setPixmap(clahe_pixmap)
                    QApplication.processEvents()
                except Exception as e:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setText(f"Error in CLAHE processing: {str(e)}")
                    msg.setWindowTitle("Error")
                    msg.setStyleSheet("""
                        QMessageBox {
                            background-color: #f0f0f0;
                        }
                        QLabel {
                            background: transparent;
                            border: none;
                            padding: 0;
                            color: #2c3e50;
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
                    """)
                    msg.exec_()
                    return

            # Step 4: Run classification and display final result (always run this step)
            try:
                # Create a visual indicator for the border thickness
                if self.border_thickness > 0:
                    # Create a copy of the mask for visualization
                    border_vis_mask = self.current_mask.copy()
                    
                    # Create the eroded mask (same as in classification)
                    kernel = np.ones((self.border_thickness, self.border_thickness), np.uint8)
                    eroded_mask = cv2.erode(border_vis_mask, kernel, iterations=1)
                    
                    # Create a visualization where the border area is highlighted
                    border_area = border_vis_mask - eroded_mask
                    
                    # Convert to RGB for visualization
                    vis_mask = np.zeros((border_vis_mask.shape[0], border_vis_mask.shape[1], 3), dtype=np.uint8)
                    vis_mask[border_vis_mask == 1] = [255, 255, 255]  # White for leaf area
                    vis_mask[border_area == 1] = [255, 0, 0]  # Red for border area
                    
                    # Display the visualization
                    h, w, ch = vis_mask.shape
                    bytes_per_line = ch * w
                    qt_vis = QImage(vis_mask.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    vis_pixmap = QPixmap.fromImage(qt_vis).scaled(250, 350, Qt.KeepAspectRatio)  # Updated height
                    self.segmentation_label.setPixmap(vis_pixmap)
                else:
                    # If border thickness is 0, show the original mask without any red border
                    mask_display = (self.current_mask * 255).astype(np.uint8)
                    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2RGB)
                    h, w, ch = mask_display.shape
                    bytes_per_line = ch * w
                    qt_mask = QImage(mask_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    mask_pixmap = QPixmap.fromImage(qt_mask).scaled(250, 350, Qt.KeepAspectRatio)  # Updated height
                    self.segmentation_label.setPixmap(mask_pixmap)
                
                QApplication.processEvents()
                
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
                result_pixmap = QPixmap.fromImage(qt_img).scaled(250, 350, Qt.KeepAspectRatio)  # Updated height
                self.result_label.setPixmap(result_pixmap)
                
                # Update percentage display
                self.percentage_label.setText(f"Disease Percentage: {percent:.2f}%")
                QApplication.processEvents()
            except Exception as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText(f"Error in classification: {str(e)}")
                msg.setWindowTitle("Error")
                msg.setStyleSheet("""
                    QMessageBox {
                        background-color: #f0f0f0;
                    }
                    QLabel {
                        background: transparent;
                        border: none;
                        padding: 0;
                        color: #2c3e50;
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
                """)
                msg.exec_()
                return

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"Error during processing: {str(e)}")
            msg.setWindowTitle("Error")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #f0f0f0;
                }
                QLabel {
                    background: transparent;
                    border: none;
                    padding: 0;
                    color: #2c3e50;
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
            """)
            msg.exec_()

    def reset_analysis(self):
        """Reset the analysis to default settings"""
        self.slider.setValue(0)
        self.border_thickness = 0
        self.slider_label.setText("Border Thickness: 0 px")
        
        # Clear results but keep the original image
        if self.image_path:
            # Reset the segmentation, CLAHE, and result labels to their initial text
            self.segmentation_label.clear()
            self.segmentation_label.setText("Segmentation will appear here")
            self.segmentation_label.setStyleSheet("""
                background-color: #f8f9fa;
                color: #6c757d;
            """)
            
            self.clahe_label.clear()
            self.clahe_label.setText("Enhanced image will appear here")
            self.clahe_label.setStyleSheet("""
                background-color: #f8f9fa;
                color: #6c757d;
            """)
            
            self.result_label.clear()
            self.result_label.setText("Final result will appear here")
            self.result_label.setStyleSheet("""
                background-color: #f8f9fa;
                color: #6c757d;
            """)
            
            self.percentage_label.setText("Disease Percentage: --")
            
            # Keep the original image display
            pixmap = QPixmap(self.image_path).scaled(250, 350, Qt.KeepAspectRatio)  # Updated height
            self.original_label.setPixmap(pixmap)
            
            # Clear stored results
            self.current_mask = None
            self.current_clahe_img = None
            self.current_original_img = None
    
    def save_results(self):
        """Save the current analysis results"""
        if not self.image_path or self.result_label.pixmap() is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No results to save. Please run the analysis first.")
            msg.setWindowTitle("Warning")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #f0f0f0;
                }
                QLabel {
                    background: transparent;
                    border: none;
                    padding: 0;
                    color: #2c3e50;
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
            """)
            msg.exec_()
            return
            
        # Get the original filename without extension
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Let user choose where to save the results
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results", 
                                                   os.path.dirname(self.image_path),
                                                   QFileDialog.ShowDirsOnly)
        
        if not save_dir:  # User cancelled the dialog
            return
            
        try:
            # Save the final result image
            result_path = os.path.join(save_dir, f"{base_name}_result.jpg")
            result_pixmap = self.result_label.pixmap()
            result_pixmap.save(result_path)
            
            # Save the percentage information
            percent_text = self.percentage_label.text()
            
            # Create a text file with analysis details
            info_path = os.path.join(save_dir, f"{base_name}_info.txt")
            with open(info_path, "w") as f:
                f.write(f"Leaf Disease Analysis Results\n")
                f.write(f"==========================\n\n")
                f.write(f"Original Image: {self.image_path}\n")
                f.write(f"Border Thickness: {self.border_thickness} px\n")
                f.write(f"{percent_text}\n")
                f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            QMessageBox.information(self, "Success", f"Results saved to {save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = LeafDiseaseGUI()
    gui.show()
    sys.exit(app.exec_())
