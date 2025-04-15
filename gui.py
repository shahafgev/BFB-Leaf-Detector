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
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter
from PyQt5.QtCore import Qt, pyqtSignal
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
            QPushButton:checked {
                background-color: #2E7D32;
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
        
        # Store the current result image for editing
        self.current_result_img = None
        self.is_editing = False
        self.brush_size = 5  # Size of the editing brush in pixels
        self.edit_mode = "diseased"  # Default edit mode: "diseased" or "healthy"
        self.edit_window = None  # Reference to the edit window

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
        original_info.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #3498db;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0;
            }
            QToolTip {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
        """)
        
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
        segmentation_info.setToolTip("Segmentation mask showing the leaf area. Red border indicates the excluded area based on border thickness.")
        segmentation_info.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #3498db;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0;
            }
            QToolTip {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
        """)
        
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
        clahe_info.setToolTip("Contrast Limited Adaptive Histogram Equalization applied to enhance the leaf image")
        clahe_info.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #3498db;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0;
            }
            QToolTip {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
        """)
        
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
        result_info.setToolTip("Final result with diseased areas highlighted in red")
        result_info.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #3498db;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0;
            }
            QToolTip {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
        """)
        
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

        # Add edit mode toggle button
        self.edit_btn = QPushButton("Edit Mode")
        self.edit_btn.clicked.connect(self.open_edit_window)
        self.edit_btn.setIcon(self.style().standardIcon(self.style().SP_FileDialogDetailedView))

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
        slider_info.setToolTip("Adjust the border thickness to exclude pixels from the edge of the leaf. This helps avoid misclassification of leaf edges.")
        slider_info.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #3498db;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0;
            }
            QToolTip {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
        """)
        
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
        button_layout.addWidget(self.edit_btn)
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
                
                # Display the visualization - use the same size as the original mask
                h, w, ch = vis_mask.shape
                bytes_per_line = ch * w
                qt_vis = QImage(vis_mask.data, w, h, bytes_per_line, QImage.Format_RGB888)
                vis_pixmap = QPixmap.fromImage(qt_vis).scaled(250, 350, Qt.KeepAspectRatio)
                self.segmentation_label.setPixmap(vis_pixmap)
            else:
                # If border thickness is 0, show the original mask without any red border
                mask_display = (self.current_mask * 255).astype(np.uint8)
                mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2RGB)
                h, w, ch = mask_display.shape
                bytes_per_line = ch * w
                qt_mask = QImage(mask_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
                mask_pixmap = QPixmap.fromImage(qt_mask).scaled(250, 350, Qt.KeepAspectRatio)
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
                    # Ensure mask dimensions match the original image
                    if self.current_mask.shape[:2] != self.current_original_img.shape[:2]:
                        # Resize mask to match original image dimensions
                        self.current_mask = cv2.resize(self.current_mask, 
                                                      (self.current_original_img.shape[1], self.current_original_img.shape[0]), 
                                                      interpolation=cv2.INTER_NEAREST)
                    
                    self.current_clahe_img = apply_clahe_only_on_leaf(self.current_original_img, self.current_mask)

                    # Display CLAHE result
                    clahe_rgb = cv2.cvtColor(self.current_clahe_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = clahe_rgb.shape
                    bytes_per_line = ch * w
                    qt_clahe = QImage(clahe_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    clahe_pixmap = QPixmap.fromImage(qt_clahe).scaled(250, 350, Qt.KeepAspectRatio)
                    
                    # Create a QLabel with the pixmap and set alignment to center
                    self.clahe_label.setPixmap(clahe_pixmap)
                    self.clahe_label.setAlignment(Qt.AlignCenter)
                    
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
                
                # Ensure mask dimensions match the original image for classification
                if self.current_mask.shape[:2] != self.current_original_img.shape[:2]:
                    # Resize mask to match original image dimensions
                    self.current_mask = cv2.resize(self.current_mask, 
                                                  (self.current_original_img.shape[1], self.current_original_img.shape[0]), 
                                                  interpolation=cv2.INTER_NEAREST)
                
                # Run classification with the original mask and CLAHE image
                overlay_img, percent = classify_leaf(
                    self.image_path,
                    mask=self.current_mask,
                    border_thickness=self.border_thickness,
                    processed_img=self.current_clahe_img
                )

                # Store the result image for editing
                self.current_result_img = overlay_img.copy()

                # Display final result
                rgb_image = cv2.cvtColor(self.current_result_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                result_pixmap = QPixmap.fromImage(qt_img).scaled(250, 350, Qt.KeepAspectRatio)

                # Create a QLabel with the pixmap and set alignment to center
                self.result_label.setPixmap(result_pixmap)
                self.result_label.setAlignment(Qt.AlignCenter)

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

    def open_edit_window(self):
        """Open a large window for editing the result image"""
        if self.current_result_img is None or self.current_mask is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please run the analysis first before editing.")
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
            
        # Create a new window for editing
        self.edit_window = EditWindow(self.current_result_img.copy(), self.current_mask, self.current_clahe_img, self)
        self.edit_window.result_updated.connect(self.update_result_from_edit)
        self.edit_window.show()

    def update_result_from_edit(self, edited_img, percent):
        """Update the main window with the edited result"""
        self.current_result_img = edited_img
        
        # Update the display
        rgb_image = cv2.cvtColor(self.current_result_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        result_pixmap = QPixmap.fromImage(qt_img).scaled(250, 350, Qt.KeepAspectRatio)
        self.result_label.setPixmap(result_pixmap)
        
        # Update the disease percentage
        self.percentage_label.setText(f"Disease Percentage: {percent:.2f}%")

    def apply_clahe(self, image_path):
        """Apply CLAHE to the image and save the result."""
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to read image")

            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # Merge channels
            limg = cv2.merge((cl, a, b))

            # Convert back to BGR
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # Save the enhanced image
            filename = os.path.basename(image_path)
            output_path = os.path.join("data/processed/clahe", filename)
            os.makedirs("data/processed/clahe", exist_ok=True)
            cv2.imwrite(output_path, enhanced)

            return enhanced

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply CLAHE: {str(e)}")
            return None

class EditWindow(QWidget):
    """A separate window for editing the result image"""
    result_updated = pyqtSignal(np.ndarray, float)  # Signal to send edited image and percentage back to main window
    
    def __init__(self, result_img, mask, clahe_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Result")
        self.resize(1000, 800)
        
        self.result_img = result_img
        self.mask = mask
        self.clahe_img = clahe_img
        self.original_img = parent.current_original_img.copy()  # Store the original image
        self.brush_size = 5
        self.edit_mode = "diseased"  # Default edit mode: "diseased" or "healthy"
        self.view_mode = "original"  # Default view mode: "original" or "enhanced"
        
        # Initialize history with the original state
        self.history = [result_img.copy()]  # Store history of changes for undo
        self.current_step = 0  # Current position in history
        
        # Track the current brush stroke
        self.is_drawing = False
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create view mode selection at the top
        view_mode_layout = QHBoxLayout()
        view_mode_label = QLabel("View Mode:")
        self.original_view_btn = QPushButton("Original Leaf")
        self.original_view_btn.setCheckable(True)
        self.original_view_btn.setChecked(True)
        self.original_view_btn.clicked.connect(lambda: self.set_view_mode("original"))
        
        self.enhanced_view_btn = QPushButton("Enhanced Leaf")
        self.enhanced_view_btn.setCheckable(True)
        self.enhanced_view_btn.clicked.connect(lambda: self.set_view_mode("enhanced"))
        
        view_mode_layout.addWidget(view_mode_label)
        view_mode_layout.addWidget(self.original_view_btn)
        view_mode_layout.addWidget(self.enhanced_view_btn)
        view_mode_layout.addStretch()
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 5px;
        """)
        
        # Update the display
        self.update_display()
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Mode selection buttons
        self.diseased_btn = QPushButton("Mark as Diseased")
        self.diseased_btn.setCheckable(True)
        self.diseased_btn.setChecked(True)
        self.diseased_btn.clicked.connect(lambda: self.set_edit_mode("diseased"))
        self.diseased_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            QPushButton:pressed {
                background-color: #7f0000;
            }
            QPushButton:checked {
                background-color: #d32f2f;
                border: 2px solid #ffffff;
            }
        """)
        
        self.healthy_btn = QPushButton("Mark as Healthy")
        self.healthy_btn.setCheckable(True)
        self.healthy_btn.clicked.connect(lambda: self.set_edit_mode("healthy"))
        self.healthy_btn.setStyleSheet("""
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
            QPushButton:checked {
                background-color: #4CAF50;
                border: 2px solid #ffffff;
            }
        """)
        
        # Brush size slider
        brush_label = QLabel("Brush Size:")
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(20)
        self.brush_slider.setValue(5)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        
        # Undo and Redo buttons
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_changes)
        self.undo_btn.setEnabled(False)  # Initially disabled
        
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo_changes)
        self.redo_btn.setEnabled(False)  # Initially disabled
        
        # Apply and Cancel buttons
        apply_btn = QPushButton("Apply Changes")
        apply_btn.clicked.connect(self.apply_changes)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel_changes)
        
        # Add widgets to controls layout
        controls_layout.addWidget(self.diseased_btn)
        controls_layout.addWidget(self.healthy_btn)
        controls_layout.addWidget(brush_label)
        controls_layout.addWidget(self.brush_slider)
        controls_layout.addWidget(self.undo_btn)
        controls_layout.addWidget(self.redo_btn)
        controls_layout.addWidget(apply_btn)
        controls_layout.addWidget(cancel_btn)
        
        # Add layouts to main layout
        layout.addLayout(view_mode_layout)
        layout.addWidget(self.image_label)
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
        
        # Enable mouse tracking
        self.image_label.mousePressEvent = self.image_label_mouse_press
        self.image_label.mouseMoveEvent = self.image_label_mouse_move
        self.image_label.mouseReleaseEvent = self.image_label_mouse_release
        self.image_label.setCursor(Qt.CrossCursor)
    
    def set_view_mode(self, mode):
        """Set the view mode (original or enhanced)"""
        self.view_mode = mode
        if mode == "original":
            self.original_view_btn.setChecked(True)
            self.enhanced_view_btn.setChecked(False)
        else:
            self.original_view_btn.setChecked(False)
            self.enhanced_view_btn.setChecked(True)
        
        # Update the display
        self.update_display()
    
    def set_edit_mode(self, mode):
        """Set the edit mode (diseased or healthy)"""
        self.edit_mode = mode
        if mode == "diseased":
            self.diseased_btn.setChecked(True)
            self.healthy_btn.setChecked(False)
        else:
            self.diseased_btn.setChecked(False)
            self.healthy_btn.setChecked(True)
    
    def update_brush_size(self, value):
        """Update the brush size"""
        self.brush_size = value
    
    def update_display(self):
        """Update the image display"""
        # Create a copy of the result image for display
        display_img = self.result_img.copy()
        
        # If in original view mode, replace non-diseased areas with original image
        if self.view_mode == "original":
            # Find pixels that are not marked as diseased (not red)
            non_diseased = ~np.all(display_img == [0, 0, 255], axis=2)
            # Replace those pixels with the original image
            display_img[non_diseased] = self.original_img[non_diseased]
        else:  # enhanced view mode
            # Find pixels that are not marked as diseased (not red)
            non_diseased = ~np.all(display_img == [0, 0, 255], axis=2)
            # Replace those pixels with the CLAHE image
            display_img[non_diseased] = self.clahe_img[non_diseased]
        
        # Convert to RGB for display
        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(800, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
    
    def image_label_mouse_press(self, event):
        """Handle mouse press events"""
        if self.result_img is None or self.mask is None:
            return
        
        # Get the position in the image
        pos = event.pos()
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if pixmap:
            # Calculate the scaling factor and offset
            pixmap_size = pixmap.size()
            scale_x = self.result_img.shape[1] / pixmap_size.width()
            scale_y = self.result_img.shape[0] / pixmap_size.height()
            
            # Calculate the offset to center the image
            offset_x = (label_size.width() - pixmap_size.width()) / 2
            offset_y = (label_size.height() - pixmap_size.height()) / 2
            
            # Convert position to image coordinates, accounting for offset
            x = int((pos.x() - offset_x) * scale_x)
            y = int((pos.y() - offset_y) * scale_y)
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, self.result_img.shape[1] - 1))
            y = max(0, min(y, self.result_img.shape[0] - 1))
            
            # Start a new brush stroke
            self.is_drawing = True
            
            # Apply the edit
            self.edit_image(x, y)
    
    def image_label_mouse_move(self, event):
        """Handle mouse move events"""
        if self.result_img is None or self.mask is None or not self.is_drawing:
            return
        
        # Get the position in the image
        pos = event.pos()
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if pixmap:
            # Calculate the scaling factor and offset
            pixmap_size = pixmap.size()
            scale_x = self.result_img.shape[1] / pixmap_size.width()
            scale_y = self.result_img.shape[0] / pixmap_size.height()
            
            # Calculate the offset to center the image
            offset_x = (label_size.width() - pixmap_size.width()) / 2
            offset_y = (label_size.height() - pixmap_size.height()) / 2
            
            # Convert position to image coordinates, accounting for offset
            x = int((pos.x() - offset_x) * scale_x)
            y = int((pos.y() - offset_y) * scale_y)
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, self.result_img.shape[1] - 1))
            y = max(0, min(y, self.result_img.shape[0] - 1))
            
            # Apply the edit if left button is pressed
            if event.buttons() & Qt.LeftButton:
                self.edit_image(x, y)
    
    def image_label_mouse_release(self, event):
        """Handle mouse release events"""
        if self.is_drawing:
            # End the current brush stroke and save it to history
            self.is_drawing = False
            
            # Store the current state before making changes
            if len(self.history) > self.current_step + 1:
                # If we've undone some changes, remove the future history
                self.history = self.history[:self.current_step + 1]
            
            # Save the current state
            self.history.append(self.result_img.copy())
            self.current_step += 1
            
            # Update button states
            self.update_undo_redo_buttons()
    
    def update_undo_redo_buttons(self):
        """Update the enabled state of undo and redo buttons"""
        # Enable undo button if we have history to undo
        self.undo_btn.setEnabled(self.current_step > 0)
        
        # Enable redo button if we have history to redo
        self.redo_btn.setEnabled(self.current_step < len(self.history) - 1)
    
    def edit_image(self, x, y):
        """Edit the image at the given coordinates"""
        if self.result_img is None or self.mask is None:
            return
            
        # Only allow editing within the leaf mask
        if self.mask[y, x] == 0:
            return
            
        # Create a circular brush
        y_indices, x_indices = np.ogrid[-self.brush_size:self.brush_size+1, -self.brush_size:self.brush_size+1]
        mask = x_indices*x_indices + y_indices*y_indices <= self.brush_size*self.brush_size
        
        # Apply the brush
        for dy in range(-self.brush_size, self.brush_size+1):
            for dx in range(-self.brush_size, self.brush_size+1):
                if mask[dy+self.brush_size, dx+self.brush_size]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.result_img.shape[0] and 
                        0 <= nx < self.result_img.shape[1] and 
                        self.mask[ny, nx] == 1):
                        if self.edit_mode == "diseased":
                            self.result_img[ny, nx] = [0, 0, 255]  # Red for diseased
                        else:
                            # Restore original color from original image
                            self.result_img[ny, nx] = self.original_img[ny, nx]
        
        # Update the display
        self.update_display()
    
    def undo_changes(self):
        """Undo the last change"""
        if self.current_step > 0:
            self.current_step -= 1
            self.result_img = self.history[self.current_step].copy()
            self.update_display()
            self.update_undo_redo_buttons()
    
    def redo_changes(self):
        """Redo the last undone change"""
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            self.result_img = self.history[self.current_step].copy()
            self.update_display()
            self.update_undo_redo_buttons()
    
    def cancel_changes(self):
        """Cancel all changes and close the window"""
        self.close()
    
    def apply_changes(self):
        """Apply changes and send the result back to the main window"""
        # Calculate the disease percentage
        red_pixels = np.all(self.result_img == [0, 0, 255], axis=2)
        total_leaf_pixels = np.sum(self.mask == 1)
        
        if total_leaf_pixels > 0:
            percent = (np.sum(red_pixels & (self.mask == 1)) / total_leaf_pixels) * 100
        else:
            percent = 0.0
            
        # Send the result back to the main window
        self.result_updated.emit(self.result_img, percent)
        
        # Close the edit window
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set custom tooltip style
    app.setStyleSheet("""
        QToolTip {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #4CAF50;
            border-radius: 4px;
            padding: 5px;
            font-size: 12px;
        }
    """)
    
    gui = LeafDiseaseGUI()
    gui.show()
    sys.exit(app.exec_())
