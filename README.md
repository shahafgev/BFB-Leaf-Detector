# BFB Leaf Detector

A machine learning-based system for detecting and classifying Banana Fusarium Wilt (BFB) in banana leaves using computer vision and machine learning techniques.

## Overview

This project provides a comprehensive solution for detecting and classifying Banana Fusarium Wilt (BFB) in banana leaves. It combines computer vision techniques with machine learning models to analyze leaf images and identify signs of the disease. The system includes both a graphical user interface for easy interaction and a robust backend for image processing and classification.

## Features

- **Image Processing Pipeline**:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced leaf visibility
  - SAM (Segment Anything Model) for precise leaf segmentation
  - HSV color space conversion for better feature extraction

- **Machine Learning Models**:
  - Multiple classifier options (Logistic Regression, Random Forest)
  - Balanced dataset training to handle class imbalance
  - Hyperparameter tuning via GridSearchCV
  - Comprehensive model evaluation with confusion matrices

- **User Interface**:
  - Intuitive GUI for image upload and processing
  - Real-time visualization of processing steps
  - Detailed classification results display

## Installation

### Prerequisites

- Python 3.8+
- PyQt5
- OpenCV
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Segment Anything Model (SAM)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/BFB-Leaf-Detector.git
   cd BFB-Leaf-Detector
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the SAM model checkpoint:
   - Download `sam_vit_b.pth` from [Meta AI's SAM repository](https://github.com/facebookresearch/segment-anything)
   - Place it in the `checkpoints` directory

## Usage

### Training Models

To train the classification models with balanced dataset:

```
python train.py
```

This will:
- Load and preprocess the dataset
- Balance the training data by downsampling the majority class
- Train and tune multiple models
- Evaluate models and save the best one
- Generate confusion matrices in the `outputs` directory

### Running the GUI

To launch the graphical user interface:

```
python gui.py
```

The GUI allows you to:
1. Upload leaf images
2. Process images with CLAHE and SAM
3. Classify leaves as healthy or infected
4. View detailed results and visualizations

## Project Structure

```
BFB-Leaf-Detector/
├── checkpoints/           # Model checkpoints (SAM)
├── models/                # Trained classification models
├── outputs/               # Evaluation results and visualizations
├── processed_data/        # Processed datasets
├── src/
│   ├── data/              # Data processing utilities
│   ├── modeling/          # Model training and evaluation
│   └── visualization/     # Visualization utilities
├── gui.py                 # Graphical user interface
├── train.py               # Model training script
└── requirements.txt       # Project dependencies
```

## Data

The system uses a dataset of 36 melon leaf images with the following features:
- RGB color channels
- HSV color space conversion
- Balanced training data to handle class imbalance

## Model Performance

The models are evaluated using:
- Accuracy
- Confusion matrices
- Classification reports

Results are saved in the `outputs` directory after training.

## License

[Your chosen license]

## Acknowledgments

- Meta AI for the Segment Anything Model (SAM)
- [Any other acknowledgments] 
