# Objection Detection for Breast Cancer Screening

This repository contains the implementation and code used for my MSc thesis project at Imperial College London on the application of deep learning techniques for breast cancer screening using mammography images. The primary focus of the project is object detection methodologies aimed at identifying and localizing abnormalities in mammography images.

## Project Overview

Breast cancer is one of the leading causes of cancer-related deaths among women worldwide. Early detection through screening is essential to improve survival rates. This project leverages advanced object detection models to assist radiologists by detecting potential lesions in mammography images. The models aim to enhance detection accuracy and reduce false positives and false negatives, improving the diagnostic process.

## Models Implemented

The project implements two main object detection architectures:

- **Faster R-CNN**: A two-stage model known for its high accuracy, well-suited for identifying subtle abnormalities in breast tissue.
- **RetinaNet**: A one-stage model optimized for detecting objects at multiple scales, particularly useful for addressing class imbalance in medical datasets using its focal loss function.

The models were trained and evaluated on publicly available mammography datasets, including **VinDr-Mammo** and **EMBED**, which provide high-quality annotated mammography images.

## Directory Structure

- `anchor_optimization.py`: Script for anchor box size optimization.
- `anchor_utils.py`: Utilities for generating anchor boxes.
- `cbamretina.py`: RetinaNet model with Convolutional Block Attention Module (CBAM) integration.
- `ensemble.py`: Ensemble learning implementation for multiple trained RetinaNet models. 
- `evaluate_cbam.py`: Evaluation script for testing CBAM integrated models.
- `faster_rcnn.py`: Implementation of Faster R-CNN architecture for object detection.
- `merger_scripy.py`: Method for merging EMBED metadata and clinical data into one csv file. 
- `model.py`: RetinaNet implementation adapted for mammography images.
- `multi-class.py`: Script to handle RetinaNet multi-class classification and detection.
- `resnet50.py`: Adaptation of the ResNet50 model used as a backbone for feature extraction.
- `retinanet.py`: Adjusted copy of retinanet.py from pytorch library.
- `preprocessor.ipynb`: Jupyter notebook for data preprocessing and conversion to CSV format for model training.
- `results.ipynb`: Notebook to analyze model results.
- `displays.ipynb`: Jupyter notebook for plotting results used in the final report.

## Datasets

- **VinDr-Mammo**: A mammography dataset from Vietnam, containing 5,000 exams and 20,000 images annotated with BI-RADS scores and breast density information.
- **EMBED**: A large-scale dataset with over 3.4 million images, including detailed lesion-level annotations and demographic information.
