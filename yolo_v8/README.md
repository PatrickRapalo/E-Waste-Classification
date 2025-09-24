## E-Waste Classification with YOLO v8 - WAVE Competition Entry

# Competition Overview
This repository contains our submission for the Florida Atlantic University Wave Competition, featuring a YOLO v8 based deep learning model for automated electronic waste classification. This solution addresses the critical challenge of e-waste sorting in recycling facilities, contributing to more sustainable waste management practices with real-time object detection capabilities.

# Project Objective
Develop an AI-powered visual classification and detection system capable of accurately identifying and categorizing different types of electronic waste in real-time to improve recycling efficiency and reduce environmental impact using state-of-the-art object detection architecture.

# Model Architecture
YOLO v8 Modified for E-Waste Classification

Base Architecture:
YOLO v8 (You Only Look Once) - Latest version with improved accuracy and speed

Input Size: 640x640x3 RGB images (scalable)

Output Classes: 5 categories of electronic waste

Framework: Ultralytics YOLOv8

Total Parameters: ~11M parameters (YOLOv8n) to ~68M parameters (YOLOv8x)

# Model Specifications
Input Layer: 640x640x3 (default, configurable)

Backbone: CSPDarknet with C2f modules

Neck: Feature Pyramid Network (FPN) + Path Aggregation Network (PANet)

Head: Decoupled detection head

Anchor-free detection with:
- Classification branch
- Bounding box regression
- Objectness prediction

Output: Real-time detection with bounding boxes and class predictions

# Key Features
- Anchor-free detection architecture
- Real-time inference capabilities
- Multi-scale object detection
- Advanced data augmentation
- Mosaic and MixUp training
- Efficient model variants (n, s, m, l, x)
- Auto-anchor optimization
- State-of-the-art mAP performance

# Dataset categories
The model detects and classifies e-waste into the following categories:

Smartphones/ Mobile devices and accessories/ iPhones, Android phones, cases

Laptops / Portable computers / Notebooks, ultrabooks, gaming laptops

Tablets / Tablet computers / iPads, Android tablets, e-readers

Circuit Boards / PCBs and electronic components / Motherboards, RAM, processors

Other Electronics / Miscellaneous e-waste / Cables, chargers, small devices

# Prerequisites

Python >= 3.8

ultralytics >= 8.0.0

torch >= 1.8.0

torchvision >= 0.9.0

opencv-python

PIL (Pillow)

matplotlib

numpy

PyYAML

# Data Organization
data/

├── train/

    ├── images/

    └── labels/

├── test/

    ├── images/

    └── labels/

└── val/

    ├── images/

    └── labels/

# YOLO Format
Labels should be in YOLO format:
- One .txt file per image
- One row per object
- Format: class x_center y_center width height
- Values normalized to [0, 1]

# Usage
Run the YOLO v8 model:
```python
python yolo_v8_code.py
```

Train custom model:
```bash
yolo train data=custom_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

# Model Variants
- YOLOv8n: Nano - Fastest inference, lowest accuracy
- YOLOv8s: Small - Balanced speed and accuracy
- YOLOv8m: Medium - Good accuracy, moderate speed
- YOLOv8l: Large - High accuracy, slower inference
- YOLOv8x: Extra Large - Highest accuracy, slowest inference

# Model Advantages
- Real-time object detection and classification
- Superior speed-accuracy trade-off
- No anchor boxes needed
- Advanced loss functions
- Built-in augmentation strategies
- Easy deployment and inference
- Comprehensive metrics and visualization