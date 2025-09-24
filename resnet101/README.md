## E-Waste Classification with ResNet-101 - WAVE Competition Entry

# Competition Overview
This repository contains our submission for the Florida Atlantic University Wave Competition, featuring a ResNet-101 based deep learning model for automated electronic waste classification. This solution addresses the critical challenge of e-waste sorting in recycling facilities, contributing to more sustainable waste management practices.

# Project Objective
Develop an AI-powered visual classification system capable of accurately identifying and categorizing different types of electronic waste to improve recycling efficiency and reduce environmental impact using state-of-the-art residual network architecture.

# Model Architecture
ResNet-101 Modified for E-Waste Classification

Base Architecture:
Modified ResNet-101 with 101 layers featuring residual connections

Input Size: 224x224x3 RGB images

Output Classes: 5 categories of electronic waste

Framework: PyTorch

Total Parameters: ~44M parameters

# Model Specifications
Input Layer: 224x224x3

Initial Conv: 64 filters, 7x7 kernel, stride 2

Max Pooling: 3x3 kernel, stride 2

Residual Blocks:
- conv2_x: 3 blocks, 64 filters
- conv3_x: 4 blocks, 128 filters
- conv4_x: 23 blocks, 256 filters
- conv5_x: 3 blocks, 512 filters

Global Average Pooling

FC Layer: 512 → 5 classes

Output: 5 classes (softmax)

# Key Features
- Skip connections for gradient flow
- Batch normalization for training stability
- Deep architecture for complex feature learning
- Pre-trained weights for transfer learning
- Dropout for regularization

# Dataset categories
The model classifies e-waste into the following categories:

Smartphones/ Mobile devices and accessories/ iPhones, Android phones, cases

Laptops / Portable computers / Notebooks, ultrabooks, gaming laptops

Tablets / Tablet computers / iPads, Android tablets, e-readers

Circuit Boards / PCBs and electronic components / Motherboards, RAM, processors

Other Electronics / Miscellaneous e-waste / Cables, chargers, small devices

# Prerequisites

Python >= 3.7

PyTorch >= 1.9.0

torchvision >= 0.10.0

PIL (Pillow)

matplotlib

scikit-learn

numpy

# Data Organization
data/

├── train/

    ├── smartphones/

    ├── laptops/

    ├── tablets/

    ├── circuit_boards/

    └── other_electronics/

└── test/

    ├── smartphones/

    ├── laptops/

    ├── tablets/

    ├── circuit_boards/

    └── other_electronics/

# Usage
Run the ResNet-101 model:
```python
python res_net_101_model.py
```

# Model Advantages
- Superior gradient flow through skip connections
- Excellent performance on image classification tasks
- Robust feature extraction capabilities
- Pre-trained ImageNet weights available
- Proven architecture for computer vision