## E-Waste Classification with AlexNet - WAVE Competition Entry

# Competition Overview
This repository contains our submission for the Florida Atlantic University Wave Competition, featuring an AlexNet-based deep learning model for automated electronic waste classification. My solution addresses the critical challenge of e-waste sorting in recycling facilities, contributing to more sustainable waste management practices.

# Project Objective
Develop an AI-powered visual classification system capable of accurately identifying and categorizing different types of electronic waste to improve recycling efficiency and reduce environmental impact.

# Model Architecture
AlexNet Modified for E-Waste Classification

Base Architecture: Modified AlexNet with 8 layers (5 convolutional + 3 fully connected)
Input Size: 224x224x3 RGB images
Output Classes: 5 categories of electronic waste
Framework: PyTorch
Total Parameters: ~60M parameters

# Model Specifications
Input Layer: 224x224x3
Conv1: 64 filters, 11x11 kernel, stride 4
Conv2: 192 filters, 5x5 kernel
Conv3: 384 filters, 3x3 kernel
Conv4: 256 filters, 3x3 kernel  
Conv5: 256 filters, 3x3 kernel
FC1: 4096 neurons
FC2: 4096 neurons
Output: 5 classes (softmax)

# Dataset categories
My model classifies e-waste into the following categories:

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
    
