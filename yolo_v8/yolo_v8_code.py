# Install YOLOv8
!pip install ultralytics

from ultralytics import YOLO
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

print("🚀 Simple YOLOv8 Classification Setup")
print("="*50)

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Simple function to train YOLOv8 directly on your folder structure
def train_yolo_simple(data_root, model_size='n', epochs=50):
    """
    Train YOLOv8 classification model directly on your existing folder structure

    Your folders should be organized like:
    data_root/
    ├── train/
    │   ├── smartphones/
    │   ├── laptops/
    │   ├── tablets/
    │   ├── circuit_boards/
    │   └── other_electronics/
    └── test/  (or val/)
        ├── smartphones/
        ├── laptops/
        ├── tablets/
        ├── circuit_boards/
        └── other_electronics/
    """

    print(f"📂 Training on data from: {data_root}")

    # Check if data exists
    data_path = Path(data_root)
    train_path = data_path / 'train'

    # Check for test or val folder
    val_path = data_path / 'test' if (data_path / 'test').exists() else data_path / 'val'

    if not train_path.exists():
        raise FileNotFoundError(f"Training folder not found: {train_path}")

    if not val_path.exists():
        print("⚠️ No validation folder found, will use train data for validation")
        val_path = train_path

    print(f"✅ Train folder: {train_path}")
    print(f"✅ Val folder: {val_path}")

    # Show classes
    classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    print(f"📋 Classes found: {classes}")

    # Count images
    total_train = sum(len(list((train_path / cls).glob('*.*'))) for cls in classes)
    total_val = sum(len(list((val_path / cls).glob('*.*'))) for cls in classes)
    print(f"📊 Train images: {total_train}")
    print(f"📊 Val images: {total_val}")

    # Load YOLOv8 classification model
    print(f"🏗️ Loading YOLOv8{model_size}-cls model...")
    model = YOLO(f'yolov8{model_size}-cls.pt')

    # Train the model - SUPER SIMPLE!
    print(f"🚀 Starting training for {epochs} epochs...")

    results = model.train(
        data=str(data_path),  # Just pass the root data folder!
        epochs=epochs,
        batch=32 if device == 'cuda' else 8,
        device=device,
        patience=15,
        save=True,
        plots=True,
        val=True,
        project='yolo_runs',  # Where to save results
        name='ewaste_classification'
    )

    print("✅ Training complete!")
    print(f"📁 Results saved in: yolo_runs/ewaste_classification/")

    return model, classes

# Even simpler - just specify your data folder!
def quick_train():
    """One-liner training function"""

    # Your data path
    data_root = '/content/E-Waste-Classification/yolo_v8/data

    # Check if repo exists
    if not os.path.exists('/content/E-Waste-Classification'):
        print("📥 Cloning repository...")
        os.chdir('/content')
        os.system('git clone https://github.com/PatrickRapalo/E-Waste-Classification.git')

    # Train model
    print("🎯 Training YOLOv8 on your E-Waste data...")
    model, classes = train_yolo_simple(data_root, model_size='s', epochs=30)

    return model, classes

# Prediction function
def predict_image(model, image_path, class_names):
    """Predict on a single image"""
    results = model(image_path)
    result = results[0]

    # Get predictions
    probs = result.probs
    top1_idx = probs.top1
    top1_conf = probs.top1conf

    # Show results
    image = Image.open(image_path)
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Prediction: {class_names[top1_idx]}\nConfidence: {top1_conf:.3f}')

    # Show all probabilities
    plt.subplot(1, 2, 2)
    all_probs = probs.data.cpu().numpy()
    bars = plt.bar(class_names, all_probs)
    bars[top1_idx].set_color('red')
    plt.xticks(rotation=45)
    plt.title('Class Probabilities')
    plt.ylabel('Probability')

    plt.tight_layout()
    plt.show()

    return class_names[top1_idx], top1_conf

# Load saved model
def load_model(model_path=None):
    """Load a trained model"""
    if model_path is None:
        model_path = 'yolo_runs/ewaste_classification/weights/best.pt'

    print(f"📂 Loading model from: {model_path}")
    model = YOLO(model_path)
    return model

# Test on sample images
def test_model(model, data_root, class_names, num_samples=5):
    """Test model on random sample images"""

    test_path = Path(data_root) / 'test'
    if not test_path.exists():
        test_path = Path(data_root) / 'val'
    if not test_path.exists():
        print("No test folder found")
        return

    # Get some random images
    all_images = []
    for class_folder in test_path.iterdir():
        if class_folder.is_dir():
            images = list(class_folder.glob('*.*'))[:2]  # 2 per class
            all_images.extend(images)

    # Test on first few images
    for i, img_path in enumerate(all_images[:num_samples]):
        print(f"\n🔍 Testing image {i+1}: {img_path.name}")
        pred_class, confidence = predict_image(model, str(img_path), class_names)
        true_class = img_path.parent.name

        status = "✅" if pred_class == true_class else "❌"
        print(f"{status} True: {true_class}, Predicted: {pred_class} ({confidence:.3f})")

# ULTRA SIMPLE USAGE
print("🎯 ULTRA SIMPLE YOLOV8 TRAINING")
print("Just run: model, classes = quick_train()")
print("")
print("📋 Step by step:")
print("1. model, classes = quick_train()  # Train the model")
print("2. test_model(model, 'data_path', classes)  # Test it")
print("3. predict_image(model, 'image.jpg', classes)  # Predict single image")

# Uncomment to run automatically:
model, classes = quick_train()
