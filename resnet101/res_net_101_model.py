# ResNet-101 E-Waste Classification
# Google Colab Implementation

# Install required packages
!pip install torch torchvision matplotlib scikit-learn pillow numpy tqdm

# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm
import time
import copy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define data transforms
# Training transforms with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
def load_data(data_dir='/content/E-Waste-Classification/wave_competition_res_101_code/data', batch_size=32):
    """Load the E-Waste datasets"""
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Load datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Print dataset info
    print(f"Classes: {train_dataset.classes}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, test_loader, train_dataset.classes

# Create ResNet-101 model
def create_resnet101(num_classes=5, pretrained=True):
    """Create ResNet-101 model for E-Waste classification"""
    # Load pre-trained ResNet-101
    model = models.resnet101(pretrained=pretrained)
    
    # Get the number of features in the final fully connected layer
    num_features = model.fc.in_features
    
    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=25, patience=7):
    """Train the ResNet-101 model"""
    since = time.time()
    
    # Initialize tracking variables
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar
            pbar = tqdm(dataloader, desc=f'{phase.capitalize()} Epoch {epoch+1}')
            
            # Iterate over data
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
                if scheduler is not None:
                    scheduler.step()
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())
                
                # Deep copy the model if it's the best so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f'New best model saved with accuracy: {best_acc:.4f}')
                else:
                    epochs_no_improve += 1
        
        print()
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, train_losses, train_accuracies, val_losses, val_accuracies

# Evaluation function
def evaluate_model(model, test_loader, class_names):
    """Evaluate the model and generate detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    # Classification report
    print('\nDetailed Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - ResNet-101 E-Waste Classification', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Per-class accuracy
    print('\nPer-class Accuracy:')
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f'{class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)')
    
    return accuracy, all_preds, all_labels

# Plotting function
def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training and validation metrics"""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, [acc*100 for acc in train_accuracies], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc*100 for acc in val_accuracies], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Prediction function for single images
def predict_single_image(model, image_path, transform, class_names, device):
    """Predict class for a single image"""
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence = probabilities[predicted.item()].item()
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.3f}', fontsize=14)
    
    # Show probability distribution
    probs = probabilities.cpu().numpy()
    bars = ax2.bar(class_names, probs, color='skyblue', alpha=0.7)
    bars[predicted.item()].set_color('red')
    ax2.set_title('Class Probabilities', fontsize=14)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence

# Main execution
def main():
    """Main function to run the E-Waste classification training"""
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    PATIENCE = 7
    
    print("="*60)
    print("ResNet-101 E-Waste Classification Training")
    print("="*60)
    
    try:
        # Load data
        print("Loading datasets...")
        train_loader, test_loader, class_names = load_data(batch_size=BATCH_SIZE)
        
        # Create model
        print(f"\nCreating ResNet-101 model for {len(class_names)} classes...")
        model = create_resnet101(num_classes=len(class_names), pretrained=True)
        model = model.to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)
        
        # Train model
        print(f"\nStarting training for {NUM_EPOCHS} epochs...")
        model, train_losses, train_accs, val_losses, val_accs = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, 
            num_epochs=NUM_EPOCHS, patience=PATIENCE
        )
        
        # Plot training history
        print("\nPlotting training history...")
        plot_training_history(train_losses, train_accs, val_losses, val_accs)
        
        # Evaluate model
        print("\nEvaluating model on test set...")
        test_accuracy, _, _ = evaluate_model(model, test_loader, class_names)
        
        # Save model
        print("\nSaving model...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': class_names,
            'test_accuracy': test_accuracy
        }, 'resnet101_ewaste_model.pth')
        print("Model saved as 'resnet101_ewaste_model.pth'")
        
        return model, class_names
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure your data directory structure is:")
        print("data/")
        print("├── train/")
        print("│   ├── smartphones/")
        print("│   ├── laptops/")
        print("│   ├── tablets/")
        print("│   ├── circuit_boards/")
        print("│   └── other_electronics/")
        print("└── test/")
        print("    ├── smartphones/")
        print("    ├── laptops/")
        print("    ├── tablets/")
        print("    ├── circuit_boards/")
        print("    └── other_electronics/")

# Run the main function
if __name__ == "__main__":
    model, class_names = main()

# Example usage for prediction (uncomment to use)
# predict_single_image(model, 'path_to_your_image.jpg', test_transform, class_names, device)
