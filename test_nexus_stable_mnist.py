"""
NEXUS-Stable Optimizer on MNIST with CNN, Early Stopping, and F1 Score

This script uses the stable version of NEXUS (NEXUS_Stable) which removes
the problematic meta-learning adaptive learning rate feature and focuses on
the 7 stable features for better convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from optimizers.nexus_stable import NEXUS_Stable
from models.cnn_mnist import SimpleCNN


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0.0001, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            verbose (bool): Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.Inf
    
    def __call__(self, val_loss, val_acc, model):
        """
        Call this after each epoch to check early stopping condition.
        
        Args:
            val_loss: Validation loss
            val_acc: Validation accuracy
            model: Model to save if best
        """
        score = -val_loss  # Use negative loss (higher is better)
        
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, val_acc, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                print(f'Best loss: {self.best_loss:.6f}, Current loss: {val_loss:.6f}')
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, val_acc, model)
            self.counter = 0
            if self.verbose:
                print(f'Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f}). Resetting counter.')
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f'Early stopping triggered after {self.counter} epochs without improvement!')
    
    def save_checkpoint(self, val_loss, val_acc, model):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, 'nexus_stable_mnist_best.pth')


def get_data_loaders(batch_size=128):
    """Get MNIST train and test data loaders."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # No augmentation for test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    # Split train into train and validation (90-10 split)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def calculate_metrics(model, data_loader, device):
    """Calculate accuracy, precision, recall, and F1 score."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.6f}')
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training function."""
    print("=" * 80)
    print("NEXUS-Stable Optimizer on MNIST with CNN")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - Model: SimpleCNN")
    print("  - Optimizer: NEXUS-Stable (stable version)")
    print("  - Epochs: 100")
    print("  - Early Stopping: Enabled (patience=7)")
    print("  - Metrics: Accuracy, Precision, Recall, F1 Score")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Get data loaders
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=128)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = SimpleCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Create NEXUS-Stable optimizer
    print("\nInitializing NEXUS-Stable optimizer...")
    optimizer = NEXUS_Stable(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.99, 0.999),
        momentum_scales=3,
        direction_consistency_alpha=0.1,
        layer_adaptation=True,
        lookahead_steps=5,
        lookahead_alpha=0.5,
        noise_scale=0.01,
        noise_decay=0.99,
        curvature_window=10,
        adaptive_weight_decay=True,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        state_precision='fp32'
    )
    print("  NEXUS-Stable optimizer created successfully!")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.0001, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # Training loop
    num_epochs = 100
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    start_time = time.time()
    best_epoch = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_acc, val_precision, val_recall, val_f1 = calculate_metrics(model, val_loader, device)
        val_loss = 1.0 - val_acc / 100.0  # Approximate loss from accuracy
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
        
        # Get NEXUS-Stable statistics
        dir_stats = optimizer.get_direction_consistency_stats()
        curv_stats = optimizer.get_curvature_stats()
        print(f'  NEXUS-Stable - DirCons: {dir_stats["mean_consistency"]:.4f}, '
              f'Curv: {curv_stats["mean_curvature"]:.4f}')
        
        # Early stopping check
        early_stopping(val_loss, val_acc, model)
        
        if early_stopping.early_stop:
            print(f'\n{"=" * 80}')
            print(f'Early stopping triggered at epoch {epoch+1}!')
            print(f'Best validation loss: {early_stopping.best_loss:.6f} at epoch {best_epoch}')
            print(f'{"=" * 80}')
            break
        
        # Update best epoch
        if val_loss < early_stopping.best_loss:
            best_epoch = epoch + 1
    
    # Final test evaluation
    print(f'\n{"=" * 80}')
    print('Training completed! Evaluating on test set...')
    print("=" * 80)
    
    test_acc, test_precision, test_recall, test_f1 = calculate_metrics(model, test_loader, device)
    
    total_time = time.time() - start_time
    
    # Print final results
    print(f'\n{"=" * 80}')
    print('FINAL RESULTS')
    print("=" * 80)
    print(f'Total training time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)')
    print(f'Epochs completed: {epoch+1} (out of {num_epochs})')
    print(f'Best epoch: {best_epoch}')
    print(f'\nTest Set Performance:')
    print(f'  Accuracy:  {test_acc:.2f}%')
    print(f'  Precision: {test_precision:.4f}')
    print(f'  Recall:    {test_recall:.4f}')
    print(f'  F1 Score:  {test_f1:.4f}')
    
    # NEXUS-Stable final statistics
    print(f'\nNEXUS-Stable Optimizer Final Statistics:')
    print("=" * 80)
    
    dir_stats = optimizer.get_direction_consistency_stats()
    print(f'Gradient Direction Consistency:')
    print(f'  Mean: {dir_stats["mean_consistency"]:.6f}')
    print(f'  Std:  {dir_stats["std_consistency"]:.6f}')
    print(f'  Min:  {dir_stats["min_consistency"]:.6f}')
    print(f'  Max:  {dir_stats["max_consistency"]:.6f}')
    
    curv_stats = optimizer.get_curvature_stats()
    print(f'\nCurvature Estimates:')
    print(f'  Mean: {curv_stats["mean_curvature"]:.6f}')
    print(f'  Std:  {curv_stats["std_curvature"]:.6f}')
    print(f'  Min:  {curv_stats["min_curvature"]:.6f}')
    print(f'  Max:  {curv_stats["max_curvature"]:.6f}')
    
    mom_stats = optimizer.get_momentum_stats()
    print(f'\nMulti-Scale Momentum:')
    print(f'  Scale 0 (short-term):  Mean={mom_stats["scale_0"]["mean"]:.6f}, '
          f'Std={mom_stats["scale_0"]["std"]:.6f}')
    print(f'  Scale 1 (medium-term):  Mean={mom_stats["scale_1"]["mean"]:.6f}, '
          f'Std={mom_stats["scale_1"]["std"]:.6f}')
    print(f'  Scale 2 (long-term):  Mean={mom_stats["scale_2"]["mean"]:.6f}, '
          f'Std={mom_stats["scale_2"]["std"]:.6f}')
    
    mem_stats = optimizer.get_memory_usage()
    print(f'\nMemory Usage:')
    print(f'  Total: {mem_stats["total_mb"]:.2f} MB')
    print(f'  State: {mem_stats["state_mb"]:.2f} MB')
    print(f'  Parameters: {mem_stats["param_mb"]:.2f} MB')
    
    # Plot training curves
    plot_training_curves(history, best_epoch)
    
    print(f'\n{"=" * 80}')
    print('Training completed successfully!')
    print(f'Best model saved to: nexus_stable_mnist_best.pth')
    print("=" * 80)


def plot_training_curves(history, best_epoch):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NEXUS-Stable Optimizer Training Progress on MNIST', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1, Precision, Recall curves
    axes[0, 2].plot(epochs, history['val_precision'], label='Precision', linewidth=2)
    axes[0, 2].plot(epochs, history['val_recall'], label='Recall', linewidth=2)
    axes[0, 2].plot(epochs, history['val_f1'], label='F1 Score', linewidth=2, color='green')
    axes[0, 2].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Score', fontsize=12)
    axes[0, 2].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Direction consistency over time
    axes[1, 0].plot(epochs, [1.0] * len(epochs), 'k--', alpha=0.3, label='Target Consistency')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Direction Consistency', fontsize=12)
    axes[1, 0].set_title('Gradient Direction Consistency', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Curvature over time
    axes[1, 1].plot(epochs, [1.0] * len(epochs), 'k--', alpha=0.3, label='Initial Curvature')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Curvature', fontsize=12)
    axes[1, 1].set_title('Curvature Estimates', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nexus_stable_mnist_training_curves.png', dpi=150, bbox_inches='tight')
    print(f'\nTraining curves saved to: nexus_stable_mnist_training_curves.png')
    plt.close()


if __name__ == '__main__':
    main()

