"""
MNIST Classification Example with AMCAS Optimizer

This example demonstrates training a simple CNN on MNIST dataset
using the AMCAS optimizer and comparing it with other optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm

# Import AMCAS optimizer
import sys
sys.path.append('..')
from optimizers.amcas import AMCAS


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def load_mnist_data(batch_size=64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'Loss': loss.item(),
                'Acc': 100. * correct / total
            })
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(optimizer_name, optimizer_class, optimizer_params, 
                num_epochs=10, lr=0.001, device='cpu'):
    """Train model with specified optimizer."""
    print(f"\nTraining with {optimizer_name}...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Create model
    model = SimpleCNN().to(device)
    
    # Create optimizer
    optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_params)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'time_per_epoch': []
    }
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # Test
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        epoch_time = time.time() - start_time
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['time_per_epoch'].append(epoch_time)
        
        print(f'Epoch {epoch}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')
    
    return model, history


def compare_optimizers(num_epochs=10, lr=0.001, device='cpu'):
    """Compare different optimizers on MNIST."""
    print("=" * 60)
    print("MNIST Optimizer Comparison")
    print("=" * 60)
    
    # Define optimizers to compare
    optimizers = {
        'AMCAS': (AMCAS, {'betas': (0.9, 0.999), 'gamma': 0.1, 'lambda_consistency': 0.01}),
        'Adam': (torch.optim.Adam, {'betas': (0.9, 0.999)}),
        'SGD': (torch.optim.SGD, {'momentum': 0}),
        'SGD+Momentum': (torch.optim.SGD, {'momentum': 0.9}),
        'RMSprop': (torch.optim.RMSprop, {}),
    }
    
    results = {}
    
    for name, (optimizer_class, params) in optimizers.items():
        model, history = train_model(
            name, optimizer_class, params, num_epochs, lr, device
        )
        results[name] = history
        
        # Save model checkpoint
        torch.save(model.state_dict(), f'mnist_{name.lower()}_model.pth')
    
    return results


def plot_results(results, output_dir='mnist_results'):
    """Plot comparison results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot training loss
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['train_loss'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Training Loss Convergence')
    
    # Plot test accuracy
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['test_acc'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Test Accuracy')
    
    # Plot training accuracy
    ax = axes[1, 0]
    for name, history in results.items():
        ax.plot(history['train_acc'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Training Accuracy')
    
    # Plot time per epoch
    ax = axes[1, 1]
    for name, history in results.items():
        avg_time = np.mean(history['time_per_epoch'])
        ax.bar(name, avg_time)
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Average Time per Epoch (s)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Computational Efficiency')
    
    plt.tight_layout()
    plt.savefig(output_path / 'mnist_comparison.png', dpi=150)
    plt.close()
    
    # Create summary table
    print("\n" + "=" * 60)
    print("Final Results Summary")
    print("=" * 60)
    
    summary_data = []
    for name, history in results.items():
        final_test_acc = history['test_acc'][-1]
        final_train_acc = history['train_acc'][-1]
        final_test_loss = history['test_loss'][-1]
        avg_time = np.mean(history['time_per_epoch'])
        
        summary_data.append({
            'optimizer': name,
            'final_test_accuracy': final_test_acc,
            'final_train_accuracy': final_train_acc,
            'final_test_loss': final_test_loss,
            'avg_time_per_epoch': avg_time,
            'generalization_gap': final_train_acc - final_test_acc
        })
    
    # Sort by test accuracy
    summary_data.sort(key=lambda x: x['final_test_accuracy'], reverse=True)
    
    print("\nRanked by Test Accuracy:")
    print("-" * 60)
    print(f"{'Optimizer':<15} {'Test Acc':<12} {'Train Acc':<12} {'Test Loss':<12} {'Time/Epoch':<12} {'Gen Gap':<12}")
    print("-" * 60)
    
    for data in summary_data:
        print(f"{data['optimizer']:<15} "
              f"{data['final_test_accuracy']:>10.2f}% "
              f"{data['final_train_accuracy']:>10.2f}% "
              f"{data['final_test_loss']:>11.4f} "
              f"{data['avg_time_per_epoch']:>11.2f}s "
              f"{data['generalization_gap']:>11.2f}%")
    
    # Save results
    with open(output_path / 'mnist_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_path / 'mnist_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}/")
    print(f"Best optimizer: {summary_data[0]['optimizer']}")
    
    return summary_data


def analyze_optimizer_statistics(results):
    """Analyze optimizer statistics if available."""
    print("\n" + "=" * 60)
    print("Optimizer Statistics Analysis")
    print("=" * 60)
    
    # This would require modifying the training loop to collect optimizer stats
    # For now, we'll just note what statistics we could collect
    print("\nAvailable statistics to collect (with modifications):")
    print("1. Gradient consistency over time")
    print("2. Curvature estimates distribution")
    print("3. Trust ratio evolution")
    print("4. Learning rate adaptation")
    print("5. Momentum adaptation")
    
    print("\nTo collect these statistics, modify train_epoch to:")
    print("1. Store optimizer.get_gradient_consistency() each iteration")
    print("2. Store optimizer.get_curvature_stats() each epoch")
    print("3. Store optimizer.get_trust_ratio() each iteration")


def main():
    """Main function to run MNIST comparison."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Compare optimizers
    results = compare_optimizers(num_epochs=5, lr=0.001, device=device)
    
    # Plot and save results
    summary = plot_results(results)
    
    # Analyze statistics
    analyze_optimizer_statistics(results)
    
    print("\n" + "=" * 60)
    print("MNIST Comparison Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()