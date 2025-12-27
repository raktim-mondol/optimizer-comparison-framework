"""
CIFAR-10 Classification Example with AMCAS Optimizer

This example demonstrates training a ResNet-18 on CIFAR-10 dataset
using the AMCAS optimizer and comparing it with other optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
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


class CIFAR10ResNet(nn.Module):
    """ResNet-18 adapted for CIFAR-10."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Use a pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=False)
        
        # Modify first conv layer for CIFAR-10 (3 channels instead of RGB images)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the original fc layer and add a new one for CIFAR-10
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


def load_cifar10_data(batch_size=128):
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, scheduler=None):
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
        
        if scheduler is not None:
            scheduler.step()
        
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
                num_epochs=20, lr=0.1, device='cpu', use_scheduler=True):
    """Train model with specified optimizer."""
    print(f"\nTraining with {optimizer_name}...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
    
    # Load data
    train_loader, test_loader = load_cifar10_data(batch_size=128)
    
    # Create model
    model = CIFAR10ResNet().to(device)
    
    # Create optimizer
    optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_params)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_loader)
        )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'time_per_epoch': [],
        'learning_rates': []
    }
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch, scheduler
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
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f'Epoch {epoch:2d}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:6.2f}%, '
              f'Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return model, history


def compare_optimizers(num_epochs=20, lr=0.1, device='cpu'):
    """Compare different optimizers on CIFAR-10."""
    print("=" * 60)
    print("CIFAR-10 Optimizer Comparison")
    print("=" * 60)
    
    # Define optimizers to compare
    optimizers = {
        'AMCAS': (AMCAS, {'betas': (0.9, 0.999), 'gamma': 0.1, 'lambda_consistency': 0.01}),
        'Adam': (torch.optim.Adam, {'betas': (0.9, 0.999)}),
        'AdamW': (torch.optim.AdamW, {'betas': (0.9, 0.999), 'weight_decay': 0.01}),
        'SGD': (torch.optim.SGD, {'momentum': 0, 'weight_decay': 5e-4}),
        'SGD+Momentum': (torch.optim.SGD, {'momentum': 0.9, 'weight_decay': 5e-4}),
        'RMSprop': (torch.optim.RMSprop, {'weight_decay': 5e-4}),
    }
    
    results = {}
    
    for name, (optimizer_class, params) in optimizers.items():
        model, history = train_model(
            name, optimizer_class, params, num_epochs, lr, device, use_scheduler=True
        )
        results[name] = history
        
        # Save model checkpoint
        torch.save(model.state_dict(), f'cifar10_{name.lower()}_model.pth')
        
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results


def plot_results(results, output_dir='cifar10_results'):
    """Plot comparison results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot training loss
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['train_loss'], label=name, marker='o', markersize=2, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Training Loss Convergence')
    
    # Plot test accuracy
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['test_acc'], label=name, marker='o', markersize=2, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Test Accuracy')
    
    # Plot training accuracy
    ax = axes[0, 2]
    for name, history in results.items():
        ax.plot(history['train_acc'], label=name, marker='o', markersize=2, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Training Accuracy')
    
    # Plot time per epoch
    ax = axes[1, 0]
    optimizer_names = list(results.keys())
    avg_times = [np.mean(history['time_per_epoch']) for history in results.values()]
    ax.bar(optimizer_names, avg_times)
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Average Time per Epoch (s)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Computational Efficiency')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot learning rate schedule
    ax = axes[1, 1]
    for name, history in results.items():
        if 'learning_rates' in history:
            ax.plot(history['learning_rates'], label=name, marker='o', markersize=2, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    
    # Plot generalization gap
    ax = axes[1, 2]
    for name, history in results.items():
        if len(history['train_acc']) > 0 and len(history['test_acc']) > 0:
            gen_gap = [train - test for train, test in zip(history['train_acc'], history['test_acc'])]
            ax.plot(gen_gap, label=name, marker='o', markersize=2, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generalization Gap (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Generalization Gap (Train - Test)')
    
    plt.tight_layout()
    plt.savefig(output_path / 'cifar10_comparison.png', dpi=150)
    plt.close()
    
    # Create summary table
    print("\n" + "=" * 60)
    print("Final Results Summary")
    print("=" * 60)
    
    summary_data = []
    for name, history in results.items():
        final_test_acc = history['test_acc'][-1] if history['test_acc'] else 0
        final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        final_test_loss = history['test_loss'][-1] if history['test_loss'] else 0
        avg_time = np.mean(history['time_per_epoch']) if history['time_per_epoch'] else 0
        peak_train_acc = max(history['train_acc']) if history['train_acc'] else 0
        peak_test_acc = max(history['test_acc']) if history['test_acc'] else 0
        
        summary_data.append({
            'optimizer': name,
            'final_test_accuracy': final_test_acc,
            'final_train_accuracy': final_train_acc,
            'peak_test_accuracy': peak_test_acc,
            'peak_train_accuracy': peak_train_acc,
            'final_test_loss': final_test_loss,
            'avg_time_per_epoch': avg_time,
            'generalization_gap': final_train_acc - final_test_acc,
            'peak_generalization_gap': peak_train_acc - peak_test_acc
        })
    
    # Sort by test accuracy
    summary_data.sort(key=lambda x: x['final_test_accuracy'], reverse=True)
    
    print("\nRanked by Final Test Accuracy:")
    print("-" * 90)
    print(f"{'Optimizer':<15} {'Test Acc':<10} {'Peak Test':<10} {'Train Acc':<10} {'Gen Gap':<10} {'Time/Epoch':<12}")
    print("-" * 90)
    
    for data in summary_data:
        print(f"{data['optimizer']:<15} "
              f"{data['final_test_accuracy']:>8.2f}% "
              f"{data['peak_test_accuracy']:>8.2f}% "
              f"{data['final_train_accuracy']:>8.2f}% "
              f"{data['generalization_gap']:>8.2f}% "
              f"{data['avg_time_per_epoch']:>10.2f}s")
    
    # Save results
    with open(output_path / 'cifar10_results.json', 'w') as f:
        # Convert any numpy arrays to lists
        serializable_results = {}
        for name, history in results.items():
            serializable_results[name] = {}
            for key, value in history.items():
                if isinstance(value, list) and value and isinstance(value[0], (np.generic, np.ndarray)):
                    serializable_results[name][key] = [float(v) if isinstance(v, np.generic) else v for v in value]
                elif isinstance(value, (np.generic, np.ndarray)):
                    serializable_results[name][key] = float(value)
                else:
                    serializable_results[name][key] = value
        json.dump(serializable_results, f, indent=2)
    
    with open(output_path / 'cifar10_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}/")
    print(f"Best optimizer (final test accuracy): {summary_data[0]['optimizer']}")
    print(f"Best optimizer (peak test accuracy): {max(summary_data, key=lambda x: x['peak_test_accuracy'])['optimizer']}")
    
    return summary_data


def analyze_convergence_speed(results):
    """Analyze convergence speed of different optimizers."""
    print("\n" + "=" * 60)
    print("Convergence Speed Analysis")
    print("=" * 60)
    
    convergence_thresholds = [70, 75, 80, 85]
    
    print("\nEpochs to reach accuracy thresholds:")
    print("-" * 60)
    print(f"{'Optimizer':<15} " + "".join([f"{thresh}% Acc".rjust(10) for thresh in convergence_thresholds]))
    print("-" * 60)
    
    for name, history in results.items():
        test_acc = history['test_acc']
        epochs_to_threshold = []
        
        for threshold in convergence_thresholds:
            epoch_reached = None
            for epoch, acc in enumerate(test_acc, 1):
                if acc >= threshold:
                    epoch_reached = epoch
                    break
            epochs_to_threshold.append(epoch_reached or len(test_acc))
        
        print(f"{name:<15} " + "".join([f"{epoch:>10}" for epoch in epochs_to_threshold]))
    
    # Find fastest optimizer for each threshold
    print("\nFastest optimizer for each threshold:")
    print("-" * 60)
    for threshold in convergence_thresholds:
        fastest = None
        fastest_epoch = float('inf')
        
        for name, history in results.items():
            test_acc = history['test_acc']
            for epoch, acc in enumerate(test_acc, 1):
                if acc >= threshold:
                    if epoch < fastest_epoch:
                        fastest_epoch = epoch
                        fastest = name
                    break
        
        if fastest:
            print(f"{threshold}% accuracy: {fastest} (epoch {fastest_epoch})")
        else:
            print(f"{threshold}% accuracy: None reached threshold")


def main():
    """Main function to run CIFAR-10 comparison."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Compare optimizers
    results = compare_optimizers(num_epochs=10, lr=0.1, device=device)
    
    # Plot and save results
    summary = plot_results(results)
    
    # Analyze convergence speed
    analyze_convergence_speed(results)
    
    print("\n" + "=" * 60)
    print("CIFAR-10 Comparison Complete!")
    print("=" * 60)
    
    # Recommendations based on results
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    best_final = summary[0]['optimizer']
    best_peak = max(summary, key=lambda x: x['peak_test_accuracy'])['optimizer']
    fastest_70 = None
    fastest_80 = None
    
    # Simple analysis for demonstration
    print(f"\n1. For best final accuracy: Use {best_final}")
    print(f"2. For best peak accuracy: Use {best_peak}")
    print("3. For fastest convergence: Check convergence speed analysis above")
    print("4. For best generalization: Choose optimizer with smallest generalization gap")
    print("5. For computational efficiency: Choose optimizer with lowest time per epoch")


if __name__ == '__main__':
    main()