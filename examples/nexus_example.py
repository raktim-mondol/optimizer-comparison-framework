"""
NEXUS Optimizer Example

This example demonstrates the usage of the NEXUS optimizer on a simple neural network.
NEXUS combines multiple advanced techniques for faster convergence and better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from optimizers import NEXUS, Adam, SGD, ULTRON_V2


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def generate_synthetic_data(num_samples=10000, input_size=784, num_classes=10):
    """Generate synthetic data for testing."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_model(model, optimizer, train_loader, num_epochs=10, device='cpu'):
    """Train a model with the given optimizer."""
    model.train()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
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
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return losses, accuracies


def compare_optimizers():
    """Compare NEXUS with other optimizers."""
    print("=" * 80)
    print("NEXUS Optimizer Comparison")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X_train, y_train = generate_synthetic_data(num_samples=5000)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimizers to compare
    optimizers_config = [
        {
            'name': 'NEXUS',
            'optimizer': lambda model: NEXUS(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.99, 0.999),
                meta_lr=1e-4,
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
        },
        {
            'name': 'Adam',
            'optimizer': lambda model: Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        },
        {
            'name': 'SGD with Momentum',
            'optimizer': lambda model: SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        },
        {
            'name': 'ULTRON_V2',
            'optimizer': lambda model: ULTRON_V2(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
                normalize_gradients=True,
                adaptive_clipping=True
            )
        }
    ]
    
    # Train with each optimizer
    results = {}
    for config in optimizers_config:
        print(f"\n{'=' * 80}")
        print(f"Training with {config['name']}")
        print(f"{'=' * 80}")
        
        # Create fresh model
        model = SimpleNet()
        optimizer = config['optimizer'](model)
        
        # Train
        losses, accuracies = train_model(
            model, optimizer, train_loader, num_epochs=15, device=device
        )
        
        results[config['name']] = {
            'losses': losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1],
            'final_loss': losses[-1]
        }
        
        # Print NEXUS-specific statistics
        if config['name'] == 'NEXUS':
            print("\nNEXUS Statistics:")
            print("-" * 40)
            lr_stats = optimizer.get_adaptive_lr_stats()
            print(f"Adaptive LR - Mean: {lr_stats['mean_adaptive_lr']:.6f}, "
                  f"Std: {lr_stats['std_adaptive_lr']:.6f}")
            
            dir_stats = optimizer.get_direction_consistency_stats()
            print(f"Direction Consistency - Mean: {dir_stats['mean_consistency']:.6f}, "
                  f"Std: {dir_stats['std_consistency']:.6f}")
            
            curv_stats = optimizer.get_curvature_stats()
            print(f"Curvature - Mean: {curv_stats['mean_curvature']:.6f}, "
                  f"Std: {curv_stats['std_curvature']:.6f}")
            
            mom_stats = optimizer.get_momentum_stats()
            print(f"Momentum Scale 0 - Mean: {mom_stats['scale_0']['mean']:.6f}, "
                  f"Std: {mom_stats['scale_0']['std']:.6f}")
            print(f"Momentum Scale 1 - Mean: {mom_stats['scale_1']['mean']:.6f}, "
                  f"Std: {mom_stats['scale_1']['std']:.6f}")
            print(f"Momentum Scale 2 - Mean: {mom_stats['scale_2']['mean']:.6f}, "
                  f"Std: {mom_stats['scale_2']['std']:.6f}")
            
            mem_stats = optimizer.get_memory_usage()
            print(f"Memory Usage - Total: {mem_stats['total_mb']:.2f} MB, "
                  f"State: {mem_stats['state_mb']:.2f} MB")
    
    # Plot results
    plot_comparison(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for name, result in results.items():
        print(f"{name:20s} - Final Accuracy: {result['final_accuracy']:.2f}%, "
              f"Final Loss: {result['final_loss']:.4f}")
    
    return results


def plot_comparison(results):
    """Plot comparison of optimizers."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    for name, result in results.items():
        axes[0].plot(result['losses'], label=name, marker='o', markersize=4)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    for name, result in results.items():
        axes[1].plot(result['accuracies'], label=name, marker='o', markersize=4)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nexus_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'nexus_comparison.png'")
    plt.close()


def test_nexus_features():
    """Test various features of NEXUS optimizer."""
    print("\n" + "=" * 80)
    print("Testing NEXUS Features")
    print("=" * 80)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Create optimizer with various features enabled
    optimizer = NEXUS(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.99, 0.999),
        meta_lr=1e-4,
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
    
    # Generate some dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train for a few epochs
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        total_loss = 0.0
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    # Test state reset
    print("\nTesting state reset...")
    optimizer.reset_state()
    print("State reset successfully!")
    
    # Test statistics
    print("\nOptimizer Statistics:")
    print("-" * 40)
    lr_stats = optimizer.get_adaptive_lr_stats()
    print(f"Adaptive LR: Mean={lr_stats['mean_adaptive_lr']:.6f}, "
          f"Std={lr_stats['std_adaptive_lr']:.6f}")
    
    dir_stats = optimizer.get_direction_consistency_stats()
    print(f"Direction Consistency: Mean={dir_stats['mean_consistency']:.6f}, "
          f"Std={dir_stats['std_consistency']:.6f}")
    
    curv_stats = optimizer.get_curvature_stats()
    print(f"Curvature: Mean={curv_stats['mean_curvature']:.6f}, "
          f"Std={curv_stats['std_curvature']:.6f}")
    
    mem_stats = optimizer.get_memory_usage()
    print(f"Memory: Total={mem_stats['total_mb']:.2f} MB, "
          f"State={mem_stats['state_mb']:.2f} MB")


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("NEXUS Optimizer - Novel Neural EXploration with Unified Scaling")
    print("=" * 80)
    print("\nNEXUS combines multiple advanced techniques:")
    print("1. Meta-Learning Adaptive Learning Rates (MLALR)")
    print("2. Multi-Scale Momentum (MSM)")
    print("3. Gradient Direction Consistency (GDC)")
    print("4. Layer-wise Adaptation (LWA)")
    print("5. Adaptive Step Size with Lookahead (ASSL)")
    print("6. Dynamic Weight Decay (DWD)")
    print("7. Curvature-Aware Scaling (CAS)")
    print("8. Gradient Noise Injection (GNI)")
    
    # Test features
    test_nexus_features()
    
    # Compare optimizers
    results = compare_optimizers()
    
    print("\n" + "=" * 80)
    print("NEXUS Optimizer Example Completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

