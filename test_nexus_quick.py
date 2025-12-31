"""
Quick NEXUS Test - Minimal comparison with fewer epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
from optimizers import NEXUS


# Define a simple neural network
class TestNet(nn.Module):
    def __init__(self, input_size=100, hidden_size=64, num_classes=10):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def generate_synthetic_data(num_samples=1000, input_size=100, num_classes=10):
    """Generate synthetic data for testing."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def main():
    print("=" * 80)
    print("NEXUS Optimizer Quick Test")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X_train, y_train = generate_synthetic_data(num_samples=1000, input_size=100, num_classes=10)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = TestNet(input_size=100, hidden_size=64, num_classes=10)
    model.to(device)
    
    # Create NEXUS optimizer
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
    
    # Train
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5
    losses = []
    accuracies = []
    
    print("\nTraining...")
    start_time = time.time()
    
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
    
    training_time = time.time() - start_time
    
    # Print NEXUS statistics
    print("\n" + "=" * 80)
    print("NEXUS Final Statistics:")
    print("=" * 80)
    
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
    
    print(f"\nTraining Time: {training_time:.2f} seconds")
    print(f"Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"Final Loss: {losses[-1]:.4f}")
    
    print("\n" + "=" * 80)
    print("NEXUS Optimizer Test Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

