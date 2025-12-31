"""
Simple NEXUS MNIST Test - Quick version with fewer epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from optimizers import NEXUS
from models.cnn_mnist import SimpleCNN


def get_data_loaders(batch_size=128):
    """Get MNIST train and test data loaders."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def main():
    print("=" * 80)
    print("NEXUS Optimizer on MNIST - Quick Test")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Get data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=128)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = SimpleCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create NEXUS optimizer
    print("\nInitializing NEXUS optimizer...")
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
    print("NEXUS optimizer created!")
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
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
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%')
        
        # Print NEXUS stats every 5 epochs
        if (epoch + 1) % 5 == 0:
            lr_stats = optimizer.get_adaptive_lr_stats()
            dir_stats = optimizer.get_direction_consistency_stats()
            curv_stats = optimizer.get_curvature_stats()
            print(f'  NEXUS - LR: {lr_stats["mean_adaptive_lr"]:.4f}, '
                  f'DirCons: {dir_stats["mean_consistency"]:.4f}, '
                  f'Curv: {curv_stats["mean_curvature"]:.4f}')
    
    total_time = time.time() - start_time
    
    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f'Total training time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)')
    print(f'Final test accuracy: {test_acc:.2f}%')
    
    # NEXUS final statistics
    print("\nNEXUS Optimizer Final Statistics:")
    print("-" * 80)
    lr_stats = optimizer.get_adaptive_lr_stats()
    print(f'Adaptive Learning Rate:')
    print(f'  Mean: {lr_stats["mean_adaptive_lr"]:.6f}')
    print(f'  Std:  {lr_stats["std_adaptive_lr"]:.6f}')
    print(f'  Min:  {lr_stats["min_adaptive_lr"]:.6f}')
    print(f'  Max:  {lr_stats["max_adaptive_lr"]:.6f}')
    
    dir_stats = optimizer.get_direction_consistency_stats()
    print(f'\nGradient Direction Consistency:')
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
    print(f'  Scale 1 (medium-term): Mean={mom_stats["scale_1"]["mean"]:.6f}, '
          f'Std={mom_stats["scale_1"]["std"]:.6f}')
    print(f'  Scale 2 (long-term):  Mean={mom_stats["scale_2"]["mean"]:.6f}, '
          f'Std={mom_stats["scale_2"]["std"]:.6f}')
    
    mem_stats = optimizer.get_memory_usage()
    print(f'\nMemory Usage:')
    print(f'  Total: {mem_stats["total_mb"]:.2f} MB')
    print(f'  State: {mem_stats["state_mb"]:.2f} MB')
    print(f'  Parameters: {mem_stats["param_mb"]:.2f} MB')
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

