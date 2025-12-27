#!/usr/bin/env python3
"""
Test script for ULTRON optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Import optimizers
from optimizers.ultron import ULTRON
from optimizers.amcas import AMCAS


def test_simple_function():
    """Test ULTRON on a simple quadratic function."""
    print("Testing ULTRON on simple quadratic function...")
    
    # Define a simple quadratic: f(x) = (x - 2)^2
    x = torch.tensor([0.0], requires_grad=True)
    
    # Create optimizers
    optimizers = {
        'ULTRON': ULTRON([x], lr=0.1),
        'AMCAS': AMCAS([x], lr=0.1),
        'Adam': torch.optim.Adam([x], lr=0.1),
        'SGD': torch.optim.SGD([x], lr=0.1),
    }
    
    results = {}
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}...")
        
        # Reset parameter
        x.data = torch.tensor([0.0])
        x.grad = None
        
        losses = []
        for i in range(100):
            optimizer.zero_grad()
            loss = (x - 2).pow(2)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if i % 20 == 0:
                print(f"  Iteration {i}: x = {x.item():.4f}, loss = {loss.item():.6f}")
        
        results[name] = {
            'losses': losses,
            'final_x': x.item(),
            'final_loss': losses[-1],
        }
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        ax.plot(data['losses'], label=name, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Simple Quadratic Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ultron_simple_test.png', dpi=150)
    plt.close()
    
    print("\nFinal results:")
    for name, data in results.items():
        print(f"{name:10} Final x: {data['final_x']:.6f}, Final loss: {data['final_loss']:.6e}")
    
    return results


def test_rosenbrock():
    """Test ULTRON on Rosenbrock function."""
    print("\n\nTesting ULTRON on Rosenbrock function...")
    
    # Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    x = torch.tensor([-1.5], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    
    optimizers = {
        'ULTRON': ULTRON([x, y], lr=0.001, clip_threshold=0.1),
        'AMCAS': AMCAS([x, y], lr=0.001),
        'Adam': torch.optim.Adam([x, y], lr=0.001),
        'SGD': torch.optim.SGD([x, y], lr=0.001),
    }
    
    results = {}
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}...")
        
        # Reset parameters
        x.data = torch.tensor([-1.5])
        y.data = torch.tensor([2.0])
        x.grad = None
        y.grad = None
        
        losses = []
        for i in range(1000):
            optimizer.zero_grad()
            loss = (1 - x)**2 + 100 * (y - x**2)**2
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if i % 200 == 0:
                print(f"  Iteration {i}: x = {x.item():.4f}, y = {y.item():.4f}, loss = {loss.item():.6f}")
        
        results[name] = {
            'losses': losses,
            'final_x': x.item(),
            'final_y': y.item(),
            'final_loss': losses[-1],
        }
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        ax.plot(data['losses'], label=name, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Rosenbrock Function Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ultron_rosenbrock_test.png', dpi=150)
    plt.close()
    
    print("\nFinal results:")
    for name, data in results.items():
        print(f"{name:10} Final (x,y): ({data['final_x']:.6f}, {data['final_y']:.6f}), Final loss: {data['final_loss']:.6e}")
    
    return results


def test_mnist_cnn():
    """Test ULTRON on a simple MNIST CNN."""
    print("\n\nTesting ULTRON on MNIST CNN...")
    
    # Simple CNN for MNIST
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # Create model and dummy data
    model = SimpleCNN()
    dummy_input = torch.randn(64, 1, 28, 28)
    dummy_target = torch.randint(0, 10, (64,))
    
    optimizers = {
        'ULTRON': ULTRON(model.parameters(), lr=0.001, weight_decay=1e-4),
        'AMCAS': AMCAS(model.parameters(), lr=0.001, weight_decay=1e-4),
        'Adam': torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4),
        'SGD': torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9),
    }
    
    results = {}
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}...")
        
        # Reset model weights
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        losses = []
        times = []
        
        start_time = time.time()
        for i in range(100):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = F.nll_loss(output, dummy_target)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            times.append(time.time() - start_time)
            
            if i % 20 == 0:
                print(f"  Iteration {i}: loss = {loss.item():.6f}")
        
        results[name] = {
            'losses': losses,
            'times': times,
            'final_loss': losses[-1],
            'total_time': times[-1],
        }
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss plot
    for name, data in results.items():
        ax1.plot(data['losses'], label=name, linewidth=2)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('MNIST CNN Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time plot
    for name, data in results.items():
        ax2.plot(data['times'], data['losses'], label=name, linewidth=2)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ultron_mnist_test.png', dpi=150)
    plt.close()
    
    print("\nFinal results:")
    for name, data in results.items():
        print(f"{name:10} Final loss: {data['final_loss']:.6f}, Total time: {data['total_time']:.3f}s")
    
    return results


def test_memory_efficiency():
    """Test memory efficiency of ULTRON vs other optimizers."""
    print("\n\nTesting memory efficiency...")
    
    # Create a larger model
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(1000, 2000),
                nn.ReLU(),
                nn.Linear(2000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 10),
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = LargeModel()
    dummy_input = torch.randn(128, 1000)
    dummy_target = torch.randint(0, 10, (128,))
    
    optimizers = {
        'ULTRON': ULTRON(model.parameters(), lr=0.001),
        'AMCAS': AMCAS(model.parameters(), lr=0.001),
        'Adam': torch.optim.Adam(model.parameters(), lr=0.001),
        'SGD+Momentum': torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
    }
    
    # Measure memory usage by checking state size
    memory_usage = {}
    for name, optimizer in optimizers.items():
        # Do one step to initialize optimizer state
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = F.cross_entropy(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        # Calculate state size
        state_size = 0
        for param in model.parameters():
            if param.grad is not None:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state_size += value.numel() * value.element_size()
        
        memory_usage[name] = state_size / (1024 * 1024)  # Convert to MB
        print(f"{name:15} State memory: {memory_usage[name]:.2f} MB")
    
    # Plot memory usage
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(memory_usage.keys())
    values = list(memory_usage.values())
    
    bars = ax.bar(names, values, color=['blue', 'green', 'orange', 'red'])
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Optimizer State Memory Usage')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.2f} MB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('ultron_memory_test.png', dpi=150)
    plt.close()
    
    return memory_usage


def test_gradient_clipping():
    """Test gradient clipping functionality."""
    print("\n\nTesting gradient clipping...")
    
    x = torch.tensor([0.0], requires_grad=True)
    
    # Test different clip thresholds
    clip_thresholds = [0.01, 0.1, 1.0, 10.0]
    
    results = {}
    for clip_thresh in clip_thresholds:
        print(f"\nTesting with clip_threshold = {clip_thresh}")
        
        optimizer = ULTRON([x], lr=0.1, clip_threshold=clip_thresh)
        
        # Reset parameter
        x.data = torch.tensor([0.0])
        x.grad = None
        
        update_magnitudes = []
        for i in range(50):
            optimizer.zero_grad()
            loss = (x - 5).pow(2)
            loss.backward()
            
            # Record gradient before update
            grad_mag = torch.abs(x.grad).item()
            
            optimizer.step()
            
            # Record update magnitude
            update_mag = torch.abs(optimizer.state[x]['momentum']).item()
            update_magnitudes.append(update_mag)
            
            if i % 10 == 0:
                print(f"  Iteration {i}: x = {x.item():.4f}, grad = {grad_mag:.4f}, update = {update_mag:.4f}")
        
        results[clip_thresh] = {
            'update_magnitudes': update_magnitudes,
            'max_update': max(update_magnitudes),
            'final_x': x.item(),
        }
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for clip_thresh, data in results.items():
        ax.plot(data['update_magnitudes'], label=f'clip={clip_thresh}', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Update Magnitude')
    ax.set_title('Effect of Clip Threshold on Update Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ultron_clipping_test.png', dpi=150)
    plt.close()
    
    print("\nMaximum update magnitudes:")
    for clip_thresh, data in results.items():
        print(f"Clip threshold {clip_thresh:5}: Max update = {data['max_update']:.4f}, Final x = {data['final_x']:.4f}")
    
    return results


def main():
    """Run all tests."""
    print("=" * 60)
    print("ULTRON Optimizer Test Suite")
    print("=" * 60)
    
    all_results = {}
    
    # Run tests
    all_results['simple'] = test_simple_function()
    all_results['rosenbrock'] = test_rosenbrock()
    all_results['mnist'] = test_mnist_cnn()
    all_results['memory'] = test_memory_efficiency()
    all_results['clipping'] = test_gradient_clipping()
    
    # Generate summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    # Compare final losses across tests
    optimizer_performance = {}
    for test_name, test_results in all_results.items():
        if test_name == 'memory':
            continue  # Skip memory test for loss comparison
        
        print(f"\n{test_name.upper()}:")
        for optimizer_name, data in test_results.items():
            if optimizer_name not in optimizer_performance:
                optimizer_performance[optimizer_name] = []
            
            if 'final_loss' in data:
                optimizer_performance[optimizer_name].append(data['final_loss'])
                print(f"  {optimizer_name:15} Final loss: {data['final_loss']:.6e}")
    
    # Calculate average performance
    print("\nAverage Performance (lower is better):")
    print("-" * 40)
    for optimizer_name, losses in optimizer_performance.items():
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"{optimizer_name:15} Average loss: {avg_loss:.6e}")
    
    print("\nTests completed successfully!")
    print("Plots saved as:")
    print("  - ultron_simple_test.png")
    print("  - ultron_rosenbrock_test.png")
    print("  - ultron_mnist_test.png")
    print("  - ultron_memory_test.png")
    print("  - ultron_clipping_test.png")


if __name__ == '__main__':
    main()
