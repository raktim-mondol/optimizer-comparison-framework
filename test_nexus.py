"""
Test NEXUS Optimizer

This script tests the NEXUS optimizer on a simple neural network task
to verify its functionality and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from optimizers import NEXUS


# Define a simple neural network for testing
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


def generate_synthetic_data(num_samples=5000, input_size=100, num_classes=10):
    """Generate synthetic data for testing."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def test_nexus_basic():
    """Test basic functionality of NEXUS optimizer."""
    print("=" * 80)
    print("Test 1: Basic NEXUS Functionality")
    print("=" * 80)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
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
    
    # Generate dummy data
    X = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    
    # Training loop
    criterion = nn.CrossEntropyLoss()
    
    print("\nRunning 10 training steps...")
    for step in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")
    
    print("\n✓ Basic functionality test passed!")
    
    # Test statistics
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
    
    return True


def test_nexus_state_reset():
    """Test state reset functionality."""
    print("\n" + "=" * 80)
    print("Test 2: State Reset Functionality")
    print("=" * 80)
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    optimizer = NEXUS(model.parameters(), lr=1e-3)
    
    # Train for a few steps
    X = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(5):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Get statistics before reset
    lr_stats_before = optimizer.get_adaptive_lr_stats()
    print(f"\nBefore reset - Adaptive LR Mean: {lr_stats_before['mean_adaptive_lr']:.6f}")
    
    # Reset state
    optimizer.reset_state()
    print("State reset completed...")
    
    # Get statistics after reset
    lr_stats_after = optimizer.get_adaptive_lr_stats()
    print(f"After reset - Adaptive LR Mean: {lr_stats_after['mean_adaptive_lr']:.6f}")
    
    # Verify reset worked
    if abs(lr_stats_after['mean_adaptive_lr'] - 1.0) < 0.01:
        print("\n✓ State reset test passed!")
        return True
    else:
        print("\n✗ State reset test failed!")
        return False


def test_nexus_comparison():
    """Compare NEXUS with Adam and SGD."""
    print("\n" + "=" * 80)
    print("Test 3: NEXUS vs Adam vs SGD Comparison")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X_train, y_train = generate_synthetic_data(num_samples=2000, input_size=100, num_classes=10)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimizers to test
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
            'optimizer': lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        },
        {
            'name': 'SGD',
            'optimizer': lambda model: torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        }
    ]
    
    # Train with each optimizer
    results = {}
    for config in optimizers_config:
        print(f"\n{'=' * 80}")
        print(f"Training with {config['name']}")
        print(f"{'=' * 80}")
        
        # Create fresh model
        model = TestNet(input_size=100, hidden_size=64, num_classes=10)
        model.to(device)
        optimizer = config['optimizer'](model)
        
        # Train
        criterion = nn.CrossEntropyLoss()
        num_epochs = 10
        losses = []
        accuracies = []
        
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
        
        results[config['name']] = {
            'losses': losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1],
            'final_loss': losses[-1],
            'training_time': training_time
        }
        
        # Print NEXUS-specific statistics
        if config['name'] == 'NEXUS':
            print("\nNEXUS Final Statistics:")
            print("-" * 40)
            lr_stats = optimizer.get_adaptive_lr_stats()
            print(f"Adaptive LR - Mean: {lr_stats['mean_adaptive_lr']:.6f}")
            
            dir_stats = optimizer.get_direction_consistency_stats()
            print(f"Direction Consistency - Mean: {dir_stats['mean_consistency']:.6f}")
            
            curv_stats = optimizer.get_curvature_stats()
            print(f"Curvature - Mean: {curv_stats['mean_curvature']:.6f}")
            
            mem_stats = optimizer.get_memory_usage()
            print(f"Memory Usage - Total: {mem_stats['total_mb']:.2f} MB")
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(f"{'Optimizer':<15} {'Final Acc':<12} {'Final Loss':<12} {'Time (s)':<12}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<15} {result['final_accuracy']:<12.2f} "
              f"{result['final_loss']:<12.4f} {result['training_time']:<12.2f}")
    
    # Determine best performer
    best_acc = max(results.items(), key=lambda x: x[1]['final_accuracy'])
    best_loss = min(results.items(), key=lambda x: x[1]['final_loss'])
    
    print("\n" + "-" * 60)
    print(f"Best Accuracy: {best_acc[0]} ({best_acc[1]['final_accuracy']:.2f}%)")
    print(f"Best Loss: {best_loss[0]} ({best_loss[1]['final_loss']:.4f})")
    
    return results


def test_nexus_convergence():
    """Test NEXUS convergence on a simple optimization problem."""
    print("\n" + "=" * 80)
    print("Test 4: NEXUS Convergence Test")
    print("=" * 80)
    
    # Create a simple quadratic optimization problem
    # Minimize f(x) = 0.5 * x^T A x - b^T x
    torch.manual_seed(42)
    n = 50
    A = torch.randn(n, n)
    A = A @ A.T + torch.eye(n) * 0.1  # Make it positive definite
    b = torch.randn(n)
    
    # Initial point
    x = torch.randn(n, requires_grad=True)
    
    # Create NEXUS optimizer
    optimizer = NEXUS([x], lr=1e-2, betas=(0.9, 0.99, 0.999))
    
    # Optimize
    losses = []
    print("\nOptimizing...")
    for step in range(100):
        optimizer.zero_grad()
        loss = 0.5 * x @ A @ x - b @ x
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")
    
    # Check convergence
    final_loss = losses[-1]
    initial_loss = losses[0]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nInitial Loss: {initial_loss:.6f}")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Loss Reduction: {reduction:.2f}%")
    
    if reduction > 90:
        print("\n✓ Convergence test passed!")
        return True
    else:
        print("\n✗ Convergence test failed!")
        return False


def test_nexus_gradients():
    """Test that NEXUS handles gradients correctly."""
    print("\n" + "=" * 80)
    print("Test 5: Gradient Handling Test")
    print("=" * 80)
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    optimizer = NEXUS(model.parameters(), lr=1e-3)
    
    # Generate data
    X = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Check gradients exist after backward
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    
    has_gradients = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    if has_gradients:
        print("\n✓ Gradients are properly computed")
    else:
        print("\n✗ Some parameters are missing gradients")
        return False
    
    # Check gradients are not all zeros
    grad_norms = [torch.norm(p.grad).item() for p in model.parameters() if p.grad is not None]
    max_grad_norm = max(grad_norms)
    
    if max_grad_norm > 1e-6:
        print(f"✓ Gradients have non-zero values (max norm: {max_grad_norm:.6f})")
    else:
        print(f"✗ Gradients are too small (max norm: {max_grad_norm:.6f})")
        return False
    
    # Check parameters are updated
    params_before = [p.clone() for p in model.parameters()]
    optimizer.step()
    params_after = list(model.parameters())
    
    params_changed = any(not torch.allclose(before, after) 
                       for before, after in zip(params_before, params_after))
    
    if params_changed:
        print("✓ Parameters are updated after optimizer.step()")
    else:
        print("✗ Parameters are not updated")
        return False
    
    print("\n✓ Gradient handling test passed!")
    return True


def main():
    """Run all NEXUS tests."""
    print("\n" + "=" * 80)
    print("NEXUS Optimizer Test Suite")
    print("=" * 80)
    print("\nTesting NEXUS: Neural EXploration with Unified Scaling")
    print("A novel optimizer combining meta-learning, multi-scale momentum,")
    print("and curvature-aware scaling for faster convergence.")
    
    results = {}
    
    # Run tests
    try:
        results['basic'] = test_nexus_basic()
    except Exception as e:
        print(f"\n✗ Basic test failed with error: {e}")
        results['basic'] = False
    
    try:
        results['state_reset'] = test_nexus_state_reset()
    except Exception as e:
        print(f"\n✗ State reset test failed with error: {e}")
        results['state_reset'] = False
    
    try:
        results['convergence'] = test_nexus_convergence()
    except Exception as e:
        print(f"\n✗ Convergence test failed with error: {e}")
        results['convergence'] = False
    
    try:
        results['gradients'] = test_nexus_gradients()
    except Exception as e:
        print(f"\n✗ Gradient test failed with error: {e}")
        results['gradients'] = False
    
    try:
        results['comparison'] = test_nexus_comparison()
    except Exception as e:
        print(f"\n✗ Comparison test failed with error: {e}")
        results['comparison'] = None
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⚠ COMPLETED"
        print(f"{test_name:<20} {status}")
    
    # Overall result
    all_passed = all(r is True for r in results.values() if r is not None)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 80)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

