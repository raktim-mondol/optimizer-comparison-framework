#!/usr/bin/env python3
"""
Minimal test script for ULTRON optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import optimizers
from optimizers.ultron import ULTRON
from optimizers.amcas import AMCAS


def test_basic_functionality():
    """Test basic functionality of ULTRON."""
    print("Testing basic functionality of ULTRON...")
    
    # Test 1: Simple quadratic optimization
    print("\n1. Simple quadratic optimization:")
    x = torch.tensor([0.0], requires_grad=True)
    optimizer = ULTRON([x], lr=0.1)
    
    for i in range(10):
        optimizer.zero_grad()
        loss = (x - 2).pow(2)
        loss.backward()
        optimizer.step()
        print(f"  Iteration {i}: x = {x.item():.4f}, loss = {loss.item():.6f}")
    
    # Test 2: Multiple parameters
    print("\n2. Multiple parameters optimization:")
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([-1.0], requires_grad=True)
    optimizer = ULTRON([x, y], lr=0.01)
    
    for i in range(5):
        optimizer.zero_grad()
        loss = x**2 + y**2
        loss.backward()
        optimizer.step()
        print(f"  Iteration {i}: x = {x.item():.4f}, y = {y.item():.4f}, loss = {loss.item():.6f}")
    
    # Test 3: Weight decay
    print("\n3. Weight decay test:")
    x = torch.tensor([1.0], requires_grad=True)
    optimizer = ULTRON([x], lr=0.1, weight_decay=0.1)
    
    for i in range(5):
        optimizer.zero_grad()
        loss = x**2
        loss.backward()
        optimizer.step()
        print(f"  Iteration {i}: x = {x.item():.4f}, loss = {loss.item():.6f}")
    
    # Test 4: Gradient clipping
    print("\n4. Gradient clipping test:")
    x = torch.tensor([0.0], requires_grad=True)
    optimizer = ULTRON([x], lr=0.1, clip_threshold=0.5)
    
    for i in range(5):
        optimizer.zero_grad()
        loss = (x - 10).pow(2)  # Large gradient
        loss.backward()
        optimizer.step()
        
        # Check momentum magnitude
        momentum = optimizer.state[x]['momentum'].item()
        print(f"  Iteration {i}: x = {x.item():.4f}, momentum = {momentum:.4f}, loss = {loss.item():.6f}")
    
    return True


def test_comparison():
    """Compare ULTRON with other optimizers."""
    print("\n\nComparing ULTRON with other optimizers...")
    
    # Rosenbrock function
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    # Test each optimizer
    optimizers = {
        'ULTRON': lambda params: ULTRON(params, lr=0.001, clip_threshold=0.1),
        'AMCAS': lambda params: AMCAS(params, lr=0.001),
        'Adam': lambda params: torch.optim.Adam(params, lr=0.001),
        'SGD': lambda params: torch.optim.SGD(params, lr=0.001),
    }
    
    results = {}
    for name, optimizer_fn in optimizers.items():
        print(f"\nTesting {name}...")
        
        x = torch.tensor([-1.5], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        optimizer = optimizer_fn([x, y])
        
        losses = []
        for i in range(100):
            optimizer.zero_grad()
            loss = rosenbrock(x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if i % 25 == 0:
                print(f"  Iteration {i}: loss = {loss.item():.6f}")
        
        results[name] = {
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'final_x': x.item(),
            'final_y': y.item(),
        }
    
    # Print comparison
    print("\n\nOptimizer Comparison (Rosenbrock function):")
    print("-" * 60)
    print(f"{'Optimizer':<10} {'Final Loss':<15} {'Min Loss':<15} {'Final (x,y)':<20}")
    print("-" * 60)
    
    for name, data in results.items():
        print(f"{name:<10} {data['final_loss']:<15.6e} {data['min_loss']:<15.6e} ({data['final_x']:.4f}, {data['final_y']:.4f})")
    
    return results


def test_memory():
    """Test memory efficiency."""
    print("\n\nTesting memory efficiency...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    
    # Test different optimizers
    optimizers = {
        'ULTRON': ULTRON(model.parameters(), lr=0.001),
        'AMCAS': AMCAS(model.parameters(), lr=0.001),
        'Adam': torch.optim.Adam(model.parameters(), lr=0.001),
        'SGD+Momentum': torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
    }
    
    # Create dummy data
    dummy_input = torch.randn(32, 100)
    dummy_target = torch.randint(0, 10, (32,))
    
    memory_stats = {}
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}...")
        
        # Reset model
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # Do one training step
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = F.cross_entropy(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        # Count state parameters
        state_size = 0
        for param in model.parameters():
            if param.grad is not None:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state_size += value.numel()
        
        memory_stats[name] = state_size
        print(f"  State parameters: {state_size:,}")
    
    # Print memory comparison
    print("\n\nMemory Usage Comparison:")
    print("-" * 40)
    print(f"{'Optimizer':<15} {'State Parameters':<20}")
    print("-" * 40)
    
    for name, size in memory_stats.items():
        print(f"{name:<15} {size:<20,}")
    
    # Calculate memory reduction
    baseline = memory_stats.get('Adam', 1)
    for name, size in memory_stats.items():
        if name != 'Adam':
            reduction = (1 - size / baseline) * 100
            print(f"{name} uses {reduction:.1f}% less memory than Adam")
    
    return memory_stats


def main():
    """Run all tests."""
    print("=" * 60)
    print("ULTRON Optimizer Minimal Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        test_basic_functionality()
        results = test_comparison()
        memory_stats = test_memory()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
        # Find best optimizer
        best_optimizer = min(results.items(), key=lambda x: x[1]['min_loss'])[0]
        print(f"\nBest optimizer for Rosenbrock: {best_optimizer}")
        
        # Check if ULTRON is competitive
        ultron_loss = results['ULTRON']['min_loss']
        best_loss = results[best_optimizer]['min_loss']
        ratio = ultron_loss / best_loss if best_loss > 0 else 1.0
        
        if ratio < 1.5:  # Within 50% of best
            print(f"ULTRON is competitive! (within {ratio:.1f}x of best)")
        else:
            print(f"ULTRON needs improvement ({(ratio-1)*100:.0f}% worse than best)")
        
        return True
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
