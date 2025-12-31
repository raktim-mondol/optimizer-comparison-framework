"""
Simple NEXUS Optimizer Test

A minimal test to verify NEXUS optimizer works correctly.
This test only requires torch and numpy.
"""

import sys
import os

# Add optimizers to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from optimizers.nexus import NEXUS
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install required packages:")
    print("  pip install torch numpy")
    sys.exit(1)


def test_basic_functionality():
    """Test basic NEXUS functionality."""
    print("\n" + "=" * 80)
    print("Test 1: Basic Functionality")
    print("=" * 80)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Create NEXUS optimizer
    try:
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
        print("✓ NEXUS optimizer created successfully!")
    except Exception as e:
        print(f"✗ Failed to create NEXUS optimizer: {e}")
        return False
    
    # Generate dummy data
    X = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nRunning 10 training steps...")
    try:
        for step in range(10):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                print(f"  Step {step}: Loss = {loss.item():.6f}")
        print("✓ Training steps completed successfully!")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Test statistics methods
    print("\nTesting statistics methods...")
    try:
        lr_stats = optimizer.get_adaptive_lr_stats()
        print(f"  ✓ get_adaptive_lr_stats() works")
        
        dir_stats = optimizer.get_direction_consistency_stats()
        print(f"  ✓ get_direction_consistency_stats() works")
        
        curv_stats = optimizer.get_curvature_stats()
        print(f"  ✓ get_curvature_stats() works")
        
        mom_stats = optimizer.get_momentum_stats()
        print(f"  ✓ get_momentum_stats() works")
        
        mem_stats = optimizer.get_memory_usage()
        print(f"  ✓ get_memory_usage() works")
        print(f"    Total memory: {mem_stats['total_mb']:.2f} MB")
    except Exception as e:
        print(f"✗ Statistics methods failed: {e}")
        return False
    
    print("\n✓ Basic functionality test PASSED!")
    return True


def test_state_reset():
    """Test state reset functionality."""
    print("\n" + "=" * 80)
    print("Test 2: State Reset")
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
    print(f"Before reset - Adaptive LR Mean: {lr_stats_before['mean_adaptive_lr']:.6f}")
    
    # Reset state
    try:
        optimizer.reset_state()
        print("✓ State reset completed!")
    except Exception as e:
        print(f"✗ State reset failed: {e}")
        return False
    
    # Get statistics after reset
    lr_stats_after = optimizer.get_adaptive_lr_stats()
    print(f"After reset - Adaptive LR Mean: {lr_stats_after['mean_adaptive_lr']:.6f}")
    
    # Verify reset worked (adaptive LR should be close to 1.0)
    if abs(lr_stats_after['mean_adaptive_lr'] - 1.0) < 0.1:
        print("✓ State values properly reset!")
        print("\n✓ State reset test PASSED!")
        return True
    else:
        print("✗ State values not properly reset!")
        print("\n✗ State reset test FAILED!")
        return False


def test_gradient_handling():
    """Test gradient handling."""
    print("\n" + "=" * 80)
    print("Test 3: Gradient Handling")
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
        print("✓ Gradients are properly computed")
    else:
        print("✗ Some parameters are missing gradients")
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
    
    print("\n✓ Gradient handling test PASSED!")
    return True


def test_parameter_validation():
    """Test parameter validation."""
    print("\n" + "=" * 80)
    print("Test 4: Parameter Validation")
    print("=" * 80)
    
    model = nn.Linear(10, 10)
    
    # Test invalid learning rate
    try:
        optimizer = NEXUS(model.parameters(), lr=-1.0)
        print("✗ Should reject negative learning rate")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected negative learning rate: {e}")
    
    # Test invalid betas
    try:
        optimizer = NEXUS(model.parameters(), lr=1e-3, betas=(1.5, 0.99, 0.999))
        print("✗ Should reject beta > 1.0")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid beta: {e}")
    
    # Test invalid momentum scales
    try:
        optimizer = NEXUS(model.parameters(), lr=1e-3, momentum_scales=10)
        print("✗ Should reject too many momentum scales")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid momentum scales: {e}")
    
    # Test invalid state precision
    try:
        optimizer = NEXUS(model.parameters(), lr=1e-3, state_precision='invalid')
        print("✗ Should reject invalid state precision")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid state precision: {e}")
    
    print("\n✓ Parameter validation test PASSED!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("NEXUS Optimizer Simple Test Suite")
    print("=" * 80)
    print("\nTesting NEXUS: Neural EXploration with Unified Scaling")
    print("A novel optimizer combining meta-learning, multi-scale momentum,")
    print("and curvature-aware scaling for faster convergence.")
    
    results = {}
    
    # Run tests
    tests = [
        ('basic', test_basic_functionality),
        ('state_reset', test_state_reset),
        ('gradients', test_gradient_handling),
        ('validation', test_parameter_validation),
    ]
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"{test_name:<20} {status}")
    
    # Overall result
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED! ✓")
        print("\nNEXUS optimizer is working correctly!")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 80)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

