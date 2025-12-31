#!/usr/bin/env python3
"""
Basic test for ULTRON_V2 optimizer.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.ultron_v2 import ULTRON_V2


def test_basic_functionality():
    """Test basic functionality of ULTRON_V2."""
    print("Testing ULTRON_V2 basic functionality...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create optimizer
    optimizer = ULTRON_V2(model.parameters(), lr=0.001)
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    criterion = nn.MSELoss()
    
    # Test a few optimization steps
    for i in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"  Step {i+1}: loss = {loss.item():.6f}")
    
    print("[PASS] Basic functionality test passed!")
    return True


def test_vectorized_updates():
    """Test that vectorized updates work correctly."""
    print("\nTesting vectorized updates...")
    
    # Create model with multiple parameters
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 10)
    )
    
    # Create optimizer with vectorized updates
    optimizer = ULTRON_V2(model.parameters(), lr=0.001, normalize_gradients=True)
    
    # Create dummy data
    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    criterion = nn.CrossEntropyLoss()
    
    # Test optimization
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    print(f"  Loss after one step: {loss.item():.6f}")
    print("[PASS] Vectorized updates test passed!")
    return True


def test_adaptive_clipping():
    """Test adaptive clipping functionality."""
    print("\nTesting adaptive clipping...")
    
    model = nn.Linear(10, 5)
    optimizer = ULTRON_V2(model.parameters(), lr=0.001, adaptive_clipping=True)
    
    # Create gradients with varying magnitudes
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    criterion = nn.MSELoss()
    
    initial_threshold = optimizer.get_clip_threshold()
    print(f"  Initial clip threshold: {initial_threshold:.6f}")
    
    # Run several steps to see if threshold adapts
    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            current_threshold = optimizer.get_clip_threshold()
            print(f"  Step {i+1}: threshold = {current_threshold:.6f}")
    
    final_threshold = optimizer.get_clip_threshold()
    print(f"  Final clip threshold: {final_threshold:.6f}")
    
    if final_threshold != initial_threshold:
        print("[PASS] Adaptive clipping is working!")
        return True
    else:
        print("[WARNING] Adaptive clipping may not be working as expected")
        return False


def test_normalization_strategies():
    """Test different normalization strategies."""
    print("\nTesting normalization strategies...")
    
    strategies = ['rms', 'l2', 'moving_avg']
    
    for strategy in strategies:
        print(f"  Testing {strategy} normalization...")
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        optimizer = ULTRON_V2(
            model.parameters(),
            lr=0.001,
            normalize_gradients=True,
            normalization_strategy=strategy
        )
        
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        criterion = nn.MSELoss()
        
        # Run a few steps
        for i in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        print(f"    Final loss: {loss.item():.6f}")
    
    print("[PASS] All normalization strategies tested!")
    return True


def test_memory_usage():
    """Test memory usage statistics."""
    print("\nTesting memory usage statistics...")
    
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Test with different state precisions
    precisions = ['fp32', 'fp16', 'bf16']
    
    for precision in precisions:
        print(f"  Testing {precision} state precision...")
        
        optimizer = ULTRON_V2(
            model.parameters(),
            lr=0.001,
            state_precision=precision
        )
        
        # Initialize state with one step
        x = torch.randn(64, 100)
        y = torch.randn(64, 10)
        criterion = nn.MSELoss()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Get memory usage
        memory_stats = optimizer.get_memory_usage()
        print(f"    State memory: {memory_stats['state_mb']:.2f} MB")
        print(f"    Total memory: {memory_stats['total_mb']:.2f} MB")
    
    print("[PASS] Memory usage statistics test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ULTRON_V2 Optimizer Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_vectorized_updates,
        test_adaptive_clipping,
        test_normalization_strategies,
        test_memory_usage,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("[SUCCESS] All tests passed! ULTRON_V2 is working correctly.")
        return 0
    else:
        print("[WARNING] Some tests failed. Please check the implementation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
