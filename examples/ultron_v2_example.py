"""
ULTRON_V2 Optimizer Usage Examples

This file demonstrates various usage patterns for the ULTRON_V2 optimizer,
including basic usage, advanced features, and best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from optimizers.ultron_v2 import ULTRON_V2


def example_1_basic_usage():
    """
    Example 1: Basic Usage of ULTRON_V2
    
    Demonstrates the simplest way to use ULTRON_V2 with default parameters.
    """
    print("=" * 60)
    print("Example 1: Basic Usage of ULTRON_V2")
    print("=" * 60)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create ULTRON_V2 optimizer with default parameters
    optimizer = ULTRON_V2(model.parameters(), lr=0.001)
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nTraining for 10 epochs...")
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print("\n✓ Basic usage example completed!")
    print("\nKey Points:")
    print("- ULTRON_V2 uses vectorized updates for efficiency")
    print("- Default parameters work well for most tasks")
    print("- Learning rate of 0.001 is a good starting point")


def example_2_normalization_strategies():
    """
    Example 2: Different Normalization Strategies
    
    Demonstrates the three normalization strategies available in ULTRON_V2:
    - RMS normalization (default, similar to Adam)
    - L2 normalization (simpler, faster)
    - Moving average normalization (adaptive)
    """
    print("\n" + "=" * 60)
    print("Example 2: Different Normalization Strategies")
    print("=" * 60)
    
    # Create identical models for fair comparison
    strategies = ['rms', 'l2', 'moving_avg']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.upper()} normalization...")
        
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Create optimizer with specific normalization strategy
        optimizer = ULTRON_V2(
            model.parameters(),
            lr=0.001,
            normalize_gradients=True,
            normalization_strategy=strategy
        )
        
        # Create dummy data
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        criterion = nn.CrossEntropyLoss()
        
        # Train for 10 epochs
        losses = []
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results[strategy] = losses[-1]
        print(f"  Final loss: {losses[-1]:.4f}")
    
    print("\n" + "-" * 60)
    print("Results Summary:")
    print("-" * 60)
    for strategy, final_loss in results.items():
        print(f"  {strategy.upper():<10} {final_loss:.4f}")
    
    print("\nRecommendations:")
    print("- RMS: Best for most tasks (default)")
    print("- L2: Faster, good for simple problems")
    print("- Moving Average: Adaptive, good for non-stationary data")


def example_3_adaptive_clipping():
    """
    Example 3: Adaptive Clipping
    
    Demonstrates ULTRON_V2's adaptive clipping feature,
    which automatically adjusts the clip threshold based on gradient statistics.
    """
    print("\n" + "=" * 60)
    print("Example 3: Adaptive Clipping")
    print("=" * 60)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Compare adaptive vs fixed clipping
    print("\nComparing adaptive vs fixed clipping...")
    
    # Test with adaptive clipping
    optimizer_adaptive = ULTRON_V2(
        model.parameters(),
        lr=0.001,
        adaptive_clipping=True,
        clip_alpha=0.99,
        clip_percentile=95.0
    )
    
    # Test with fixed clipping
    optimizer_fixed = ULTRON_V2(
        model.parameters(),
        lr=0.001,
        adaptive_clipping=False,
        clip_threshold=1.0
    )
    
    # Create dummy data with varying gradient magnitudes
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Train both for 10 epochs
    adaptive_losses = []
    fixed_losses = []
    
    for epoch in range(10):
        # Adaptive clipping
        optimizer_adaptive.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer_adaptive.step()
        adaptive_losses.append(loss.item())
        
        # Fixed clipping
        optimizer_fixed.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer_fixed.step()
        fixed_losses.append(loss.item())
        
        if (epoch + 1) % 2 == 0:
            adaptive_threshold = optimizer_adaptive.get_clip_threshold()
            print(f"  Epoch {epoch+1}: Adaptive threshold = {adaptive_threshold:.4f}")
    
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"  Adaptive clipping final loss: {adaptive_losses[-1]:.4f}")
    print(f"  Fixed clipping final loss: {fixed_losses[-1]:.4f}")
    
    print("\nRecommendations:")
    print("- Adaptive clipping: Better for non-stationary data")
    print("- Fixed clipping: More predictable, good for debugging")


def example_4_mixed_precision():
    """
    Example 4: Mixed Precision Training
    
    Demonstrates ULTRON_V2's support for mixed precision training
    with FP16/BF16 state buffers.
    """
    print("\n" + "=" * 60)
    print("Example 4: Mixed Precision Training")
    print("=" * 60)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Test different state precisions
    precisions = ['fp32', 'fp16', 'bf16']
    results = {}
    
    for precision in precisions:
        print(f"\nTesting {precision.upper()} state precision...")
        
        # Create optimizer with specific precision
        optimizer = ULTRON_V2(
            model.parameters(),
            lr=0.001,
            state_precision=precision
        )
        
        # Create dummy data
        x = torch.randn(64, 100)
        y = torch.randint(0, 10, (64,))
        criterion = nn.CrossEntropyLoss()
        
        # Train for 5 epochs
        losses = []
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Get memory usage
        memory_stats = optimizer.get_memory_usage()
        results[precision] = {
            'final_loss': losses[-1],
            'state_mb': memory_stats['state_mb']
        }
        
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  State memory: {memory_stats['state_mb']:.2f} MB")
    
    print("\n" + "-" * 60)
    print("Results Summary:")
    print("-" * 60)
    for precision, stats in results.items():
        print(f"  {precision.upper():<10} Loss: {stats['final_loss']:.4f}, "
              f"State: {stats['state_mb']:.2f} MB")
    
    print("\nRecommendations:")
    print("- FP32: Best accuracy, highest memory")
    print("- FP16/BF16: Lower memory, good for large models")
    print("- Use FP16/BF16 when memory is constrained")


def example_5_nesterov_momentum():
    """
    Example 5: Nesterov Momentum
    
    Demonstrates ULTRON_V2's Nesterov-style lookahead feature,
    which can improve convergence on certain problems.
    """
    print("\n" + "=" * 60)
    print("Example 5: Nesterov Momentum")
    print("=" * 60)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Compare standard vs Nesterov
    print("\nComparing standard vs Nesterov momentum...")
    
    # Standard momentum
    optimizer_standard = ULTRON_V2(
        model.parameters(),
        lr=0.001,
        nesterov=False
    )
    
    # Nesterov momentum
    optimizer_nesterov = ULTRON_V2(
        model.parameters(),
        lr=0.001,
        nesterov=True
    )
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Train both for 10 epochs
    standard_losses = []
    nesterov_losses = []
    
    for epoch in range(10):
        # Standard
        optimizer_standard.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer_standard.step()
        standard_losses.append(loss.item())
        
        # Nesterov
        optimizer_nesterov.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer_nesterov.step()
        nesterov_losses.append(loss.item())
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: Standard = {standard_losses[-1]:.4f}, "
                  f"Nesterov = {nesterov_losses[-1]:.4f}")
    
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"  Standard momentum final loss: {standard_losses[-1]:.4f}")
    print(f"  Nesterov momentum final loss: {nesterov_losses[-1]:.4f}")
    
    print("\nRecommendations:")
    print("- Nesterov: Often converges faster on convex problems")
    print("- Standard: More stable for noisy gradients")


def example_6_learning_rate_scheduling():
    """
    Example 6: Learning Rate Scheduling
    
    Demonstrates ULTRON_V2's built-in learning rate scheduling
    with warmup and decay.
    """
    print("\n" + "=" * 60)
    print("Example 6: Learning Rate Scheduling")
    print("=" * 60)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create optimizer with warmup and decay
    optimizer = ULTRON_V2(
        model.parameters(),
        lr=0.001,
        warmup_steps=100,
        decay_steps=1000,
        decay_rate=0.95
    )
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Train for 20 epochs
    losses = []
    learning_rates = []
    
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}")
    
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Initial LR: {learning_rates[0]:.6f}")
    print(f"  Final LR: {learning_rates[-1]:.6f}")
    
    print("\nRecommendations:")
    print("- Warmup: Helps with unstable early training")
    print("- Decay: Improves final convergence")
    print("- Combined: Best for long training runs")


def example_7_cnn_training():
    """
    Example 7: CNN Training on MNIST
    
    Demonstrates ULTRON_V2 on a real CNN architecture
    with MNIST dataset.
    """
    print("\n" + "=" * 60)
    print("Example 7: CNN Training on MNIST")
    print("=" * 60)
    
    # Import MNIST model
    from models.cnn_mnist import SimpleCNN
    
    # Create model
    model = SimpleCNN()
    
    # Create ULTRON_V2 optimizer
    optimizer = ULTRON_V2(
        model.parameters(),
        lr=0.001,
        normalize_gradients=True,
        normalization_strategy='rms',
        adaptive_clipping=True
    )
    
    # Create dummy data (simulating MNIST)
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nTraining for 5 epochs...")
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print("\n✓ CNN training example completed!")
    
    print("\nKey Points:")
    print("- ULTRON_V2 works well with CNN architectures")
    print("- Adaptive clipping helps with varying gradient magnitudes")
    print("- RMS normalization provides stable training")


def example_8_comparison_with_ultron():
    """
    Example 8: Comparison with Original ULTRON
    
    Demonstrates the improvements of ULTRON_V2 over the original ULTRON.
    """
    print("\n" + "=" * 60)
    print("Example 8: Comparison with Original ULTRON")
    print("=" * 60)
    
    # Import original ULTRON
    from optimizers.ultron import ULTRON
    
    # Create identical models
    model1 = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    model2 = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Copy parameters to ensure identical initialization
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.copy_(p1)
    
    # Create optimizers with similar hyperparameters
    optimizer_v2 = ULTRON_V2(
        model1.parameters(),
        lr=0.001,
        normalize_gradients=True,
        normalization_strategy='rms',
        adaptive_clipping=False  # Disable for fair comparison
    )
    
    optimizer_original = ULTRON(
        model2.parameters(),
        lr=0.001,
        normalize_gradients=True
    )
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Train both for 10 epochs
    v2_losses = []
    original_losses = []
    
    for epoch in range(10):
        # ULTRON_V2
        optimizer_v2.zero_grad()
        output1 = model1(x)
        loss1 = criterion(output1, y)
        loss1.backward()
        optimizer_v2.step()
        v2_losses.append(loss1.item())
        
        # Original ULTRON
        optimizer_original.zero_grad()
        output2 = model2(x)
        loss2 = criterion(output2, y)
        loss2.backward()
        optimizer_original.step()
        original_losses.append(loss2.item())
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: ULTRON_V2 = {v2_losses[-1]:.4f}, "
                  f"ULTRON = {original_losses[-1]:.4f}")
    
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"  ULTRON_V2 final loss: {v2_losses[-1]:.4f}")
    print(f"  ULTRON final loss: {original_losses[-1]:.4f}")
    print(f"  Difference: {abs(v2_losses[-1] - original_losses[-1]):.4f}")
    
    print("\nRecommendations:")
    print("- ULTRON_V2: Faster due to vectorized updates")
    print("- ULTRON_V2: More memory efficient with single buffer")
    print("- Original ULTRON: Simpler, easier to debug")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ULTRON_V2 Optimizer Usage Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Normalization Strategies", example_2_normalization_strategies),
        ("Adaptive Clipping", example_3_adaptive_clipping),
        ("Mixed Precision", example_4_mixed_precision),
        ("Nesterov Momentum", example_5_nesterov_momentum),
        ("Learning Rate Scheduling", example_6_learning_rate_scheduling),
        ("CNN Training", example_7_cnn_training),
        ("Comparison with ULTRON", example_8_comparison_with_ultron),
    ]
    
    print(f"\nAvailable examples ({len(examples)}):")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i+1}. {name}")
    
    # Run all examples
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    
    print("\nFor more information, see:")
    print("- README_ULTRON_V2.md (API documentation)")
    print("- benchmarks/ultron_v2_benchmark.py (performance benchmarks)")


if __name__ == '__main__':
    main()
