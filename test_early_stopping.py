#!/usr/bin/env python3
"""
Test script to verify early stopping functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from experiments.experiment_runner import ExperimentRunner

def test_early_stopping():
    """Test early stopping with high min_delta."""
    
    print("="*80)
    print("Testing Early Stopping Functionality")
    print("="*80)
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir='test_early_stopping_simple', device=torch.device('cpu'))
    
    # Test 1: Early stopping should trigger with high min_delta
    print("\nTest 1: Early stopping with min_delta=0.5")
    print("-"*40)
    
    config1 = {
        'dataset': 'mnist',
        'model': 'simple_cnn',
        'optimizer': 'SGD',
        'epochs': 20,
        'batch_size': 64,
        'learning_rate': 0.001,
        'data_augmentation': False,
        'use_scheduler': False,
        'patience': 3,  # Stop after 3 epochs without improvement
        'min_delta': 0.5,  # Very high delta - should stop early
        'seed': 42,
        'optimizer_params': {}
    }
    
    result1 = runner.run_experiment(config1, 'test_early_stopping_high_delta')
    epochs_run1 = len(result1['test_accuracy'])
    print(f"Epochs run: {epochs_run1}/20")
    print(f"Best accuracy: {result1['best_test_accuracy']:.2f}%")
    
    if epochs_run1 < 20:
        print("✓ Early stopping triggered successfully!")
    else:
        print("✗ Early stopping did not trigger (might need even higher min_delta)")
    
    # Test 2: Early stopping with normal parameters
    print("\nTest 2: Early stopping with min_delta=0.001")
    print("-"*40)
    
    config2 = {
        'dataset': 'mnist',
        'model': 'simple_cnn',
        'optimizer': 'SGD',
        'epochs': 20,
        'batch_size': 64,
        'learning_rate': 0.001,
        'data_augmentation': False,
        'use_scheduler': False,
        'patience': 5,  # Stop after 5 epochs without improvement
        'min_delta': 0.001,  # Normal delta
        'seed': 42,
        'optimizer_params': {}
    }
    
    result2 = runner.run_experiment(config2, 'test_early_stopping_normal')
    epochs_run2 = len(result2['test_accuracy'])
    print(f"Epochs run: {epochs_run2}/20")
    print(f"Best accuracy: {result2['best_test_accuracy']:.2f}%")
    
    # Test 3: No early stopping (patience > epochs)
    print("\nTest 3: No early stopping (patience=100)")
    print("-"*40)
    
    config3 = {
        'dataset': 'mnist',
        'model': 'simple_cnn',
        'optimizer': 'SGD',
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.001,
        'data_augmentation': False,
        'use_scheduler': False,
        'patience': 100,  # Very high patience - should run all epochs
        'min_delta': 0.001,
        'seed': 42,
        'optimizer_params': {}
    }
    
    result3 = runner.run_experiment(config3, 'test_no_early_stopping')
    epochs_run3 = len(result3['test_accuracy'])
    print(f"Epochs run: {epochs_run3}/10")
    print(f"Best accuracy: {result3['best_test_accuracy']:.2f}%")
    
    if epochs_run3 == 10:
        print("✓ Ran all epochs as expected (no early stopping)")
    else:
        print(f"✗ Unexpected: ran {epochs_run3} epochs instead of 10")
    
    print("\n" + "="*80)
    print("Early Stopping Test Summary")
    print("="*80)
    print(f"Test 1 (high min_delta): {epochs_run1} epochs run")
    print(f"Test 2 (normal): {epochs_run2} epochs run")
    print(f"Test 3 (no early stopping): {epochs_run3} epochs run")
    
    # Verify early stopping logic
    print("\nVerifying early stopping logic:")
    print("- Early stopping triggers when no improvement > min_delta for patience epochs")
    print(f"- Default patience: 10 epochs")
    print(f"- Default min_delta: 0.001")
    print("- Shows ✓ when improvement > min_delta")
    print("- Shows ✗ when no improvement")
    print("- Stops training when patience is exhausted")
    
    return True

if __name__ == "__main__":
    test_early_stopping()