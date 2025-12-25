#!/usr/bin/env python3
"""
Simple test script to verify the optimizer comparison framework works.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from experiments.experiment_runner import ExperimentRunner

def main():
    print("Testing optimizer comparison framework...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir='test_simple_results', device=device)
    print("Experiment runner created successfully")
    
    # Test dataset loading
    print("\nTesting MNIST dataset loading...")
    train_loader, test_loader = runner.get_dataset('mnist', batch_size=32)
    print(f"MNIST dataset loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")
    
    print("\nTesting CIFAR10 dataset loading...")
    train_loader, test_loader = runner.get_dataset('cifar10', batch_size=32)
    print(f"CIFAR10 dataset loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")
    
    # Test model creation
    print("\nTesting model creation...")
    mnist_model = runner.get_model('simple_cnn', 'mnist')
    print(f"MNIST model created: {type(mnist_model).__name__}")
    
    cifar10_model = runner.get_model('resnet', 'cifar10')
    print(f"CIFAR10 model created: {type(cifar10_model).__name__}")
    
    # Test optimizer creation
    print("\nTesting optimizer creation...")
    optimizers_to_test = ['AMCAS', 'Adam', 'SGD']
    
    for opt_name in optimizers_to_test:
        optimizer = runner.get_optimizer(opt_name, mnist_model, lr=0.001)
        print(f"{opt_name} optimizer created: {type(optimizer).__name__}")
    
    # Run a simple experiment
    print("\nRunning a simple experiment...")
    config = {
        'experiment_name': 'test_mnist_simple_cnn_AMCAS',
        'dataset': 'mnist',
        'model': 'simple_cnn',
        'optimizer': 'AMCAS',
        'epochs': 1,  # Just 1 epoch for testing
        'batch_size': 32,
        'learning_rate': 0.001,
        'data_augmentation': False,
        'use_scheduler': False,
        'patience': 10,
        'min_delta': 0.001,
        'seed': 42,
        'optimizer_params': {}
    }
    
    try:
        result = runner.run_experiment(config, 'test_experiment')
        print(f"\nExperiment completed successfully!")
        print(f"Final test accuracy: {result.get('best_test_accuracy', 'N/A'):.2f}%")
        print(f"Total training time: {result.get('total_training_time', 'N/A'):.2f}s")
        print(f"Results saved to: test_simple_results/")
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("All tests passed! The optimizer comparison framework is working correctly.")
    print("="*80)
    return 0

if __name__ == '__main__':
    sys.exit(main())