#!/usr/bin/env python3
"""
Test script to run one complete experiment.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from experiments.experiment_runner import ExperimentRunner

def main():
    print("Testing one complete experiment...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir='test_one_experiment', device=device)
    
    # Run one experiment with Adam optimizer (we know this works)
    config = {
        'experiment_name': 'test_mnist_simple_cnn_adam',
        'dataset': 'mnist',
        'model': 'simple_cnn',
        'optimizer': 'Adam',
        'epochs': 2,  # Just 2 epochs for quick test
        'batch_size': 64,
        'learning_rate': 0.001,
        'data_augmentation': False,
        'use_scheduler': False,
        'patience': 10,
        'min_delta': 0.001,
        'seed': 42,
        'optimizer_params': {'betas': (0.9, 0.999)}
    }
    
    print(f"\nRunning experiment: {config['experiment_name']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Model: {config['model']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    try:
        result = runner.run_experiment(config, config['experiment_name'])
        print(f"\n✓ Experiment completed successfully!")
        print(f"Best test accuracy: {result.get('best_test_accuracy', 'N/A'):.2f}%")
        print(f"Total training time: {result.get('total_training_time', 'N/A'):.2f}s")
        
        # Check if results were saved
        result_path = Path('test_one_experiment') / 'experiments' / 'raw_results' / f"{config['experiment_name']}_result.json"
        if result_path.exists():
            print(f"Results saved to: {result_path}")
        
        checkpoint_path = Path('test_one_experiment') / 'experiments' / 'checkpoints' / f"{config['experiment_name']}_best.pth"
        if checkpoint_path.exists():
            print(f"Model checkpoint saved to: {checkpoint_path}")
        
        return 0
    except Exception as e:
        print(f"\n✗ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())