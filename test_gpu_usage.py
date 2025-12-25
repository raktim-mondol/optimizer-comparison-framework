#!/usr/bin/env python3
"""
Test script to verify GPU usage during training.
"""

import torch
import time
import subprocess
import threading
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from experiments.experiment_runner import ExperimentRunner

def monitor_gpu_usage(duration=10):
    """Monitor GPU usage for specified duration."""
    print(f"\nMonitoring GPU usage for {duration} seconds...")
    print("GPU Memory Usage (MB):")
    
    for i in range(duration):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  {i+1}s: Allocated: {memory_allocated:.1f} MB, Reserved: {memory_reserved:.1f} MB")
        else:
            print(f"  {i+1}s: CUDA not available")
        time.sleep(1)

def test_gpu_usage():
    """Test if GPU is being used during a simple experiment."""
    print("="*60)
    print("GPU Usage Test")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. PyTorch cannot use GPU.")
        return False
    
    print(f"✓ CUDA is available")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create a simple experiment config
    config = {
        'experiment_name': 'gpu_test_mnist_simple_cnn_adam',
        'dataset': 'mnist',
        'model': 'simple_cnn',
        'optimizer': 'Adam',
        'epochs': 1,  # Just 1 epoch to test
        'batch_size': 64,
        'learning_rate': 0.001,
        'data_augmentation': False,
        'use_scheduler': False,
        'patience': 10,
        'min_delta': 0.001,
        'seed': 42,
        'optimizer_params': {'betas': (0.9, 0.999)}
    }
    
    # Set device to GPU
    device = torch.device('cuda:0')
    print(f"\n✓ Using device: {device}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("\nStarting experiment...")
    print("Initial GPU memory:")
    if torch.cuda.is_available():
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(5,))
    monitor_thread.start()
    
    # Run experiment
    try:
        runner = ExperimentRunner(output_dir='gpu_test_results', device=device)
        result = runner.run_experiment(config, 'gpu_test')
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        print("\n" + "="*60)
        print("Experiment completed!")
        print("="*60)
        
        # Check final GPU memory
        if torch.cuda.is_available():
            print(f"\nFinal GPU memory:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            print(f"  Reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
        
        # Check if model was actually on GPU
        print(f"\nResult keys: {result.keys()}")
        if 'best_test_accuracy' in result:
            print(f"✓ Test Accuracy: {result['best_test_accuracy']:.2f}%")
        
        print("\n✅ GPU usage test PASSED - GPU was used during training")
        return True
        
    except Exception as e:
        print(f"\n❌ GPU usage test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_gpu_usage()
    sys.exit(0 if success else 1)