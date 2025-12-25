#!/usr/bin/env python3
"""
Test script to verify the optimizer comparison framework works correctly.
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from optimizers.amcas import AMCAS
        print("✓ AMCAS optimizer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AMCAS: {e}")
        return False
    
    try:
        from models.cnn_mnist import SimpleCNN, get_mnist_model
        print("✓ MNIST CNN models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MNIST CNN models: {e}")
        return False
    
    try:
        from models.cnn_cifar10 import CIFAR10ResNet, get_cifar10_model
        print("✓ CIFAR10 CNN models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CIFAR10 CNN models: {e}")
        return False
    
    try:
        from models.vit_mnist import VisionTransformerMNISTSmall
        print("✓ MNIST Vision Transformer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MNIST Vision Transformer: {e}")
        return False
    
    try:
        from models.vit_cifar10 import VisionTransformerCIFAR10Small
        print("✓ CIFAR10 Vision Transformer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CIFAR10 Vision Transformer: {e}")
        return False
    
    try:
        from experiments.experiment_runner import ExperimentRunner
        print("✓ Experiment runner imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import experiment runner: {e}")
        return False
    
    try:
        from experiments.metrics_collector import MetricsCollector
        print("✓ Metrics collector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import metrics collector: {e}")
        return False
    
    try:
        from experiments.results_exporter import ResultsExporter
        print("✓ Results exporter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import results exporter: {e}")
        return False
    
    try:
        from benchmarks.comprehensive_benchmark import ComprehensiveBenchmark
        print("✓ Comprehensive benchmark imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import comprehensive benchmark: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that models can be created."""
    print("\nTesting model creation...")
    
    try:
        # Test MNIST models
        mnist_cnn = SimpleCNN()
        print(f"✓ Created MNIST CNN with {sum(p.numel() for p in mnist_cnn.parameters()):,} parameters")
        
        mnist_vit = VisionTransformerMNISTSmall()
        print(f"✓ Created MNIST ViT with {sum(p.numel() for p in mnist_vit.parameters()):,} parameters")
        
        # Test CIFAR10 models
        cifar10_cnn = CIFAR10ResNet()
        print(f"✓ Created CIFAR10 ResNet with {sum(p.numel() for p in cifar10_cnn.parameters()):,} parameters")
        
        cifar10_vit = VisionTransformerCIFAR10Small()
        print(f"✓ Created CIFAR10 ViT with {sum(p.numel() for p in cifar10_vit.parameters()):,} parameters")
        
        # Test model registry
        mnist_model = get_mnist_model('simple_cnn')
        print(f"✓ Retrieved MNIST model from registry: {type(mnist_model).__name__}")
        
        cifar10_model = get_cifar10_model('resnet')
        print(f"✓ Retrieved CIFAR10 model from registry: {type(cifar10_model).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create models: {e}")
        return False

def test_optimizer_creation():
    """Test that optimizers can be created."""
    print("\nTesting optimizer creation...")
    
    try:
        # Create a simple model
        model = SimpleCNN()
        
        # Test AMCAS optimizer
        optimizer_amcas = AMCAS(model.parameters(), lr=0.001)
        print("✓ Created AMCAS optimizer")
        
        # Test other optimizers
        import torch.optim as optim
        
        optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
        print("✓ Created Adam optimizer")
        
        optimizer_sgd = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        print("✓ Created SGD optimizer")
        
        optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001)
        print("✓ Created RMSprop optimizer")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create optimizers: {e}")
        return False

def test_experiment_runner():
    """Test that experiment runner can be initialized."""
    print("\nTesting experiment runner...")
    
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create experiment runner
        runner = ExperimentRunner(output_dir='test_results', device=device)
        print("✓ Created experiment runner")
        
        # Test dataset loading
        train_loader, test_loader = runner.get_dataset('mnist', batch_size=32)
        print(f"✓ Loaded MNIST dataset: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")
        
        train_loader, test_loader = runner.get_dataset('cifar10', batch_size=32)
        print(f"✓ Loaded CIFAR10 dataset: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")
        
        # Test model creation
        model = runner.get_model('simple_cnn', 'mnist')
        print(f"✓ Created model: {type(model).__name__}")
        
        # Test optimizer creation
        optimizer = runner.get_optimizer('AMCAS', model, lr=0.001)
        print(f"✓ Created optimizer: {type(optimizer).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test experiment runner: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_files():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration files...")
    
    try:
        import yaml
        
        config_files = [
            'experiments/configs/mnist_cnn.yaml',
            'experiments/configs/mnist_vit.yaml',
            'experiments/configs/cifar10_cnn.yaml',
            'experiments/configs/cifar10_vit.yaml',
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✓ Loaded configuration: {config_file}")
                print(f"  Dataset: {config.get('dataset', 'N/A')}")
                print(f"  Model: {config.get('model', 'N/A')}")
                print(f"  Optimizers: {len(config.get('optimizers', []))}")
            else:
                print(f"✗ Configuration file not found: {config_file}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration files: {e}")
        return False

def test_metrics_collector():
    """Test that metrics collector works."""
    print("\nTesting metrics collector...")
    
    try:
        from experiments.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Test timer
        collector.start_timer('test_timer')
        import time
        time.sleep(0.1)
        elapsed = collector.stop_timer('test_timer')
        print(f"✓ Timer test: {elapsed:.3f}s")
        
        # Test metric collection
        collector.add_metric('test_metric', 1.0)
        collector.add_metric('test_metric', 2.0)
        collector.add_metric('test_metric', 3.0)
        
        metrics = collector.get_metrics()
        print(f"✓ Collected metrics: {list(metrics.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test metrics collector: {e}")
        return False

def test_results_exporter():
    """Test that results exporter works."""
    print("\nTesting results exporter...")
    
    try:
        from experiments.results_exporter import ResultsExporter
        
        exporter = ResultsExporter(output_dir='test_results')
        print("✓ Created results exporter")
        
        # Create dummy results
        dummy_results = {
            'test_experiment': {
                'train_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
                'train_accuracy': [50.0, 60.0, 70.0, 80.0, 90.0],
                'test_loss': [0.6, 0.5, 0.4, 0.3, 0.2],
                'test_accuracy': [45.0, 55.0, 65.0, 75.0, 85.0],
                'best_test_accuracy': 85.0,
                'total_training_time': 120.5,
                'model_params': 1000000,
                'memory_usage': [1.2, 1.3, 1.4, 1.3, 1.2],
            }
        }
        
        # Test export
        excel_path = exporter.export_to_excel(dummy_results, 'test_results.xlsx')
        print(f"✓ Exported to Excel: {excel_path}")
        
        json_path = exporter.export_to_json(dummy_results, 'test_results.json')
        print(f"✓ Exported to JSON: {json_path}")
        
        # Test report generation
        report_path = exporter.generate_report(dummy_results, 'test_report.md')
        print(f"✓ Generated report: {report_path}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test results exporter: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmarks():
    """Test that benchmarks can be initialized."""
    print("\nTesting benchmarks...")
    
    try:
        from benchmarks.comprehensive_benchmark import ComprehensiveBenchmark
        from benchmarks.memory_profiler import MemoryProfiler
        from benchmarks.speed_benchmark import SpeedBenchmark
        from benchmarks.optimizer_comparison import OptimizerComparison
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test comprehensive benchmark
        benchmark = ComprehensiveBenchmark(output_dir='test_benchmark', device=device)
        print("✓ Created comprehensive benchmark")
        
        # Test memory profiler
        memory_profiler = MemoryProfiler(output_dir='test_memory', device=device)
        print("✓ Created memory profiler")
        
        # Test speed benchmark
        speed_benchmark = SpeedBenchmark(output_dir='test_speed', device=device)
        print("✓ Created speed benchmark")
        
        # Test optimizer comparison
        optimizer_comparison = OptimizerComparison(output_dir='test_comparison')
        print("✓ Created optimizer comparison")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*80)
    print("Testing Optimizer Comparison Framework")
    print("="*80)
    
    tests = [
        ("Import tests", test_imports),
        ("Model creation", test_model_creation),
        ("Optimizer creation", test_optimizer_creation),
        ("Experiment runner", test_experiment_runner),
        ("Configuration files", test_config_files),
        ("Metrics collector", test_metrics_collector),
        ("Results exporter", test_results_exporter),
        ("Benchmarks", test_benchmarks),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-"*40)
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("\nFramework is ready to use.")
        print("\nTo run all experiments:")
        print("  python scripts/run_all_experiments.py")
        print("\nTo run a specific configuration:")
        print("  python scripts/run_all_experiments.py --config experiments/configs/mnist_cnn.yaml")
        print("\nTo run the full benchmark suite:")
        print("  python scripts/run_all_experiments.py --full")
    else:
        print("SOME TESTS FAILED! ✗")
        print("\nPlease check the errors above and fix them before running experiments.")
    
    print("="*80)
    
    # Clean up test directories
    import shutil
    for dir_name in ['test_results', 'test_benchmark', 'test_memory', 'test_speed', 'test_comparison']:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"Cleaned up {dir_name}/")
            except:
                pass
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)