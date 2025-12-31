"""
Performance benchmark for ULTRON_V2 optimizer.
Compares ULTRON_V2 against ULTRON and standard optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from optimizers.ultron_v2 import ULTRON_V2
from optimizers.ultron import ULTRON
from optimizers.amcas import AMCAS
from models.cnn_mnist import SimpleCNN, MNISTCNNV2, MNISTCNNV3


class ULTRONV2Benchmark:
    """
    Comprehensive benchmark for ULTRON_V2 optimizer.
    """
    
    def __init__(self, output_dir='ultron_v2_benchmark', device=None):
        """
        Initialize benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
            device: PyTorch device (cpu or cuda)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Available optimizers for comparison
        self.optimizer_registry = {
            'ULTRON_V2': ULTRON_V2,
            'ULTRON': ULTRON,
            'AMCAS': AMCAS,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'SGD+Momentum': lambda params, lr, **kwargs: torch.optim.SGD(params, lr=lr, momentum=0.9, **kwargs),
            'RMSprop': torch.optim.RMSprop,
            'RAdam': torch.optim.RAdam,
            'NAdam': torch.optim.NAdam,
        }
        
        # Default optimizer parameters
        self.default_optimizer_params = {
            'ULTRON_V2': {
                'betas': (0.9, 0.999),
                'clip_threshold': 1.0,
                'normalize_gradients': True,
                'normalization_strategy': 'rms',
                'adaptive_clipping': True,
                'state_precision': 'fp32',
                'momentum_correction': True,
            },
            'ULTRON': {
                'betas': (0.9, 0.999),
                'clip_threshold': 1.0,
                'normalize_gradients': True,
            },
            'AMCAS': {
                'betas': (0.9, 0.999),
                'gamma': 0.1,
                'lambda_consistency': 0.01,
            },
            'Adam': {'betas': (0.9, 0.999)},
            'AdamW': {'betas': (0.9, 0.999), 'weight_decay': 0.01},
            'SGD': {},
            'SGD+Momentum': {},
            'RMSprop': {},
            'RAdam': {'betas': (0.9, 0.999)},
            'NAdam': {'betas': (0.9, 0.999)},
        }
        
        print(f"ULTRON_V2 Benchmark initialized with device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def create_test_model(self, model_type: str = 'simple_cnn') -> nn.Module:
        """
        Create test model for benchmarking.
        
        Args:
            model_type: Type of model to create ('simple_cnn', 'cnn_v2', 'cnn_v3')
            
        Returns:
            PyTorch model
        """
        if model_type == 'simple_cnn':
            model = SimpleCNN()
        elif model_type == 'cnn_v2':
            model = MNISTCNNV2()
        elif model_type == 'cnn_v3':
            model = MNISTCNNV3()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def benchmark_training_speed(self, 
                                optimizer_name: str,
                                model_type: str = 'simple_cnn',
                                batch_size: int = 64,
                                num_iterations: int = 1000,
                                warmup_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark training speed for a specific optimizer.
        
        Args:
            optimizer_name: Name of the optimizer to benchmark
            model_type: Type of model to use
            batch_size: Batch size for benchmarking
            num_iterations: Number of training iterations to benchmark
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with speed benchmarking results
        """
        print(f"\nBenchmarking training speed for {optimizer_name} on {model_type}...")
        
        # Create model
        model = self.create_test_model(model_type)
        
        # Create optimizer
        optimizer_class = self.optimizer_registry[optimizer_name]
        optimizer_params = self.default_optimizer_params.get(optimizer_name, {}).copy()
        
        if optimizer_name == 'SGD+Momentum':
            optimizer = optimizer_class(model.parameters(), lr=0.001, **optimizer_params)
        else:
            optimizer = optimizer_class(model.parameters(), lr=0.001, **optimizer_params)
        
        # Create dummy data
        if model_type in ['simple_cnn', 'cnn_v2', 'cnn_v3']:
            input_size = (batch_size, 1, 28, 28)  # MNIST size
        else:
            input_size = (batch_size, 3, 32, 32)  # CIFAR-10 size
        
        dummy_input = torch.randn(input_size).to(self.device)
        dummy_target = torch.randint(0, 10, (batch_size,)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Speed tracking
        speed_stats = {
            'optimizer_name': optimizer_name,
            'model_type': model_type,
            'batch_size': batch_size,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'forward_times': [],
            'backward_times': [],
            'optimizer_step_times': [],
            'total_training_times': [],
            'throughput_samples_per_sec': 0,
            'throughput_iterations_per_sec': 0,
        }
        
        # Warmup
        print(f"  Warming up for {warmup_iterations} iterations...")
        model.train()
        for _ in range(warmup_iterations):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
        
        # Benchmark training
        print(f"  Benchmarking training for {num_iterations} iterations...")
        model.train()
        
        for iteration in range(num_iterations):
            # Forward pass timing
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            forward_start = time.perf_counter()
            
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            forward_end = time.perf_counter()
            forward_time = forward_end - forward_start
            
            # Backward pass timing
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            backward_start = time.perf_counter()
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            backward_end = time.perf_counter()
            backward_time = backward_end - backward_start
            
            # Optimizer step timing
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            optimizer_start = time.perf_counter()
            
            optimizer.step()
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            optimizer_end = time.perf_counter()
            optimizer_time = optimizer_end - optimizer_start
            
            # Total training time
            total_time = forward_time + backward_time + optimizer_time
            
            # Record times
            speed_stats['forward_times'].append(forward_time)
            speed_stats['backward_times'].append(backward_time)
            speed_stats['optimizer_step_times'].append(optimizer_time)
            speed_stats['total_training_times'].append(total_time)
            
            if iteration % 100 == 0:
                print(f"    Iteration {iteration}/{num_iterations}: "
                      f"Forward: {forward_time*1000:.2f}ms, "
                      f"Backward: {backward_time*1000:.2f}ms, "
                      f"Optimizer: {optimizer_time*1000:.2f}ms, "
                      f"Total: {total_time*1000:.2f}ms")
        
        # Calculate throughput
        total_training_time = sum(speed_stats['total_training_times'])
        if total_training_time > 0:
            speed_stats['throughput_samples_per_sec'] = (batch_size * num_iterations) / total_training_time
            speed_stats['throughput_iterations_per_sec'] = num_iterations / total_training_time
        
        # Calculate statistics
        speed_stats['forward_time_mean'] = np.mean(speed_stats['forward_times'])
        speed_stats['forward_time_std'] = np.std(speed_stats['forward_times'])
        speed_stats['backward_time_mean'] = np.mean(speed_stats['backward_times'])
        speed_stats['backward_time_std'] = np.std(speed_stats['backward_times'])
        speed_stats['optimizer_step_time_mean'] = np.mean(speed_stats['optimizer_step_times'])
        speed_stats['optimizer_step_time_std'] = np.std(speed_stats['optimizer_step_times'])
        speed_stats['total_training_time_mean'] = np.mean(speed_stats['total_training_times'])
        speed_stats['total_training_time_std'] = np.std(speed_stats['total_training_times'])
        
        # Clean up
        del model, optimizer, dummy_input, dummy_target
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  Completed: Throughput = {speed_stats['throughput_samples_per_sec']:.2f} samples/sec")
        
        return speed_stats
    
    def benchmark_memory_usage(self,
                              optimizer_name: str,
                              model_type: str = 'simple_cnn') -> Dict[str, Any]:
        """
        Benchmark memory usage for a specific optimizer.
        
        Args:
            optimizer_name: Name of the optimizer to benchmark
            model_type: Type of model to use
            
        Returns:
            Dictionary with memory usage results
        """
        print(f"\nBenchmarking memory usage for {optimizer_name} on {model_type}...")
        
        # Create model
        model = self.create_test_model(model_type)
        
        # Create optimizer
        optimizer_class = self.optimizer_registry[optimizer_name]
        optimizer_params = self.default_optimizer_params.get(optimizer_name, {}).copy()
        
        if optimizer_name == 'SGD+Momentum':
            optimizer = optimizer_class(model.parameters(), lr=0.001, **optimizer_params)
        else:
            optimizer = optimizer_class(model.parameters(), lr=0.001, **optimizer_params)
        
        # Create dummy data
        if model_type in ['simple_cnn', 'cnn_v2', 'cnn_v3']:
            input_size = (64, 1, 28, 28)
        else:
            input_size = (64, 3, 32, 32)
        
        dummy_input = torch.randn(input_size).to(self.device)
        dummy_target = torch.randint(0, 10, (64,)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize state with one step
        model.train()
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        # Get memory usage
        memory_stats = {
            'optimizer_name': optimizer_name,
            'model_type': model_type,
            'total_bytes': 0,
            'state_bytes': 0,
            'param_bytes': 0,
        }
        
        # Calculate parameter memory
        param_bytes = 0
        for p in model.parameters():
            if p.requires_grad:
                param_bytes += p.numel() * p.element_size()
        
        memory_stats['param_bytes'] = param_bytes
        
        # Calculate state memory
        state_bytes = 0
        for p in model.parameters():
            if p.grad is not None:
                state = optimizer.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state_bytes += value.numel() * value.element_size()
        
        memory_stats['state_bytes'] = state_bytes
        memory_stats['total_bytes'] = param_bytes + state_bytes
        
        # Convert to MB
        memory_stats['total_mb'] = memory_stats['total_bytes'] / (1024 * 1024)
        memory_stats['state_mb'] = memory_stats['state_bytes'] / (1024 * 1024)
        memory_stats['param_mb'] = memory_stats['param_bytes'] / (1024 * 1024)
        
        # Clean up
        del model, optimizer, dummy_input, dummy_target
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  State memory: {memory_stats['state_mb']:.2f} MB")
        print(f"  Total memory: {memory_stats['total_mb']:.2f} MB")
        
        return memory_stats
    
    def benchmark_convergence(self,
                             optimizer_name: str,
                             model_type: str = 'simple_cnn',
                             num_epochs: int = 10,
                             batch_size: int = 64) -> Dict[str, Any]:
        """
        Benchmark convergence performance.
        
        Args:
            optimizer_name: Name of the optimizer to benchmark
            model_type: Type of model to use
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with convergence results
        """
        print(f"\nBenchmarking convergence for {optimizer_name} on {model_type}...")
        
        # Create model
        model = self.create_test_model(model_type)
        
        # Create optimizer
        optimizer_class = self.optimizer_registry[optimizer_name]
        optimizer_params = self.default_optimizer_params.get(optimizer_name, {}).copy()
        
        if optimizer_name == 'SGD+Momentum':
            optimizer = optimizer_class(model.parameters(), lr=0.001, **optimizer_params)
        else:
            optimizer = optimizer_class(model.parameters(), lr=0.001, **optimizer_params)
        
        # Create synthetic dataset
        if model_type in ['simple_cnn', 'cnn_v2', 'cnn_v3']:
            input_size = (1000, 1, 28, 28)
        else:
            input_size = (1000, 3, 32, 32)
        
        train_data = torch.randn(input_size).to(self.device)
        train_labels = torch.randint(0, 10, (1000,)).to(self.device)
        
        test_data = torch.randn(200, *input_size[1:]).to(self.device)
        test_labels = torch.randint(0, 10, (200,)).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        
        # Track convergence
        convergence_stats = {
            'optimizer_name': optimizer_name,
            'model_type': model_type,
            'train_losses': [],
            'test_losses': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'epoch_times': [],
        }
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            
            # Training
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Mini-batch training
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_data)
                _, predicted = output.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            
            train_loss = total_loss / len(train_data)
            train_accuracy = 100. * correct / total
            
            # Testing
            model.eval()
            with torch.no_grad():
                test_output = model(test_data)
                test_loss = criterion(test_output, test_labels).item()
                _, predicted = test_output.max(1)
                test_correct = predicted.eq(test_labels).sum().item()
                test_accuracy = 100. * test_correct / len(test_labels)
            
            epoch_time = time.perf_counter() - epoch_start
            
            # Record statistics
            convergence_stats['train_losses'].append(train_loss)
            convergence_stats['test_losses'].append(test_loss)
            convergence_stats['train_accuracies'].append(train_accuracy)
            convergence_stats['test_accuracies'].append(test_accuracy)
            convergence_stats['epoch_times'].append(epoch_time)
            
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.2f}%, "
                  f"Test Acc: {test_accuracy:.2f}%, "
                  f"Time: {epoch_time:.2f}s")
        
        # Calculate final statistics
        convergence_stats['final_train_loss'] = convergence_stats['train_losses'][-1]
        convergence_stats['final_test_loss'] = convergence_stats['test_losses'][-1]
        convergence_stats['final_train_accuracy'] = convergence_stats['train_accuracies'][-1]
        convergence_stats['final_test_accuracy'] = convergence_stats['test_accuracies'][-1]
        convergence_stats['best_test_accuracy'] = max(convergence_stats['test_accuracies'])
        convergence_stats['total_training_time'] = sum(convergence_stats['epoch_times'])
        
        # Clean up
        del model, optimizer, train_data, train_labels, test_data, test_labels
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return convergence_stats
    
    def compare_all_optimizers(self,
                              model_types: List[str] = None,
                              batch_size: int = 64,
                              num_iterations: int = 500,
                              num_epochs: int = 5) -> Dict[str, Any]:
        """
        Compare all optimizers across different models.
        
        Args:
            model_types: List of model types to test
            batch_size: Batch size for benchmarking
            num_iterations: Number of iterations for speed benchmark
            num_epochs: Number of epochs for convergence benchmark
            
        Returns:
            Dictionary with comparison results
        """
        if model_types is None:
            model_types = ['simple_cnn', 'cnn_v2', 'cnn_v3']
        
        print(f"\n{'='*80}")
        print("ULTRON_V2 Comprehensive Benchmark")
        print(f"{'='*80}")
        
        all_results = {
            'speed_comparison': {},
            'memory_comparison': {},
            'convergence_comparison': {},
        }
        
        # Test each optimizer
        for optimizer_name in self.optimizer_registry.keys():
            print(f"\n{'='*60}")
            print(f"Testing {optimizer_name}")
            print(f"{'='*60}")
            
            optimizer_results = {
                'speed': {},
                'memory': {},
                'convergence': {},
            }
            
            try:
                # Test on each model type
                for model_type in model_types:
                    print(f"\nModel: {model_type}")
                    
                    # Speed benchmark
                    speed_results = self.benchmark_training_speed(
                        optimizer_name=optimizer_name,
                        model_type=model_type,
                        batch_size=batch_size,
                        num_iterations=num_iterations,
                        warmup_iterations=100
                    )
                    optimizer_results['speed'][model_type] = speed_results
                    
                    # Memory benchmark
                    memory_results = self.benchmark_memory_usage(
                        optimizer_name=optimizer_name,
                        model_type=model_type
                    )
                    optimizer_results['memory'][model_type] = memory_results
                    
                    # Convergence benchmark
                    convergence_results = self.benchmark_convergence(
                        optimizer_name=optimizer_name,
                        model_type=model_type,
                        num_epochs=num_epochs,
                        batch_size=batch_size
                    )
                    optimizer_results['convergence'][model_type] = convergence_results
                
                # Store results
                all_results['speed_comparison'][optimizer_name] = optimizer_results['speed']
                all_results['memory_comparison'][optimizer_name] = optimizer_results['memory']
                all_results['convergence_comparison'][optimizer_name] = optimizer_results['convergence']
                
                print(f"\n✓ {optimizer_name} benchmark completed successfully!")
                
            except Exception as e:
                print(f"\n✗ Error benchmarking {optimizer_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate summary report
        self._generate_comparison_report(all_results, model_types)
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _generate_comparison_report(self, all_results: Dict[str, Any], model_types: List[str]):
        """Generate comparison report."""
        report_path = self.output_dir / 'ultron_v2_comparison_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# ULTRON_V2 Optimizer Comparison Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Device: {self.device}\n")
            if self.device.type == 'cuda':
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Models tested: {', '.join(model_types)}\n\n")
            
            # Speed comparison summary
            f.write("## Speed Performance Summary\n\n")
            f.write("| Optimizer | Simple CNN (samples/sec) | CNN V2 (samples/sec) | CNN V3 (samples/sec) |\n")
            f.write("|-----------|-------------------------|----------------------|----------------------|\n")
            
            for optimizer_name in self.optimizer_registry.keys():
                if optimizer_name in all_results['speed_comparison']:
                    speeds = []
                    for model_type in model_types:
                        if model_type in all_results['speed_comparison'][optimizer_name]:
                            speed = all_results['speed_comparison'][optimizer_name][model_type]['throughput_samples_per_sec']
                            speeds.append(f"{speed:.2f}")
                        else:
                            speeds.append("N/A")
                    f.write(f"| {optimizer_name} | {' | '.join(speeds)} |\n")
            
            f.write("\n")
            
            # Memory comparison summary
            f.write("## Memory Usage Summary\n\n")
            f.write("| Optimizer | Simple CNN (MB) | CNN V2 (MB) | CNN V3 (MB) |\n")
            f.write("|-----------|-----------------|-------------|-------------|\n")
            
            for optimizer_name in self.optimizer_registry.keys():
                if optimizer_name in all_results['memory_comparison']:
                    memories = []
                    for model_type in model_types:
                        if model_type in all_results['memory_comparison'][optimizer_name]:
                            memory = all_results['memory_comparison'][optimizer_name][model_type]['total_mb']
                            memories.append(f"{memory:.2f}")
                        else:
                            memories.append("N/A")
                    f.write(f"| {optimizer_name} | {' | '.join(memories)} |\n")
            
            f.write("\n")
            
            # Convergence comparison summary
            f.write("## Convergence Performance Summary\n\n")
            f.write("| Optimizer | Simple CNN (Test Acc %) | CNN V2 (Test Acc %) | CNN V3 (Test Acc %) |\n")
            f.write("|-----------|-------------------------|---------------------|---------------------|\n")
            
            for optimizer_name in self.optimizer_registry.keys():
                if optimizer_name in all_results['convergence_comparison']:
                    accuracies = []
                    for model_type in model_types:
                        if model_type in all_results['convergence_comparison'][optimizer_name]:
                            accuracy = all_results['convergence_comparison'][optimizer_name][model_type]['best_test_accuracy']
                            accuracies.append(f"{accuracy:.2f}")
                        else:
                            accuracies.append("N/A")
                    f.write(f"| {optimizer_name} | {' | '.join(accuracies)} |\n")
            
            f.write("\n")
            
            # Performance analysis
            f.write("## Performance Analysis\n\n")
            
            # Find best optimizer for each category
            categories = {
                'Speed (Simple CNN)': ('speed_comparison', 'simple_cnn', 'throughput_samples_per_sec', True),
                'Speed (CNN V2)': ('speed_comparison', 'cnn_v2', 'throughput_samples_per_sec', True),
                'Speed (CNN V3)': ('speed_comparison', 'cnn_v3', 'throughput_samples_per_sec', True),
                'Memory Efficiency': ('memory_comparison', 'simple_cnn', 'total_mb', False),
                'Convergence (Simple CNN)': ('convergence_comparison', 'simple_cnn', 'best_test_accuracy', True),
                'Convergence (CNN V2)': ('convergence_comparison', 'cnn_v2', 'best_test_accuracy', True),
                'Convergence (CNN V3)': ('convergence_comparison', 'cnn_v3', 'best_test_accuracy', True),
            }
            
            f.write("### Best Optimizer by Category\n\n")
            for category_name, (result_type, model_type, metric, higher_better) in categories.items():
                best_optimizer = None
                best_value = -float('inf') if higher_better else float('inf')
                
                for optimizer_name in self.optimizer_registry.keys():
                    if (optimizer_name in all_results[result_type] and 
                        model_type in all_results[result_type][optimizer_name]):
                        
                        value = all_results[result_type][optimizer_name][model_type][metric]
                        
                        if higher_better:
                            if value > best_value:
                                best_value = value
                                best_optimizer = optimizer_name
                        else:
                            if value < best_value:
                                best_value = value
                                best_optimizer = optimizer_name
                
                if best_optimizer:
                    f.write(f"- **{category_name}**: {best_optimizer} ({best_value:.2f})\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on benchmark results:\n\n")
            f.write("1. **For maximum speed**: Use ULTRON_V2 with vectorized updates\n")
            f.write("2. **For memory efficiency**: Use ULTRON_V2 with low-precision state buffers\n")
            f.write("3. **For best convergence**: Test different normalization strategies\n")
            f.write("4. **For deep networks**: Use adaptive clipping for stability\n")
            f.write("5. **For production deployment**: Enable TorchScript compilation\n")
            
            # Files generated
            f.write("\n## Generated Files\n\n")
            f.write("The following files have been generated:\n\n")
            for optimizer_name in self.optimizer_registry.keys():
                if optimizer_name in all_results['speed_comparison']:
                    f.write(f"- `{optimizer_name}_speed_results.json`: Speed benchmark results\n")
                    f.write(f"- `{optimizer_name}_memory_results.json`: Memory usage results\n")
                    f.write(f"- `{optimizer_name}_convergence_results.json`: Convergence results\n")
            f.write(f"- `ultron_v2_comparison_report.md`: This report\n")
            f.write(f"- `plots/`: Directory containing performance plots\n")
        
        print(f"\nComparison report saved to: {report_path}")
        
        # Generate plots
        self._generate_comparison_plots(all_results, model_types)
    
    def _generate_comparison_plots(self, all_results: Dict[str, Any], model_types: List[str]):
        """Generate comparison plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
            
            # Plot 1: Speed comparison
            fig, axes = plt.subplots(1, len(model_types), figsize=(15, 6))
            if len(model_types) == 1:
                axes = [axes]
            
            for idx, model_type in enumerate(model_types):
                ax = axes[idx]
                
                optimizer_names = []
                throughputs = []
                
                for optimizer_name in self.optimizer_registry.keys():
                    if (optimizer_name in all_results['speed_comparison'] and 
                        model_type in all_results['speed_comparison'][optimizer_name]):
                        
                        optimizer_names.append(optimizer_name)
                        throughput = all_results['speed_comparison'][optimizer_name][model_type]['throughput_samples_per_sec']
                        throughputs.append(throughput)
                
                # Sort by throughput
                sorted_indices = np.argsort(throughputs)[::-1]
                sorted_names = [optimizer_names[i] for i in sorted_indices]
                sorted_throughputs = [throughputs[i] for i in sorted_indices]
                
                x = np.arange(len(sorted_names))
                bars = ax.bar(x, sorted_throughputs, color=sns.color_palette("husl", len(sorted_names)))
                
                # Highlight ULTRON_V2
                for i, name in enumerate(sorted_names):
                    if name == 'ULTRON_V2':
                        bars[i].set_color('red')
                        bars[i].set_edgecolor('black')
                        bars[i].set_linewidth(2)
                
                ax.set_xlabel('Optimizer')
                ax.set_ylabel('Throughput (samples/sec)')
                ax.set_title(f'Speed Comparison - {model_type}')
                ax.set_xticks(x)
                ax.set_xticklabels(sorted_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'speed_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Memory comparison
            fig, axes = plt.subplots(1, len(model_types), figsize=(15, 6))
            if len(model_types) == 1:
                axes = [axes]
            
            for idx, model_type in enumerate(model_types):
                ax = axes[idx]
                
                optimizer_names = []
                memories = []
                
                for optimizer_name in self.optimizer_registry.keys():
                    if (optimizer_name in all_results['memory_comparison'] and 
                        model_type in all_results['memory_comparison'][optimizer_name]):
                        
                        optimizer_names.append(optimizer_name)
                        memory = all_results['memory_comparison'][optimizer_name][model_type]['total_mb']
                        memories.append(memory)
                
                # Sort by memory usage (lower is better)
                sorted_indices = np.argsort(memories)
                sorted_names = [optimizer_names[i] for i in sorted_indices]
                sorted_memories = [memories[i] for i in sorted_indices]
                
                x = np.arange(len(sorted_names))
                bars = ax.bar(x, sorted_memories, color=sns.color_palette("husl", len(sorted_names)))
                
                # Highlight ULTRON_V2
                for i, name in enumerate(sorted_names):
                    if name == 'ULTRON_V2':
                        bars[i].set_color('green')
                        bars[i].set_edgecolor('black')
                        bars[i].set_linewidth(2)
                
                ax.set_xlabel('Optimizer')
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title(f'Memory Comparison - {model_type}')
                ax.set_xticks(x)
                ax.set_xticklabels(sorted_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'memory_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 3: Convergence comparison
            fig, axes = plt.subplots(1, len(model_types), figsize=(15, 6))
            if len(model_types) == 1:
                axes = [axes]
            
            for idx, model_type in enumerate(model_types):
                ax = axes[idx]
                
                for optimizer_name in self.optimizer_registry.keys():
                    if (optimizer_name in all_results['convergence_comparison'] and 
                        model_type in all_results['convergence_comparison'][optimizer_name]):
                        
                        test_accuracies = all_results['convergence_comparison'][optimizer_name][model_type]['test_accuracies']
                        epochs = range(1, len(test_accuracies) + 1)
                        
                        linewidth = 3 if optimizer_name == 'ULTRON_V2' else 2
                        alpha = 1.0 if optimizer_name == 'ULTRON_V2' else 0.7
                        
                        ax.plot(epochs, test_accuracies, label=optimizer_name, 
                               linewidth=linewidth, alpha=alpha)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Test Accuracy (%)')
                ax.set_title(f'Convergence Comparison - {model_type}')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'convergence_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plots saved to: {plots_dir}/")
            
        except ImportError:
            print("Warning: matplotlib not installed, skipping plot generation")
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
    
    def _save_results(self, all_results: Dict[str, Any]):
        """Save benchmark results."""
        # Save individual optimizer results
        for optimizer_name in self.optimizer_registry.keys():
            if optimizer_name in all_results['speed_comparison']:
                # Speed results
                speed_file = self.output_dir / f'{optimizer_name}_speed_results.json'
                with open(speed_file, 'w') as f:
                    json.dump(all_results['speed_comparison'][optimizer_name], f, indent=2, default=str)
                
                # Memory results
                memory_file = self.output_dir / f'{optimizer_name}_memory_results.json'
                with open(memory_file, 'w') as f:
                    json.dump(all_results['memory_comparison'][optimizer_name], f, indent=2, default=str)
                
                # Convergence results
                convergence_file = self.output_dir / f'{optimizer_name}_convergence_results.json'
                with open(convergence_file, 'w') as f:
                    json.dump(all_results['convergence_comparison'][optimizer_name], f, indent=2, default=str)
        
        # Save complete results
        complete_file = self.output_dir / 'complete_benchmark_results.json'
        with open(complete_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {self.output_dir}/")
    
    def run_benchmark(self):
        """Run complete benchmark suite."""
        print(f"\n{'='*80}")
        print("Running ULTRON_V2 Comprehensive Benchmark")
        print(f"{'='*80}")
        
        # Run comparison
        results = self.compare_all_optimizers(
            model_types=['simple_cnn', 'cnn_v2', 'cnn_v3'],
            batch_size=64,
            num_iterations=500,
            num_epochs=5
        )
        
        print(f"\n{'='*80}")
        print("Benchmark completed successfully!")
        print(f"Results saved to: {self.output_dir}/")
        print(f"{'='*80}")
        
        return results


def main():
    """Main function to run benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ULTRON_V2 benchmark')
    parser.add_argument('--output', type=str, default='ultron_v2_benchmark',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = ULTRONV2Benchmark(output_dir=args.output, device=args.device)
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
