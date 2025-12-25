"""
Speed benchmarking for different optimizers.
Measures training and inference speed.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
import gc

# Import optimizers
import sys
sys.path.append('..')
from optimizers.amcas import AMCAS


class SpeedBenchmark:
    """
    Benchmarks training and inference speed of different optimizers.
    """
    
    def __init__(self, output_dir='speed_benchmark', device=None):
        """
        Initialize speed benchmark.
        
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
        
        # Available optimizers
        self.optimizer_registry = {
            'AMCAS': AMCAS,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'SGD+Momentum': lambda params, lr, **kwargs: torch.optim.SGD(params, lr=lr, momentum=0.9, **kwargs),
            'RMSprop': torch.optim.RMSprop,
            'Adagrad': torch.optim.Adagrad,
            'Adadelta': torch.optim.Adadelta,
            'NAdam': torch.optim.NAdam,
            'RAdam': torch.optim.RAdam,
        }
        
        # Default optimizer parameters
        self.default_optimizer_params = {
            'AMCAS': {'betas': (0.9, 0.999), 'gamma': 0.1, 'lambda_consistency': 0.01},
            'Adam': {'betas': (0.9, 0.999)},
            'AdamW': {'betas': (0.9, 0.999), 'weight_decay': 0.01},
            'SGD': {},
            'SGD+Momentum': {},  # momentum is already set in the lambda function
            'RMSprop': {},
            'Adagrad': {},
            'Adadelta': {},
            'NAdam': {'betas': (0.9, 0.999)},
            'RAdam': {'betas': (0.9, 0.999)},
        }
        
        print(f"Speed benchmark initialized with device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def create_test_model(self, input_size: tuple = (3, 32, 32), num_classes: int = 10) -> nn.Module:
        """
        Create a test model for speed benchmarking.
        
        Args:
            input_size: Input tensor size (channels, height, width)
            num_classes: Number of output classes
            
        Returns:
            PyTorch model
        """
        channels, height, width = input_size
        
        model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fully connected layers
            nn.Flatten(),
            nn.Linear(256 * (height // 8) * (width // 8), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
        return model
    
    def benchmark_optimizer_speed(self, optimizer_name: str, batch_size: int = 64, 
                                 num_iterations: int = 1000, warmup_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark training speed for a specific optimizer.
        
        Args:
            optimizer_name: Name of the optimizer to benchmark
            batch_size: Batch size for benchmarking
            num_iterations: Number of training iterations to benchmark
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with speed benchmarking results
        """
        print(f"\nBenchmarking speed for {optimizer_name}...")
        
        # Create model and move to device
        model = self.create_test_model().to(self.device)
        
        # Create optimizer
        optimizer_class = self.optimizer_registry[optimizer_name]
        optimizer_params = self.default_optimizer_params.get(optimizer_name, {}).copy()
        
        if optimizer_name == 'SGD+Momentum':
            optimizer = optimizer_class(model.parameters(), lr=0.001, **optimizer_params)
        else:
            optimizer = optimizer_class(model.parameters(), lr=0.001, **optimizer_params)
        
        # Create dummy data
        input_size = (batch_size, 3, 32, 32)
        dummy_input = torch.randn(input_size).to(self.device)
        dummy_target = torch.randint(0, 10, (batch_size,)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Speed tracking
        speed_stats = {
            'optimizer_name': optimizer_name,
            'batch_size': batch_size,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'forward_times': [],
            'backward_times': [],
            'optimizer_step_times': [],
            'total_training_times': [],
            'inference_times': [],
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
        
        # Benchmark inference
        print(f"  Benchmarking inference for {num_iterations} iterations...")
        model.eval()
        
        with torch.no_grad():
            for iteration in range(num_iterations):
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                inference_start = time.perf_counter()
                
                _ = model(dummy_input)
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                inference_end = time.perf_counter()
                inference_time = inference_end - inference_start
                
                speed_stats['inference_times'].append(inference_time)
        
        # Calculate throughput
        total_training_time = sum(speed_stats['total_training_times'])
        total_inference_time = sum(speed_stats['inference_times'])
        
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
        speed_stats['inference_time_mean'] = np.mean(speed_stats['inference_times'])
        speed_stats['inference_time_std'] = np.std(speed_stats['inference_times'])
        
        # Clean up
        del model, optimizer, dummy_input, dummy_target
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  Completed: Throughput = {speed_stats['throughput_samples_per_sec']:.2f} samples/sec")
        
        return speed_stats
    
    def benchmark_all_optimizers(self, batch_size: int = 64, num_iterations: int = 1000, 
                                warmup_iterations: int = 100) -> Dict[str, Dict]:
        """
        Benchmark training speed for all optimizers.
        
        Args:
            batch_size: Batch size for benchmarking
            num_iterations: Number of training iterations to benchmark
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary mapping optimizer names to speed stats
        """
        print(f"\n{'='*80}")
        print("Speed Benchmarking for All Optimizers")
        print(f"{'='*80}")
        
        all_results = {}
        
        for optimizer_name in self.optimizer_registry.keys():
            try:
                result = self.benchmark_optimizer_speed(
                    optimizer_name, batch_size, num_iterations, warmup_iterations
                )
                all_results[optimizer_name] = result
                
                # Save individual results
                self._save_optimizer_results(optimizer_name, result)
                
            except Exception as e:
                print(f"Error benchmarking {optimizer_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[optimizer_name] = {'error': str(e)}
        
        # Generate comparison report
        self._generate_speed_comparison_report(all_results, batch_size, num_iterations)
        
        return all_results
    
    def _save_optimizer_results(self, optimizer_name: str, result: Dict[str, Any]):
        """Save individual optimizer results."""
        result_file = self.output_dir / f'{optimizer_name}_speed_benchmark.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = self._make_serializable(result)
        
        with open(result_file, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
        
        print(f"  Results saved to: {result_file}")
    
    def _generate_speed_comparison_report(self, results: Dict[str, Dict], batch_size: int, num_iterations: int):
        """Generate speed comparison report."""
        report_path = self.output_dir / 'speed_comparison_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Speed Benchmarking Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Iterations: {num_iterations}\n")
            f.write(f"Device: {self.device}\n\n")
            
            if self.device.type == 'cuda':
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n\n")
            
            # Summary table
            f.write("## Speed Performance Summary\n\n")
            f.write("| Optimizer | Throughput (samples/sec) | Throughput (iter/sec) | Forward Time (ms) | Backward Time (ms) | Optimizer Step (ms) | Total Time (ms) |\n")
            f.write("|-----------|-------------------------|----------------------|------------------|-------------------|-------------------|----------------|\n")
            
            for optimizer_name, result in results.items():
                if 'error' in result:
                    f.write(f"| {optimizer_name} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |\n")
                else:
                    throughput_samples = result.get('throughput_samples_per_sec', 0)
                    throughput_iter = result.get('throughput_iterations_per_sec', 0)
                    forward_time = result.get('forward_time_mean', 0) * 1000  # Convert to ms
                    backward_time = result.get('backward_time_mean', 0) * 1000
                    optimizer_time = result.get('optimizer_step_time_mean', 0) * 1000
                    total_time = result.get('total_training_time_mean', 0) * 1000
                    
                    f.write(f"| {optimizer_name} | {throughput_samples:.2f} | {throughput_iter:.2f} | {forward_time:.3f} | {backward_time:.3f} | {optimizer_time:.3f} | {total_time:.3f} |\n")
            
            f.write("\n")
            
            # Detailed analysis
            f.write("## Detailed Analysis\n\n")
            
            # Sort by throughput (samples/sec)
            sorted_results = sorted(
                [(name, res) for name, res in results.items() if 'error' not in res],
                key=lambda x: x[1].get('throughput_samples_per_sec', 0),
                reverse=True
            )
            
            f.write("### Fastest Optimizers (Highest Throughput)\n\n")
            for i, (optimizer_name, result) in enumerate(sorted_results[:5]):
                throughput_samples = result.get('throughput_samples_per_sec', 0)
                throughput_iter = result.get('throughput_iterations_per_sec', 0)
                total_time = result.get('total_training_time_mean', 0) * 1000
                
                f.write(f"{i+1}. **{optimizer_name}**: Throughput: {throughput_samples:.2f} samples/sec "
                       f"({throughput_iter:.2f} iter/sec), Total Time: {total_time:.3f} ms/iter\n")
            
            f.write("\n")
            
            # Sort by total training time (fastest)
            sorted_results_time = sorted(
                [(name, res) for name, res in results.items() if 'error' not in res],
                key=lambda x: x[1].get('total_training_time_mean', float('inf'))
            )
            
            f.write("### Fastest Optimizers (Lowest Training Time)\n\n")
            for i, (optimizer_name, result) in enumerate(sorted_results_time[:5]):
                total_time = result.get('total_training_time_mean', 0) * 1000
                forward_time = result.get('forward_time_mean', 0) * 1000
                backward_time = result.get('backward_time_mean', 0) * 1000
                optimizer_time = result.get('optimizer_step_time_mean', 0) * 1000
                
                f.write(f"{i+1}. **{optimizer_name}**: Total: {total_time:.3f} ms, "
                       f"Forward: {forward_time:.3f} ms, Backward: {backward_time:.3f} ms, "
                       f"Optimizer: {optimizer_time:.3f} ms\n")
            
            f.write("\n")
            
            # Breakdown of time spent
            f.write("### Time Breakdown Analysis\n\n")
            f.write("Percentage of time spent in each phase:\n\n")
            
            for optimizer_name, result in sorted_results[:5]:  # Top 5 by throughput
                if 'error' in result:
                    continue
                
                total_time = result.get('total_training_time_mean', 1e-6)  # Avoid division by zero
                forward_pct = (result.get('forward_time_mean', 0) / total_time) * 100
                backward_pct = (result.get('backward_time_mean', 0) / total_time) * 100
                optimizer_pct = (result.get('optimizer_step_time_mean', 0) / total_time) * 100
                
                f.write(f"- **{optimizer_name}**: Forward: {forward_pct:.1f}%, "
                       f"Backward: {backward_pct:.1f}%, Optimizer: {optimizer_pct:.1f}%\n")
            
            f.write("\n")
            
            # Inference performance
            f.write("### Inference Performance\n\n")
            f.write("| Optimizer | Inference Time (ms) | Inference Throughput (samples/sec) |\n")
            f.write("|-----------|-------------------|-----------------------------------|\n")
            
            for optimizer_name, result in sorted_results[:5]:  # Top 5 by throughput
                if 'error' in result:
                    f.write(f"| {optimizer_name} | ERROR | ERROR |\n")
                else:
                    inference_time = result.get('inference_time_mean', 0) * 1000
                    inference_throughput = batch_size / result.get('inference_time_mean', 1e-6)
                    
                    f.write(f"| {optimizer_name} | {inference_time:.3f} | {inference_throughput:.2f} |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on speed benchmarking results:\n\n")
            
            if sorted_results:
                fastest = sorted_results[0][0]
                slowest = sorted_results[-1][0]
                
                f.write(f"1. **Fastest optimizer**: {fastest}\n")
                f.write(f"2. **Slowest optimizer**: {slowest}\n")
                f.write("3. **For training speed**: Choose optimizers with higher throughput\n")
                f.write("4. **For real-time applications**: Consider inference time as well\n")
                f.write("5. **For large-scale training**: Throughput (samples/sec) is critical\n")
                f.write("6. **For small batch sizes**: Optimizer step time becomes more important\n")
                f.write("7. **For large models**: Backward pass time dominates\n")
            
            # Files generated
            f.write("\n## Generated Files\n\n")
            f.write("The following files have been generated:\n\n")
            for optimizer_name in results.keys():
                f.write(f"- `{optimizer_name}_speed_benchmark.json`: Detailed speed benchmark\n")
            f.write(f"- `speed_comparison_report.md`: This report\n")
            f.write(f"- `speed_performance_plots.png`: Speed performance plots (if generated)\n")
        
        print(f"\nSpeed comparison report saved to: {report_path}")
        
        # Generate plots
        self._generate_speed_plots(results)
        
        return report_path
    
    def _generate_speed_plots(self, results: Dict[str, Dict]):
        """Generate speed performance plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Plot 1: Throughput comparison
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            optimizer_names = []
            throughput_samples = []
            throughput_iter = []
            total_times = []
            
            for optimizer_name, result in results.items():
                if 'error' in result:
                    continue
                
                optimizer_names.append(optimizer_name)
                throughput_samples.append(result.get('throughput_samples_per_sec', 0))
                throughput_iter.append(result.get('throughput_iterations_per_sec', 0))
                total_times.append(result.get('total_training_time_mean', 0) * 1000)  # Convert to ms
            
            if optimizer_names:
                x = np.arange(len(optimizer_names))
                width = 0.35
                
                # Throughput plot
                axes[0].bar(x - width/2, throughput_samples, width, label='Samples/sec', color='skyblue')
                axes[0].set_xlabel('Optimizer')
                axes[0].set_ylabel('Throughput (samples/sec)')
                axes[0].set_title('Training Throughput Comparison')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(optimizer_names, rotation=45, ha='right')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Total time plot
                axes[1].bar(x - width/2, total_times, width, label='Training Time', color='lightcoral')
                axes[1].set_xlabel('Optimizer')
                axes[1].set_ylabel('Training Time per Iteration (ms)')
                axes[1].set_title('Training Time Comparison')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(optimizer_names, rotation=45, ha='right')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'throughput_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Plot 2: Time breakdown for top 5 optimizers
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Get top 5 optimizers by throughput
                sorted_optimizers = sorted(
                    [(name, res) for name, res in results.items() if 'error' not in res],
                    key=lambda x: x[1].get('throughput_samples_per_sec', 0),
                    reverse=True
                )[:5]
                
                optimizer_names = [name for name, _ in sorted_optimizers]
                forward_times = []
                backward_times = []
                optimizer_times = []
                
                for _, result in sorted_optimizers:
                    forward_times.append(result.get('forward_time_mean', 0) * 1000)
                    backward_times.append(result.get('backward_time_mean', 0) * 1000)
                    optimizer_times.append(result.get('optimizer_step_time_mean', 0) * 1000)
                
                x = np.arange(len(optimizer_names))
                width = 0.25
                
                ax.bar(x - width, forward_times, width, label='Forward Pass', color='lightblue')
                ax.bar(x, backward_times, width, label='Backward Pass', color='lightgreen')
                ax.bar(x + width, optimizer_times, width, label='Optimizer Step', color='lightcoral')
                
                ax.set_xlabel('Optimizer')
                ax.set_ylabel('Time (ms)')
                ax.set_title('Time Breakdown for Top 5 Optimizers')
                ax.set_xticks(x)
                ax.set_xticklabels(optimizer_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'time_breakdown.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Plot 3: Inference time comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                
                optimizer_names = []
                inference_times = []
                
                for optimizer_name, result in results.items():
                    if 'error' in result:
                        continue
                    
                    optimizer_names.append(optimizer_name)
                    inference_times.append(result.get('inference_time_mean', 0) * 1000)
                
                if optimizer_names:
                    x = np.arange(len(optimizer_names))
                    
                    ax.bar(x, inference_times, color='purple', alpha=0.7)
                    ax.set_xlabel('Optimizer')
                    ax.set_ylabel('Inference Time (ms)')
                    ax.set_title('Inference Time Comparison')
                    ax.set_xticks(x)
                    ax.set_xticklabels(optimizer_names, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'inference_comparison.png', dpi=150, bbox_inches='tight')
                    plt.close()
                
                print(f"Speed plots saved to: {plots_dir}/")
                
        except ImportError:
            print("Warning: matplotlib not installed, skipping plot generation")
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return obj


def main():
    """Main function to run speed benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark speed of optimizers')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for benchmarking (default: 64)')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of iterations (default: 1000)')
    parser.add_argument('--warmup', type=int, default=100,
                       help='Number of warmup iterations (default: 100)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--output', type=str, default='speed_benchmark',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # Create benchmark
    benchmark = SpeedBenchmark(output_dir=args.output, device=device)
    
    # Run benchmarking
    results = benchmark.benchmark_all_optimizers(
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    print(f"\nSpeed benchmarking completed!")
    print(f"Results saved to: {args.output}/")


if __name__ == '__main__':
    main()