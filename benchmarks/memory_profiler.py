"""
Memory profiling for different optimizers.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import gc

# Import optimizers
import sys
sys.path.append('..')
from optimizers.amcas import AMCAS
from optimizers.ultron import ULTRON


class MemoryProfiler:
    """
    Profiles memory usage of different optimizers during training.
    """
    
    def __init__(self, output_dir='memory_profiling', device=None):
        """
        Initialize memory profiler.
        
        Args:
            output_dir: Directory to save profiling results
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
            'ULTRON': ULTRON,
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
            'ULTRON': {'betas': (0.9, 0.999), 'clip_threshold': 1.0, 'normalize_gradients': True},
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
        
        print(f"Memory profiler initialized with device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def create_test_model(self, input_size: tuple = (3, 32, 32), num_classes: int = 10) -> nn.Module:
        """
        Create a test model for memory profiling.
        
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
    
    def profile_optimizer_memory(self, optimizer_name: str, batch_size: int = 64, 
                                 num_iterations: int = 100) -> Dict[str, Any]:
        """
        Profile memory usage for a specific optimizer.
        
        Args:
            optimizer_name: Name of the optimizer to profile
            batch_size: Batch size for profiling
            num_iterations: Number of training iterations to profile
            
        Returns:
            Dictionary with memory profiling results
        """
        print(f"\nProfiling memory for {optimizer_name}...")
        
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
        
        # Memory tracking
        memory_stats = {
            'optimizer_name': optimizer_name,
            'batch_size': batch_size,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'memory_before_training': {},
            'memory_during_training': [],
            'memory_after_training': {},
            'peak_memory': 0,
            'average_memory': 0,
        }
        
        # Record memory before training
        self._record_memory(memory_stats['memory_before_training'])
        
        # Training loop with memory tracking
        model.train()
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Record memory after each iteration
            iteration_memory = {}
            self._record_memory(iteration_memory)
            memory_stats['memory_during_training'].append(iteration_memory)
            
            # Update peak memory
            current_memory = iteration_memory.get('gpu_memory_allocated_gb', 0)
            memory_stats['peak_memory'] = max(memory_stats['peak_memory'], current_memory)
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}/{num_iterations}: "
                      f"GPU Memory: {current_memory:.2f} GB, "
                      f"CPU Memory: {iteration_memory.get('cpu_memory_gb', 0):.2f} GB")
        
        # Record memory after training
        self._record_memory(memory_stats['memory_after_training'])
        
        # Calculate average memory
        if memory_stats['memory_during_training']:
            gpu_memories = [m.get('gpu_memory_allocated_gb', 0) for m in memory_stats['memory_during_training']]
            cpu_memories = [m.get('cpu_memory_gb', 0) for m in memory_stats['memory_during_training']]
            memory_stats['average_gpu_memory'] = np.mean(gpu_memories)
            memory_stats['average_cpu_memory'] = np.mean(cpu_memories)
            memory_stats['std_gpu_memory'] = np.std(gpu_memories)
            memory_stats['std_cpu_memory'] = np.std(cpu_memories)
        
        # Clean up
        del model, optimizer, dummy_input, dummy_target
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return memory_stats
    
    def _record_memory(self, memory_dict: Dict[str, float]):
        """Record current memory usage."""
        # CPU memory
        process = psutil.Process()
        memory_dict['cpu_memory_gb'] = process.memory_info().rss / 1e9
        
        # GPU memory
        if self.device.type == 'cuda':
            memory_dict['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated(self.device) / 1e9
            memory_dict['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(self.device) / 1e9
            memory_dict['gpu_memory_cached_gb'] = torch.cuda.memory_cached(self.device) / 1e9
    
    def profile_all_optimizers(self, batch_size: int = 64, num_iterations: int = 100) -> Dict[str, Dict]:
        """
        Profile memory usage for all optimizers.
        
        Args:
            batch_size: Batch size for profiling
            num_iterations: Number of training iterations to profile
            
        Returns:
            Dictionary mapping optimizer names to memory stats
        """
        print(f"\n{'='*80}")
        print("Memory Profiling for All Optimizers")
        print(f"{'='*80}")
        
        all_results = {}
        
        for optimizer_name in self.optimizer_registry.keys():
            try:
                result = self.profile_optimizer_memory(optimizer_name, batch_size, num_iterations)
                all_results[optimizer_name] = result
                
                # Save individual results
                self._save_optimizer_results(optimizer_name, result)
                
            except Exception as e:
                print(f"Error profiling {optimizer_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[optimizer_name] = {'error': str(e)}
        
        # Generate comparison report
        self._generate_memory_comparison_report(all_results, batch_size, num_iterations)
        
        return all_results
    
    def _save_optimizer_results(self, optimizer_name: str, result: Dict[str, Any]):
        """Save individual optimizer results."""
        result_file = self.output_dir / f'{optimizer_name}_memory_profile.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = self._make_serializable(result)
        
        with open(result_file, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
        
        print(f"  Results saved to: {result_file}")
    
    def _generate_memory_comparison_report(self, results: Dict[str, Dict], batch_size: int, num_iterations: int):
        """Generate memory comparison report."""
        report_path = self.output_dir / 'memory_comparison_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Memory Usage Comparison Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Iterations: {num_iterations}\n")
            f.write(f"Device: {self.device}\n\n")
            
            if self.device.type == 'cuda':
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n\n")
            
            # Summary table
            f.write("## Memory Usage Summary\n\n")
            f.write("| Optimizer | Peak GPU Memory (GB) | Avg GPU Memory (GB) | Peak CPU Memory (GB) | Avg CPU Memory (GB) | Model Params |\n")
            f.write("|-----------|---------------------|-------------------|---------------------|-------------------|--------------|\n")
            
            for optimizer_name, result in results.items():
                if 'error' in result:
                    f.write(f"| {optimizer_name} | ERROR | ERROR | ERROR | ERROR | ERROR |\n")
                else:
                    peak_gpu = result.get('peak_memory', 0)
                    avg_gpu = result.get('average_gpu_memory', 0)
                    peak_cpu = max([m.get('cpu_memory_gb', 0) for m in result.get('memory_during_training', [])], default=0)
                    avg_cpu = result.get('average_cpu_memory', 0)
                    params = result.get('model_parameters', 0)
                    
                    f.write(f"| {optimizer_name} | {peak_gpu:.3f} | {avg_gpu:.3f} | {peak_cpu:.3f} | {avg_cpu:.3f} | {params:,} |\n")
            
            f.write("\n")
            
            # Detailed analysis
            f.write("## Detailed Analysis\n\n")
            
            # Sort by peak GPU memory
            sorted_results = sorted(
                [(name, res) for name, res in results.items() if 'error' not in res],
                key=lambda x: x[1].get('peak_memory', float('inf'))
            )
            
            f.write("### Most Memory Efficient (Lowest Peak GPU Memory)\n\n")
            for i, (optimizer_name, result) in enumerate(sorted_results[:3]):
                peak_gpu = result.get('peak_memory', 0)
                avg_gpu = result.get('average_gpu_memory', 0)
                params = result.get('model_parameters', 0)
                
                f.write(f"{i+1}. **{optimizer_name}**: Peak GPU Memory: {peak_gpu:.3f} GB, "
                       f"Average GPU Memory: {avg_gpu:.3f} GB, Parameters: {params:,}\n")
            
            f.write("\n")
            
            # Sort by average GPU memory
            sorted_results_avg = sorted(
                [(name, res) for name, res in results.items() if 'error' not in res],
                key=lambda x: x[1].get('average_gpu_memory', float('inf'))
            )
            
            f.write("### Most Memory Efficient (Lowest Average GPU Memory)\n\n")
            for i, (optimizer_name, result) in enumerate(sorted_results_avg[:3]):
                peak_gpu = result.get('peak_memory', 0)
                avg_gpu = result.get('average_gpu_memory', 0)
                params = result.get('model_parameters', 0)
                
                f.write(f"{i+1}. **{optimizer_name}**: Average GPU Memory: {avg_gpu:.3f} GB, "
                       f"Peak GPU Memory: {peak_gpu:.3f} GB, Parameters: {params:,}\n")
            
            f.write("\n")
            
            # Memory usage over time plot data
            f.write("## Memory Usage Over Time\n\n")
            f.write("Memory usage data is available in the JSON files for each optimizer.\n")
            f.write("Key metrics tracked:\n")
            f.write("- GPU Memory Allocated\n")
            f.write("- GPU Memory Reserved\n")
            f.write("- GPU Memory Cached\n")
            f.write("- CPU Memory (RSS)\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on memory profiling results:\n\n")
            
            if sorted_results:
                most_efficient = sorted_results[0][0]
                least_efficient = sorted_results[-1][0]
                
                f.write(f"1. **Most memory efficient**: {most_efficient}\n")
                f.write(f"2. **Least memory efficient**: {least_efficient}\n")
                f.write("3. **For memory-constrained environments**: Choose optimizers with lower peak memory usage\n")
                f.write("4. **For large models**: Consider memory overhead when selecting optimizer\n")
                f.write("5. **For batch training**: Monitor memory usage as batch size increases\n")
            
            # Files generated
            f.write("\n## Generated Files\n\n")
            f.write("The following files have been generated:\n\n")
            for optimizer_name in results.keys():
                f.write(f"- `{optimizer_name}_memory_profile.json`: Detailed memory profile\n")
            f.write(f"- `memory_comparison_report.md`: This report\n")
            f.write(f"- `memory_usage_plots.png`: Memory usage plots (if generated)\n")
        
        print(f"\nMemory comparison report saved to: {report_path}")
        
        # Generate plots
        self._generate_memory_plots(results)
        
        return report_path
    
    def _generate_memory_plots(self, results: Dict[str, Dict]):
        """Generate memory usage plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Plot 1: Peak GPU memory comparison
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            optimizer_names = []
            peak_gpu_memories = []
            avg_gpu_memories = []
            peak_cpu_memories = []
            avg_cpu_memories = []
            
            for optimizer_name, result in results.items():
                if 'error' in result:
                    continue
                
                optimizer_names.append(optimizer_name)
                peak_gpu_memories.append(result.get('peak_memory', 0))
                avg_gpu_memories.append(result.get('average_gpu_memory', 0))
                
                cpu_memories = [m.get('cpu_memory_gb', 0) for m in result.get('memory_during_training', [])]
                peak_cpu_memories.append(max(cpu_memories) if cpu_memories else 0)
                avg_cpu_memories.append(np.mean(cpu_memories) if cpu_memories else 0)
            
            if optimizer_names:
                x = np.arange(len(optimizer_names))
                width = 0.35
                
                # GPU memory plot
                axes[0].bar(x - width/2, peak_gpu_memories, width, label='Peak GPU Memory', color='skyblue')
                axes[0].bar(x + width/2, avg_gpu_memories, width, label='Avg GPU Memory', color='lightcoral')
                axes[0].set_xlabel('Optimizer')
                axes[0].set_ylabel('GPU Memory (GB)')
                axes[0].set_title('GPU Memory Usage Comparison')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(optimizer_names, rotation=45, ha='right')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # CPU memory plot
                axes[1].bar(x - width/2, peak_cpu_memories, width, label='Peak CPU Memory', color='lightgreen')
                axes[1].bar(x + width/2, avg_cpu_memories, width, label='Avg CPU Memory', color='gold')
                axes[1].set_xlabel('Optimizer')
                axes[1].set_ylabel('CPU Memory (GB)')
                axes[1].set_title('CPU Memory Usage Comparison')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(optimizer_names, rotation=45, ha='right')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'memory_usage_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Plot 2: Memory usage over time for top 3 optimizers
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Get top 3 optimizers by efficiency
                sorted_optimizers = sorted(
                    [(name, res) for name, res in results.items() if 'error' not in res],
                    key=lambda x: x[1].get('peak_memory', float('inf'))
                )[:3]
                
                for idx, (optimizer_name, result) in enumerate(sorted_optimizers):
                    row = idx // 2
                    col = idx % 2
                    
                    memory_data = result.get('memory_during_training', [])
                    if memory_data:
                        iterations = list(range(len(memory_data)))
                        gpu_memory = [m.get('gpu_memory_allocated_gb', 0) for m in memory_data]
                        cpu_memory = [m.get('cpu_memory_gb', 0) for m in memory_data]
                        
                        axes[row, col].plot(iterations, gpu_memory, label='GPU Memory', color='blue', linewidth=2)
                        axes[row, col].plot(iterations, cpu_memory, label='CPU Memory', color='red', linewidth=2)
                        axes[row, col].set_xlabel('Iteration')
                        axes[row, col].set_ylabel('Memory (GB)')
                        axes[row, col].set_title(f'{optimizer_name} - Memory Usage Over Time')
                        axes[row, col].legend()
                        axes[row, col].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'memory_usage_over_time.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Memory plots saved to: {plots_dir}/")
                
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
    """Main function to run memory profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile memory usage of optimizers')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for profiling (default: 64)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations (default: 100)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--output', type=str, default='memory_profiling',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # Create profiler
    profiler = MemoryProfiler(output_dir=args.output, device=device)
    
    # Run profiling
    results = profiler.profile_all_optimizers(
        batch_size=args.batch_size,
        num_iterations=args.iterations
    )
    
    print(f"\nMemory profiling completed!")
    print(f"Results saved to: {args.output}/")


if __name__ == '__main__':
    main()