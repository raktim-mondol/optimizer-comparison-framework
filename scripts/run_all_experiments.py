#!/usr/bin/env python3
"""
Main script to run all experiments for optimizer comparison.
This script orchestrates the entire experimental pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.comprehensive_benchmark import ComprehensiveBenchmark
from benchmarks.memory_profiler import MemoryProfiler
from benchmarks.speed_benchmark import SpeedBenchmark
from benchmarks.optimizer_comparison import OptimizerComparison


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive optimizer comparison experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (MNIST CNN, MNIST ViT, CIFAR10 CNN, CIFAR10 ViT)
  python scripts/run_all_experiments.py
  
  # Run specific configuration
  python scripts/run_all_experiments.py --config experiments/configs/mnist_cnn.yaml
  
  # Run full benchmark suite (experiments + speed + memory tests)
  python scripts/run_all_experiments.py --full
  
  # Run with specific GPU
  python scripts/run_all_experiments.py --gpu 0
  
  # Run with custom output directory
  python scripts/run_all_experiments.py --output my_results
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Specific configuration file to run (default: run all)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--full', action='store_true',
                       help='Run full benchmark suite including speed and memory tests')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory for results (default: benchmark_results)')
    parser.add_argument('--memory-batch-size', type=int, default=64,
                       help='Batch size for memory profiling (default: 64)')
    parser.add_argument('--memory-iterations', type=int, default=100,
                       help='Iterations for memory profiling (default: 100)')
    parser.add_argument('--speed-batch-size', type=int, default=64,
                       help='Batch size for speed benchmarking (default: 64)')
    parser.add_argument('--speed-iterations', type=int, default=1000,
                       help='Iterations for speed benchmarking (default: 1000)')
    parser.add_argument('--speed-warmup', type=int, default=100,
                       help='Warmup iterations for speed benchmarking (default: 100)')
    parser.add_argument('--optimizer-iterations', type=int, default=1000,
                       help='Iterations for optimizer comparison (default: 1000)')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip the main experiments (only run benchmarks)')
    parser.add_argument('--skip-benchmarks', action='store_true',
                       help='Skip benchmarks (only run experiments)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Optimizer Comparison Framework")
    print("="*80)
    print(f"Output directory: {args.output}")
    print(f"GPU device: {args.gpu}")
    print(f"Full benchmark suite: {args.full}")
    print(f"Skip experiments: {args.skip_experiments}")
    print(f"Skip benchmarks: {args.skip_benchmarks}")
    print("="*80 + "\n")
    
    import torch
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_results = {}
    
    # Run experiments if not skipped
    if not args.skip_experiments:
        print("\n" + "="*80)
        print("Running Main Experiments")
        print("="*80)
        
        benchmark = ComprehensiveBenchmark(output_dir=str(output_dir / 'experiments'), device=device)
        
        if args.config:
            # Run specific configuration
            config_files = [args.config]
        else:
            # Run all default configurations
            config_files = [
                'experiments/configs/mnist_cnn.yaml',
                'experiments/configs/mnist_vit.yaml',
                'experiments/configs/cifar10_cnn.yaml',
                'experiments/configs/cifar10_vit.yaml',
            ]
        
        experiment_results = benchmark.run_all_experiments(config_files)
        all_results['experiments'] = experiment_results
    
    # Run benchmarks if not skipped and full suite requested
    if not args.skip_benchmarks and args.full:
        print("\n" + "="*80)
        print("Running Benchmarks")
        print("="*80)
        
        # Memory profiling
        print("\n1. Memory Profiling")
        print("-"*40)
        memory_profiler = MemoryProfiler(
            output_dir=str(output_dir / 'memory_benchmark'),
            device=device
        )
        memory_results = memory_profiler.profile_all_optimizers(
            batch_size=args.memory_batch_size,
            num_iterations=args.memory_iterations
        )
        all_results['memory_benchmark'] = memory_results
        
        # Speed benchmarking
        print("\n2. Speed Benchmarking")
        print("-"*40)
        speed_benchmark = SpeedBenchmark(
            output_dir=str(output_dir / 'speed_benchmark'),
            device=device
        )
        speed_results = speed_benchmark.benchmark_all_optimizers(
            batch_size=args.speed_batch_size,
            num_iterations=args.speed_iterations,
            warmup_iterations=args.speed_warmup
        )
        all_results['speed_benchmark'] = speed_results
        
        # Optimizer comparison on synthetic functions
        print("\n3. Optimizer Comparison on Synthetic Functions")
        print("-"*40)
        optimizer_comparison = OptimizerComparison(
            output_dir=str(output_dir / 'optimizer_comparison')
        )
        comparison_results = optimizer_comparison.run_all_comparisons(
            num_iterations=args.optimizer_iterations
        )
        all_results['optimizer_comparison'] = comparison_results
    
    # Generate final summary report
    print("\n" + "="*80)
    print("Generating Final Summary Report")
    print("="*80)
    
    # Save all results
    import json
    results_file = output_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        # Convert to serializable format
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return str(obj)
        
        serializable_results = make_serializable(all_results)
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nAll results saved to: {results_file}")
    
    # Generate summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Optimizer Comparison Framework - Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Date: {import datetime; f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
        f.write(f"Device: {device}\n")
        if device.type == 'cuda':
            f.write(f"GPU: {torch.cuda.get_device_name(args.gpu)}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        if 'experiments' in all_results:
            f.write("Experiments Completed:\n")
            f.write("-"*40 + "\n")
            for exp_name in all_results['experiments'].keys():
                f.write(f"  {exp_name}\n")
            f.write("\n")
        
        if 'memory_benchmark' in all_results:
            f.write("Memory Benchmark Completed\n")
            f.write("-"*40 + "\n")
            f.write(f"  Batch size: {args.memory_batch_size}\n")
            f.write(f"  Iterations: {args.memory_iterations}\n")
            f.write("\n")
        
        if 'speed_benchmark' in all_results:
            f.write("Speed Benchmark Completed\n")
            f.write("-"*40 + "\n")
            f.write(f"  Batch size: {args.speed_batch_size}\n")
            f.write(f"  Iterations: {args.speed_iterations}\n")
            f.write(f"  Warmup iterations: {args.speed_warmup}\n")
            f.write("\n")
        
        if 'optimizer_comparison' in all_results:
            f.write("Optimizer Comparison Completed\n")
            f.write("-"*40 + "\n")
            f.write(f"  Iterations per function: {args.optimizer_iterations}\n")
            f.write("\n")
        
        f.write("Files Generated:\n")
        f.write("-"*40 + "\n")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(str(output_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")
    
    print(f"Summary saved to: {summary_file}")
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)
    
    # Print quick summary
    print("\nQuick Summary:")
    print("-"*40)
    if 'experiments' in all_results:
        print(f"Experiments run: {len(all_results['experiments'])}")
    if 'memory_benchmark' in all_results:
        print(f"Memory benchmarks: {len(all_results['memory_benchmark'])} optimizers")
    if 'speed_benchmark' in all_results:
        print(f"Speed benchmarks: {len(all_results['speed_benchmark'])} optimizers")
    if 'optimizer_comparison' in all_results:
        print(f"Function comparisons: {len(all_results['optimizer_comparison'])} functions")
    
    print(f"\nResults saved in: {output_dir}")
    print("\nTo view results:")
    print(f"  - Check Excel files in: {output_dir}/experiments/")
    print(f"  - Check reports in: {output_dir}/")
    print(f"  - Check plots in: {output_dir}/*/plots/")
    
    return all_results


if __name__ == '__main__':
    main()