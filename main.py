#!/usr/bin/env python3
"""
Main entry point for running all optimizer comparison experiments.
This script runs comprehensive experiments comparing AMCAS optimizer with other popular optimizers
on MNIST and CIFAR10 datasets using both CNN and Vision Transformer (ViT) architectures.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import json
import yaml
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from experiments.experiment_runner import ExperimentRunner
from experiments.results_exporter import ResultsExporter
from benchmarks.comprehensive_benchmark import ComprehensiveBenchmark
from benchmarks.memory_profiler import MemoryProfiler
from benchmarks.speed_benchmark import SpeedBenchmark
from benchmarks.optimizer_comparison import OptimizerComparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive optimizer comparison experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (MNIST CNN, MNIST ViT, CIFAR10 CNN, CIFAR10 ViT)
  python main.py
  
  # Run specific configuration
  python main.py --config experiments/configs/mnist_cnn.yaml
  
  # Run with full benchmark suite (experiments + speed + memory tests)
  python main.py --full
  
  # Run with specific GPU
  python main.py --gpu 0
  
  # Run with custom output directory
  python main.py --output results
  
  # Run only specific datasets
  python main.py --datasets mnist cifar10
  
  # Run only specific architectures
  python main.py --architectures cnn vit
  
  # Run only specific optimizers
  python main.py --optimizers AMCAS Adam SGD
  
  # Run with custom number of epochs
  python main.py --epochs 20
  
  # Run with early stopping
  python main.py --patience 5 --min-delta 0.0001
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Specific configuration file to run (default: run all)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--full', action='store_true',
                       help='Run full benchmark suite including speed and memory tests')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'cifar10'],
                       help='Datasets to run (default: mnist cifar10)')
    parser.add_argument('--architectures', nargs='+', default=['cnn', 'vit'],
                       help='Architectures to run (default: cnn vit)')
    parser.add_argument('--optimizers', nargs='+', 
                       default=['AMCAS', 'Adam', 'AdamW', 'SGD', 'SGD+Momentum', 'RMSprop', 'Adagrad', 'Adadelta', 'NAdam', 'RAdam'],
                       help='Optimizers to compare (default: all)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--data-augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--use-scheduler', action='store_true',
                       help='Use learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience in epochs (default: 10)')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='Minimum improvement for early stopping (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
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
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def create_experiment_configs(args):
    """Create experiment configurations based on command line arguments."""
    configs = []
    
    # Model registry
    model_registry = {
        'mnist': {
            'cnn': ['simple_cnn', 'cnn_v2', 'cnn_v3'],
            'vit': ['vit_small', 'vit_medium', 'vit_large']
        },
        'cifar10': {
            'cnn': ['resnet', 'cnn', 'vgg'],
            'vit': ['vit_small', 'vit_medium', 'vit_large', 'hybrid']
        }
    }
    
    for dataset in args.datasets:
        for architecture in args.architectures:
            if dataset in model_registry and architecture in model_registry[dataset]:
                for model_name in model_registry[dataset][architecture]:
                    for optimizer_name in args.optimizers:
                        config = {
                            'experiment_name': f"{dataset}_{architecture}_{model_name}_{optimizer_name}",
                            'dataset': dataset,
                            'model': model_name,
                            'optimizer': optimizer_name,
                            'epochs': args.epochs,
                            'batch_size': args.batch_size,
                            'learning_rate': args.learning_rate,
                            'data_augmentation': args.data_augmentation,
                            'use_scheduler': args.use_scheduler,
                            'patience': args.patience,
                            'min_delta': args.min_delta,
                            'seed': args.seed,
                            'optimizer_params': {}
                        }
                        
                        # Add optimizer-specific parameters
                        if optimizer_name == 'AMCAS':
                            config['optimizer_params'] = {
                                'betas': (0.9, 0.999),
                                'gamma': 0.1,
                                'lambda_consistency': 0.01,
                                'trust_region_params': (0.8, 1.2, 1.5, 0.5)
                            }
                        elif optimizer_name == 'AdamW':
                            config['optimizer_params'] = {
                                'betas': (0.9, 0.999),
                                'weight_decay': 0.01
                            }
                        elif optimizer_name == 'SGD+Momentum':
                            config['optimizer_params'] = {'momentum': 0.9}
                        elif optimizer_name in ['Adam', 'NAdam', 'RAdam']:
                            config['optimizer_params'] = {'betas': (0.9, 0.999)}
                        
                        configs.append(config)
    
    return configs


def run_experiments(args, configs, device):
    """Run all experiments."""
    print("\n" + "="*80)
    print("Running Main Experiments")
    print("="*80)
    
    all_results = {}
    
    for i, config in enumerate(configs):
        experiment_name = config['experiment_name']
        print(f"\nExperiment {i+1}/{len(configs)}: {experiment_name}")
        print("-" * 40)
        
        try:
            # Create experiment runner
            runner = ExperimentRunner(output_dir=str(Path(args.output) / 'experiments'), device=device)
            
            # Run experiment
            result = runner.run_experiment(config, experiment_name)
            all_results[experiment_name] = result
            
            # Save individual result
            result_dir = Path(args.output) / 'experiments' / 'raw_results'
            result_dir.mkdir(exist_ok=True, parents=True)
            result_path = result_dir / f'{experiment_name}_result.json'
            
            with open(result_path, 'w') as f:
                json.dump({
                    'experiment_name': experiment_name,
                    'config': config,
                    'result': result
                }, f, indent=2, default=str)
            
            print(f"Result saved to: {result_path}")
            
        except Exception as e:
            print(f"Error running experiment {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[experiment_name] = {'error': str(e)}
    
    return all_results


def run_benchmarks(args, device):
    """Run benchmark tests."""
    print("\n" + "="*80)
    print("Running Benchmarks")
    print("="*80)
    
    benchmark_results = {}
    
    # Memory profiling
    if not args.skip_benchmarks:
        print("\n1. Memory Profiling")
        print("-"*40)
        memory_profiler = MemoryProfiler(
            output_dir=str(Path(args.output) / 'memory_benchmark'),
            device=device
        )
        memory_results = memory_profiler.profile_all_optimizers(
            batch_size=args.memory_batch_size,
            num_iterations=args.memory_iterations
        )
        benchmark_results['memory_benchmark'] = memory_results
    
    # Speed benchmarking
    if not args.skip_benchmarks:
        print("\n2. Speed Benchmarking")
        print("-"*40)
        speed_benchmark = SpeedBenchmark(
            output_dir=str(Path(args.output) / 'speed_benchmark'),
            device=device
        )
        speed_results = speed_benchmark.benchmark_all_optimizers(
            batch_size=args.speed_batch_size,
            num_iterations=args.speed_iterations,
            warmup_iterations=args.speed_warmup
        )
        benchmark_results['speed_benchmark'] = speed_results
    
    # Optimizer comparison on synthetic functions
    if not args.skip_benchmarks and args.full:
        print("\n3. Optimizer Comparison on Synthetic Functions")
        print("-"*40)
        optimizer_comparison = OptimizerComparison(
            output_dir=str(Path(args.output) / 'optimizer_comparison')
        )
        comparison_results = optimizer_comparison.run_all_comparisons(
            num_iterations=args.optimizer_iterations
        )
        benchmark_results['optimizer_comparison'] = comparison_results
    
    return benchmark_results


def generate_reports(args, experiment_results, benchmark_results):
    """Generate comprehensive reports."""
    print("\n" + "="*80)
    print("Generating Reports")
    print("="*80)
    
    # Create results exporter
    exporter = ResultsExporter(output_dir=args.output)
    
    # Export to Excel
    if experiment_results:
        print("\n1. Exporting to Excel...")
        excel_path = exporter.export_to_excel(experiment_results, 'experiment_results.xlsx')
        print(f"Excel report saved to: {excel_path}")
    
    # Export to JSON
    if experiment_results:
        print("\n2. Exporting to JSON...")
        json_path = exporter.export_to_json(experiment_results, 'experiment_results.json')
        print(f"JSON data saved to: {json_path}")
    
    # Generate plots
    if experiment_results:
        print("\n3. Generating plots...")
        exporter.generate_plots(experiment_results, plot_types=['all'])
        print(f"Plots saved to: {Path(args.output) / 'plots'}")
    
    # Generate markdown report
    if experiment_results:
        print("\n4. Generating markdown report...")
        report_path = exporter.generate_report(experiment_results, 'experiment_report.md')
        print(f"Report saved to: {report_path}")
    
    # Generate summary
    print("\n5. Generating summary...")
    summary_file = Path(args.output) / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Optimizer Comparison Framework - Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(args.gpu)}\n")
        f.write(f"Output directory: {args.output}\n\n")
        
        if experiment_results:
            f.write("Experiments Completed:\n")
            f.write("-"*40 + "\n")
            successful = [k for k, v in experiment_results.items() if 'error' not in v]
            failed = [k for k, v in experiment_results.items() if 'error' in v]
            
            f.write(f"Successful: {len(successful)} experiments\n")
            for exp_name in successful[:10]:  # Show first 10
                f.write(f"  [OK] {exp_name}\n")
            if len(successful) > 10:
                f.write(f"  ... and {len(successful) - 10} more\n")
            
            if failed:
                f.write(f"\nFailed: {len(failed)} experiments\n")
                for exp_name in failed:
                    f.write(f"  [FAILED] {exp_name}: {experiment_results[exp_name]['error']}\n")
            
            f.write("\n")
        
        if benchmark_results:
            f.write("Benchmarks Completed:\n")
            f.write("-"*40 + "\n")
            if 'memory_benchmark' in benchmark_results:
                f.write("✓ Memory profiling\n")
            if 'speed_benchmark' in benchmark_results:
                f.write("✓ Speed benchmarking\n")
            if 'optimizer_comparison' in benchmark_results:
                f.write("✓ Optimizer comparison on synthetic functions\n")
            f.write("\n")
        
        # Best performers
        if experiment_results:
            valid_results = {k: v for k, v in experiment_results.items() if 'error' not in v}
            if valid_results:
                f.write("Best Performers:\n")
                f.write("-"*40 + "\n")
                
                # By accuracy
                best_by_acc = max(valid_results.items(), 
                                key=lambda x: x[1].get('best_test_accuracy', 0))
                f.write(f"Best Accuracy: {best_by_acc[0]} ({best_by_acc[1].get('best_test_accuracy', 0):.2f}%)\n")
                
                # By speed
                best_by_speed = min(valid_results.items(),
                                  key=lambda x: x[1].get('total_training_time', float('inf')))
                best_speed_time = best_by_speed[1].get('total_training_time', 0)
                if isinstance(best_speed_time, list) and len(best_speed_time) > 0:
                    best_speed_time = best_speed_time[0]
                f.write(f"Fastest Training: {best_by_speed[0]} ({best_speed_time:.2f}s)\n")
                
                # By memory efficiency
                if any('memory_usage' in v for v in valid_results.values()):
                    best_by_memory = min(valid_results.items(),
                                       key=lambda x: max(x[1].get('memory_usage', [0])) if x[1].get('memory_usage') else 0)
                    best_memory_usage = max(best_by_memory[1].get('memory_usage', [0])) if best_by_memory[1].get('memory_usage') else 0
                    f.write(f"Most Memory Efficient: {best_by_memory[0]} ({best_memory_usage:.2f} GB)\n")
                
                f.write("\n")
        
        f.write("Files Generated:\n")
        f.write("-"*40 + "\n")
        for root, dirs, files in os.walk(args.output):
            level = root.replace(str(args.output), '').count(os.sep)
            indent = ' ' * 2 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    return summary_file


def main():
    """Main function to run all experiments."""
    args = parse_args()
    
    print("\n" + "="*80)
    print("Optimizer Comparison Framework")
    print("="*80)
    print(f"Output directory: {args.output}")
    print(f"GPU device: {args.gpu}")
    print(f"Full benchmark suite: {args.full}")
    print(f"Skip experiments: {args.skip_experiments}")
    print(f"Skip benchmarks: {args.skip_benchmarks}")
    print(f"Datasets: {args.datasets}")
    print(f"Architectures: {args.architectures}")
    print(f"Optimizers: {args.optimizers}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*80 + "\n")
    
    # Set device
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
        # Create experiment configurations
        if args.config:
            # Load specific configuration file
            with open(args.config, 'r') as f:
                configs = yaml.safe_load(f)
            if not isinstance(configs, list):
                configs = [configs]
        else:
            # Generate configurations based on command line arguments
            configs = create_experiment_configs(args)
        
        print(f"\nGenerated {len(configs)} experiment configurations")
        
        # Run experiments
        experiment_results = run_experiments(args, configs, device)
        all_results['experiments'] = experiment_results
    
    # Run benchmarks if requested
    benchmark_results = {}
    if args.full and not args.skip_benchmarks:
        benchmark_results = run_benchmarks(args, device)
        all_results.update(benchmark_results)
    
    # Generate reports
    if not args.skip_experiments:
        generate_reports(args, 
                        all_results.get('experiments', {}), 
                        benchmark_results)
    
    # Save all results
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
    
    # Print quick summary
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)
    
    print("\nQuick Summary:")
    print("-"*40)
    if 'experiments' in all_results:
        successful = len([k for k, v in all_results['experiments'].items() if 'error' not in v])
        failed = len([k for k, v in all_results['experiments'].items() if 'error' in v])
        print(f"Experiments: {successful} successful, {failed} failed")
    
    if benchmark_results:
        print(f"Benchmarks: {len(benchmark_results)} completed")
    
    print(f"\nResults saved in: {output_dir}")
    print("\nTo view results:")
    print(f"  - Check Excel file: {output_dir}/experiment_results.xlsx")
    print(f"  - Check JSON data: {output_dir}/experiment_results.json")
    print(f"  - Check report: {output_dir}/experiment_report.md")
    print(f"  - Check plots: {output_dir}/plots/")
    print(f"  - Check summary: {output_dir}/summary.txt")
    
    return all_results


if __name__ == '__main__':
    main()