"""
Comprehensive benchmarking for optimizer comparison.
Runs all experiments automatically and generates comprehensive reports.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
import traceback
import sys
import os

# Add parent directory to path to import modules
sys.path.append('..')

from experiments.experiment_runner import ExperimentRunner
from experiments.results_exporter import ResultsExporter
from models.cnn_mnist import get_mnist_model
from models.cnn_cifar10 import get_cifar10_model
from models.vit_mnist import VisionTransformerMNISTSmall, VisionTransformerMNISTMedium, VisionTransformerMNISTLarge
from models.vit_cifar10 import VisionTransformerCIFAR10Small, VisionTransformerCIFAR10Medium, VisionTransformerCIFAR10Large
from optimizers.amcas import AMCAS
from optimizers.ultron import ULTRON


class ComprehensiveBenchmark:
    """
    Comprehensive benchmark that runs all experiments and generates reports.
    """
    
    def __init__(self, output_dir='benchmark_results', device=None):
        """
        Initialize comprehensive benchmark.
        
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
            
        self.experiment_runner = ExperimentRunner(output_dir='benchmark_results/experiments', device=device)
        self.results_exporter = ResultsExporter(output_dir='benchmark_results')
        
        print(f"Comprehensive benchmark initialized with device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def run_all_experiments(self, config_files: Optional[List[str]] = None):
        """
        Run all experiments from configuration files.
        
        Args:
            config_files: List of configuration file paths. If None, runs all default experiments.
            
        Returns:
            Dictionary with all experiment results
        """
        if config_files is None:
            # Run all default experiments
            config_files = [
                'experiments/configs/mnist_cnn.yaml',
                'experiments/configs/mnist_vit.yaml',
                'experiments/configs/cifar10_cnn.yaml',
                'experiments/configs/cifar10_vit.yaml',
            ]
        
        all_results = {}
        
        for config_file in config_files:
            print(f"\n{'='*80}")
            print(f"Running experiment from config: {config_file}")
            print(f"{'='*80}")
            
            try:
                # Load configuration
                config = self.experiment_runner.load_config(config_file)
                
                # Extract experiment name
                experiment_name = config.get('experiment_name', Path(config_file).stem)
                
                # Run all optimizer combinations
                results = self._run_optimizer_comparison(config, experiment_name)
                all_results.update(results)
                
                # Save individual experiment results
                self._save_experiment_results(results, experiment_name)
                
            except Exception as e:
                print(f"Error running experiment from {config_file}: {e}")
                traceback.print_exc()
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_results)
        
        return all_results
    
    def _run_optimizer_comparison(self, config: Dict, experiment_name: str) -> Dict[str, Dict]:
        """
        Run comparison of all optimizers for a given configuration.
        
        Args:
            config: Experiment configuration
            experiment_name: Base name for the experiment
            
        Returns:
            Dictionary mapping experiment names to results
        """
        results = {}
        
        # Extract common configuration
        dataset = config['dataset']
        model_name = config['model']
        epochs = config.get('epochs', 10)
        batch_size = config.get('batch_size', 64)
        learning_rate = config.get('learning_rate', 0.001)
        data_augmentation = config.get('data_augmentation', False)
        use_scheduler = config.get('use_scheduler', False)
        seed = config.get('seed', 42)
        
        # Get optimizer configurations
        optimizers_config = config.get('optimizers', [])
        
        for optimizer_config in optimizers_config:
            optimizer_name = optimizer_config['name']
            full_experiment_name = f"{experiment_name}_{optimizer_name.lower()}"
            
            print(f"\n{'='*60}")
            print(f"Running: {full_experiment_name}")
            print(f"{'='*60}")
            
            # Create individual experiment configuration
            exp_config = {
                'dataset': dataset,
                'model': model_name,
                'optimizer': optimizer_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'data_augmentation': data_augmentation,
                'use_scheduler': use_scheduler,
                'seed': seed,
                'optimizer_params': optimizer_config.get('params', {})
            }
            
            try:
                # Run experiment
                result = self.experiment_runner.run_experiment(exp_config, full_experiment_name)
                results[full_experiment_name] = result
                
                # Clear GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error running {full_experiment_name}: {e}")
                traceback.print_exc()
                results[full_experiment_name] = {'error': str(e)}
        
        return results
    
    def _save_experiment_results(self, results: Dict[str, Dict], experiment_name: str):
        """
        Save individual experiment results.
        
        Args:
            results: Dictionary mapping experiment names to results
            experiment_name: Base experiment name
        """
        # Export to Excel
        excel_file = f"{experiment_name}_results.xlsx"
        self.results_exporter.export_to_excel(results, excel_file)
        
        # Export to JSON
        json_file = f"{experiment_name}_results.json"
        self.results_exporter.export_to_json(results, json_file)
        
        # Generate plots
        self.results_exporter.generate_plots(results, plot_types=['all'])
        
        # Generate report
        report_file = f"{experiment_name}_report.md"
        self.results_exporter.generate_report(results, report_file)
    
    def _generate_comprehensive_report(self, all_results: Dict[str, Dict]):
        """
        Generate comprehensive report comparing all experiments.
        
        Args:
            all_results: Dictionary mapping all experiment names to results
        """
        report_path = self.output_dir / 'comprehensive_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n\n")
            total_experiments = len(all_results)
            successful_experiments = sum(1 for r in all_results.values() if 'error' not in r)
            failed_experiments = total_experiments - successful_experiments
            
            f.write(f"- **Total Experiments**: {total_experiments}\n")
            f.write(f"- **Successful Experiments**: {successful_experiments}\n")
            f.write(f"- **Failed Experiments**: {failed_experiments}\n")
            f.write(f"- **Success Rate**: {successful_experiments/total_experiments*100:.1f}%\n\n")
            
            # Group results by dataset and model
            dataset_model_groups = {}
            for exp_name, result in all_results.items():
                if 'error' in result:
                    continue
                
                parts = exp_name.split('_')
                if len(parts) >= 3:
                    dataset = parts[0]
                    model = parts[1]
                    optimizer = parts[2]
                    
                    key = f"{dataset}_{model}"
                    if key not in dataset_model_groups:
                        dataset_model_groups[key] = {}
                    dataset_model_groups[key][optimizer] = result
            
            # Best performers by dataset-model group
            f.write("## Best Performers by Dataset-Model Combination\n\n")
            
            for group_name, optimizer_results in dataset_model_groups.items():
                f.write(f"### {group_name.replace('_', ' ').title()}\n\n")
                
                # Sort by best test accuracy
                sorted_results = sorted(
                    optimizer_results.items(),
                    key=lambda x: x[1].get('best_test_accuracy', 0),
                    reverse=True
                )
                
                f.write("| Optimizer | Best Test Acc (%) | Final Test Acc (%) | Training Time (s) | Peak Memory (GB) |\n")
                f.write("|-----------|-------------------|-------------------|-------------------|------------------|\n")
                
                for optimizer_name, result in sorted_results[:5]:  # Top 5
                    best_acc = result.get('best_test_accuracy', 0)
                    final_acc = result.get('test_accuracy', [0])[-1] if result.get('test_accuracy') else 0
                    training_time = result.get('total_training_time', 0)
                    peak_memory = max(result.get('memory_usage', [0]))
                    
                    f.write(f"| {optimizer_name} | {best_acc:.2f} | {final_acc:.2f} | {training_time:.2f} | {peak_memory:.2f} |\n")
                
                f.write("\n")
            
            # Performance comparison across datasets
            f.write("## Performance Comparison Across Datasets\n\n")
            
            # Create comparison table
            comparison_data = []
            for exp_name, result in all_results.items():
                if 'error' in result:
                    continue
                
                parts = exp_name.split('_')
                if len(parts) >= 3:
                    dataset = parts[0]
                    model = parts[1]
                    optimizer = parts[2]
                    
                    comparison_data.append({
                        'Dataset': dataset,
                        'Model': model,
                        'Optimizer': optimizer,
                        'Best Accuracy': result.get('best_test_accuracy', 0),
                        'Final Accuracy': result.get('test_accuracy', [0])[-1] if result.get('test_accuracy') else 0,
                        'Training Time': result.get('total_training_time', 0),
                        'Peak Memory': max(result.get('memory_usage', [0])),
                        'Parameters': result.get('model_params', 0),
                    })
            
            if comparison_data:
                import pandas as pd
                df_comparison = pd.DataFrame(comparison_data)
                
                # Best accuracy per dataset
                f.write("### Best Accuracy per Dataset\n\n")
                best_by_dataset = df_comparison.loc[df_comparison.groupby('Dataset')['Best Accuracy'].idxmax()]
                for _, row in best_by_dataset.iterrows():
                    f.write(f"- **{row['Dataset']}**: {row['Optimizer']} with {row['Model']} ({row['Best Accuracy']:.2f}% accuracy)\n")
                
                f.write("\n")
                
                # Fastest training per dataset
                f.write("### Fastest Training per Dataset\n\n")
                fastest_by_dataset = df_comparison.loc[df_comparison.groupby('Dataset')['Training Time'].idxmin()]
                for _, row in fastest_by_dataset.iterrows():
                    f.write(f"- **{row['Dataset']}**: {row['Optimizer']} with {row['Model']} ({row['Training Time']:.2f}s)\n")
                
                f.write("\n")
                
                # Most memory efficient per dataset
                f.write("### Most Memory Efficient per Dataset\n\n")
                efficient_by_dataset = df_comparison.loc[df_comparison.groupby('Dataset')['Peak Memory'].idxmin()]
                for _, row in efficient_by_dataset.iterrows():
                    f.write(f"- **{row['Dataset']}**: {row['Optimizer']} with {row['Model']} ({row['Peak Memory']:.2f}GB peak memory)\n")
            
            # Recommendations
            f.write("## Overall Recommendations\n\n")
            
            if comparison_data:
                # Find best overall optimizer
                best_overall = df_comparison.loc[df_comparison['Best Accuracy'].idxmax()]
                f.write(f"### Best Overall Performance\n")
                f.write(f"- **Optimizer**: {best_overall['Optimizer']}\n")
                f.write(f"- **Dataset-Model**: {best_overall['Dataset']} with {best_overall['Model']}\n")
                f.write(f"- **Accuracy**: {best_overall['Best Accuracy']:.2f}%\n")
                f.write(f"- **Training Time**: {best_overall['Training Time']:.2f}s\n")
                f.write(f"- **Peak Memory**: {best_overall['Peak Memory']:.2f}GB\n\n")
                
                # Find most efficient optimizer (accuracy/time tradeoff)
                df_comparison['Efficiency'] = df_comparison['Best Accuracy'] / df_comparison['Training Time']
                most_efficient = df_comparison.loc[df_comparison['Efficiency'].idxmax()]
                f.write(f"### Most Efficient (Accuracy/Time)\n")
                f.write(f"- **Optimizer**: {most_efficient['Optimizer']}\n")
                f.write(f"- **Dataset-Model**: {most_efficient['Dataset']} with {most_efficient['Model']}\n")
                f.write(f"- **Efficiency Score**: {most_efficient['Efficiency']:.4f} %/s\n")
                f.write(f"- **Accuracy**: {most_efficient['Best Accuracy']:.2f}%\n")
                f.write(f"- **Training Time**: {most_efficient['Training Time']:.2f}s\n\n")
                
                # Find most memory efficient optimizer (accuracy/memory tradeoff)
                df_comparison['MemoryEfficiency'] = df_comparison['Best Accuracy'] / df_comparison['Peak Memory']
                most_mem_efficient = df_comparison.loc[df_comparison['MemoryEfficiency'].idxmax()]
                f.write(f"### Most Memory Efficient (Accuracy/Memory)\n")
                f.write(f"- **Optimizer**: {most_mem_efficient['Optimizer']}\n")
                f.write(f"- **Dataset-Model**: {most_mem_efficient['Dataset']} with {most_mem_efficient['Model']}\n")
                f.write(f"- **Memory Efficiency Score**: {most_mem_efficient['MemoryEfficiency']:.4f} %/GB\n")
                f.write(f"- **Accuracy**: {most_mem_efficient['Best Accuracy']:.2f}%\n")
                f.write(f"- **Peak Memory**: {most_mem_efficient['Peak Memory']:.2f}GB\n\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("The following files have been generated:\n\n")
            f.write("- `comprehensive_report.md`: This report\n")
            f.write("- `benchmark_results/`: Directory containing all results\n")
            f.write("  - `experiments/`: Individual experiment results\n")
            f.write("  - `plots/`: Generated plots\n")
            f.write("  - `*.xlsx`: Excel files with detailed results\n")
            f.write("  - `*.json`: JSON files with raw data\n")
            f.write("  - `*.md`: Individual experiment reports\n")
        
        print(f"\nComprehensive report saved to: {report_path}")
        
        # Also export all results to Excel
        all_excel_path = self.output_dir / 'all_experiments_results.xlsx'
        self.results_exporter.export_to_excel(all_results, 'all_experiments_results.xlsx')
        print(f"All results exported to Excel: {all_excel_path}")
        
        return report_path
    
    def run_speed_benchmark(self, num_iterations: int = 1000):
        """
        Run speed benchmark for all optimizers.
        
        Args:
            num_iterations: Number of iterations for speed test
        """
        print(f"\n{'='*80}")
        print("Running Speed Benchmark")
        print(f"{'='*80}")
        
        from .speed_benchmark import SpeedBenchmark
        
        speed_benchmark = SpeedBenchmark(output_dir=str(self.output_dir / 'speed_benchmark'), device=self.device)
        speed_results = speed_benchmark.run_all_benchmarks(num_iterations)
        
        return speed_results
    
    def run_memory_benchmark(self, batch_size: int = 64):
        """
        Run memory benchmark for all optimizers.
        
        Args:
            batch_size: Batch size for memory test
        """
        print(f"\n{'='*80}")
        print("Running Memory Benchmark")
        print(f"{'='*80}")
        
        from .memory_profiler import MemoryProfiler
        
        memory_profiler = MemoryProfiler(output_dir=str(self.output_dir / 'memory_benchmark'), device=self.device)
        memory_results = memory_profiler.profile_all_optimizers(batch_size)
        
        return memory_results
    
    def run_optimizer_comparison(self):
        """
        Run detailed optimizer comparison on synthetic functions.
        """
        print(f"\n{'='*80}")
        print("Running Optimizer Comparison on Synthetic Functions")
        print(f"{'='*80}")
        
        from .optimizer_comparison import OptimizerComparison
        
        optimizer_comparison = OptimizerComparison(output_dir=str(self.output_dir / 'optimizer_comparison'))
        comparison_results = optimizer_comparison.run_all_comparisons()
        
        return comparison_results
    
    def run_full_benchmark_suite(self):
        """
        Run the full benchmark suite including all experiments and benchmarks.
        
        Returns:
            Dictionary with all benchmark results
        """
        print(f"\n{'='*80}")
        print("Starting Full Benchmark Suite")
        print(f"{'='*80}")
        
        all_results = {}
        
        # Run all experiments
        print("\n1. Running all experiments...")
        experiment_results = self.run_all_experiments()
        all_results['experiments'] = experiment_results
        
        # Run speed benchmark
        print("\n2. Running speed benchmark...")
        try:
            speed_results = self.run_speed_benchmark()
            all_results['speed_benchmark'] = speed_results
        except Exception as e:
            print(f"Speed benchmark failed: {e}")
            traceback.print_exc()
        
        # Run memory benchmark
        print("\n3. Running memory benchmark...")
        try:
            memory_results = self.run_memory_benchmark()
            all_results['memory_benchmark'] = memory_results
        except Exception as e:
            print(f"Memory benchmark failed: {e}")
            traceback.print_exc()
        
        # Run optimizer comparison
        print("\n4. Running optimizer comparison on synthetic functions...")
        try:
            comparison_results = self.run_optimizer_comparison()
            all_results['optimizer_comparison'] = comparison_results
        except Exception as e:
            print(f"Optimizer comparison failed: {e}")
            traceback.print_exc()
        
        # Save all results
        print("\n5. Saving all benchmark results...")
        results_file = self.output_dir / 'full_benchmark_results.json'
        with open(results_file, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for key, value in all_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = self._make_serializable(value)
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nFull benchmark suite completed!")
        print(f"Results saved to: {results_file}")
        
        return all_results
    
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
    """Main function to run the comprehensive benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive optimizer benchmark')
    parser.add_argument('--config', type=str, default=None,
                       help='Specific configuration file to run (default: run all)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--full', action='store_true',
                       help='Run full benchmark suite including speed and memory tests')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # Create benchmark
    benchmark = ComprehensiveBenchmark(output_dir=args.output, device=device)
    
    if args.full:
        # Run full benchmark suite
        benchmark.run_full_benchmark_suite()
    else:
        # Run experiments only
        config_files = [args.config] if args.config else None
        benchmark.run_all_experiments(config_files)


if __name__ == '__main__':
    main()