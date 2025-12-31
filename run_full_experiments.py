#!/usr/bin/env python3
"""
Example script to run the complete optimizer comparison experiments for publishing paper.
This script demonstrates how to run all variations of experiments as requested:
- MNIST and CIFAR10 datasets
- CNN and ViT architectures
- AMCAS optimizer vs other popular optimizers
- Multiple performance metrics
- Speed and memory benchmarking
- Excel reporting
"""

import subprocess
import sys
import os
from pathlib import Path

def run_experiment(command, experiment_name=""):
    """Run a command and print output with real-time progress."""
    print(f"\n{'='*80}")
    print(f"Running: {experiment_name}")
    print(f"Command: {command}")
    print('='*80)
    
    # Run with real-time output to show progress
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[FAIL] Error running experiment: {experiment_name}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    else:
        print(f"[PASS] Successfully completed: {experiment_name}")
        # Show summary from output
        if "Best test accuracy:" in result.stdout:
            for line in result.stdout.split('\n'):
                if "Best test accuracy:" in line or "Total training time:" in line:
                    print(f"  {line.strip()}")
    
    return result.returncode

def main():
    """Run all experiments for paper publication."""
    
    # Create output directory for paper results
    output_dir = "paper_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("OPTIMIZER COMPARISON EXPERIMENTS FOR PAPER PUBLICATION")
    print("="*80)
    print("\nThis script will run comprehensive experiments comparing AMCAS optimizer")
    print("with other popular optimizers on MNIST and CIFAR10 datasets")
    print("using both CNN and Vision Transformer (ViT) architectures.")
    print("\nAll results will be saved to:", output_dir)
    print("="*80)
    
    # List of optimizers to compare
    optimizers = [
        "AMCAS",        # User's proposed optimizer
        "Adam",
        "AdamW",
        "SGD",
        "SGD+Momentum",
        "RMSprop",
        "Adagrad",
        "Adadelta",
        "NAdam",
        "RAdam"
    ]
    
    # Configuration for paper experiments
    configs = [
        {
            "name": "MNIST_CNN_Comparison",
            "datasets": ["mnist"],
            "architectures": ["cnn"],
            "optimizers": optimizers,
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "data_augmentation": False,
            "use_scheduler": True,
            "patience": 5,
            "min_delta": 0.001,
            "full_benchmark": True
        },
        {
            "name": "MNIST_ViT_Comparison",
            "datasets": ["mnist"],
            "architectures": ["vit"],
            "optimizers": optimizers,
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "data_augmentation": False,
            "use_scheduler": True,
            "patience": 5,
            "min_delta": 0.001,
            "full_benchmark": True
        },
        {
            "name": "CIFAR10_CNN_Comparison",
            "datasets": ["cifar10"],
            "architectures": ["cnn"],
            "optimizers": optimizers,
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "data_augmentation": True,
            "use_scheduler": True,
            "patience": 10,
            "min_delta": 0.001,
            "full_benchmark": True
        },
        {
            "name": "CIFAR10_ViT_Comparison",
            "datasets": ["cifar10"],
            "architectures": ["vit"],
            "optimizers": optimizers,
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "data_augmentation": True,
            "use_scheduler": True,
            "patience": 10,
            "min_delta": 0.001,
            "full_benchmark": True
        }
    ]
    
    # Run each configuration
    all_results = []
    
    # Calculate total estimated time
    total_experiments = 0
    total_estimated_seconds = 0
    for config in configs:
        num_exps = len(config['optimizers']) * len(config['datasets']) * len(config['architectures']) * 3
        total_experiments += num_exps
        total_estimated_seconds += num_exps * config['epochs'] * 2
    
    print(f"\n[OVERVIEW]:")
    print(f"   Total experiment sets: {len(configs)}")
    print(f"   Total individual experiments: ~{total_experiments}")
    print(f"   Total estimated time: ~{total_estimated_seconds/60:.1f} minutes")
    print(f"   Output directory: {output_dir}")
    
    for i, config in enumerate(configs):
        print(f"\n\n{'='*80}")
        print(f"[EXPERIMENT SET {i+1}/{len(configs)}]: {config['name']}")
        print(f"Progress: {i+1}/{len(configs)} ({((i+1)/len(configs)*100):.1f}%)")
        print('='*80)
        
        # Build command
        cmd_parts = [
            "python main.py",
            f"--datasets {' '.join(config['datasets'])}",
            f"--architectures {' '.join(config['architectures'])}",
            f"--optimizers {' '.join(config['optimizers'])}",
            f"--epochs {config['epochs']}",
            f"--batch-size {config['batch_size']}",
            f"--learning-rate {config['learning_rate']}",
            f"--patience {config['patience']}",
            f"--min-delta {config['min_delta']}",
            f"--output {output_dir}/{config['name']}"
        ]
        
        if config['data_augmentation']:
            cmd_parts.append("--data-augmentation")
        
        if config['use_scheduler']:
            cmd_parts.append("--use-scheduler")
        
        if config['full_benchmark']:
            cmd_parts.append("--full")
        
        cmd = " ".join(cmd_parts)
        
        # Calculate estimated time and provide progress info
        num_experiments = len(config['optimizers']) * len(config['datasets']) * len(config['architectures']) * 3  # 3 models per combo
        estimated_time_per_exp = config['epochs'] * 2  # Rough estimate: 2 seconds per epoch
        total_estimated_time = num_experiments * estimated_time_per_exp
        
        print(f"\n[EXPERIMENT DETAILS]:")
        print(f"   - Optimizers: {len(config['optimizers'])}")
        print(f"   - Datasets: {', '.join(config['datasets'])}")
        print(f"   - Architectures: {', '.join(config['architectures'])}")
        print(f"   - Total experiments: ~{num_experiments}")
        print(f"   - Epochs per experiment: {config['epochs']}")
        print(f"   - Estimated time: ~{total_estimated_time} seconds (~{total_estimated_time/60:.1f} minutes)")
        
        # Run experiment
        print(f"\n[STARTING] Experiment set...")
        return_code = run_experiment(cmd, config['name'])
        
        if return_code == 0:
            print(f"\n[SUCCESS] Experiment set '{config['name']}' completed successfully!")
            all_results.append({
                "name": config['name'],
                "output_dir": f"{output_dir}/{config['name']}",
                "status": "SUCCESS"
            })
        else:
            print(f"\n[FAILED] Experiment set '{config['name']}' failed!")
            all_results.append({
                "name": config['name'],
                "output_dir": f"{output_dir}/{config['name']}",
                "status": "FAILED"
            })
    
    # Generate final summary report
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    summary_file = Path(output_dir) / "paper_experiments_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Optimizer Comparison Paper Experiments - Summary\n\n")
        f.write("## Overview\n\n")
        f.write("This document summarizes the results of comprehensive optimizer comparison experiments\n")
        f.write("conducted for paper publication.\n\n")
        
        f.write("## Experiment Configurations\n\n")
        f.write("| Configuration | Datasets | Architectures | Optimizers | Epochs | Data Augmentation | Learning Rate Scheduler |\n")
        f.write("|--------------|----------|---------------|------------|--------|-------------------|--------------------------|\n")
        
        for config in configs:
            f.write(f"| {config['name']} | {', '.join(config['datasets'])} | {', '.join(config['architectures'])} | {len(config['optimizers'])} optimizers | {config['epochs']} | {'Yes' if config['data_augmentation'] else 'No'} | {'Yes' if config['use_scheduler'] else 'No'} |\n")
        
        f.write("\n## Results Summary\n\n")
        f.write("| Experiment Set | Status | Output Directory |\n")
        f.write("|----------------|--------|-----------------|\n")
        
        for result in all_results:
            status_symbol = "✓" if result['status'] == "SUCCESS" else "✗"
            f.write(f"| {result['name']} | {status_symbol} {result['status']} | {result['output_dir']} |\n")
        
        f.write("\n## Files Generated\n\n")
        f.write("Each experiment set generates the following files:\n")
        f.write("- `experiment_results.xlsx`: Excel file with all metrics\n")
        f.write("- `experiment_results.json`: Raw data in JSON format\n")
        f.write("- `experiment_report.md`: Markdown report with analysis\n")
        f.write("- `summary.txt`: Quick summary of results\n")
        f.write("- `plots/`: Directory with visualization plots\n")
        f.write("- `experiments/`: Directory with checkpoints and raw results\n")
        f.write("- `all_results.json`: Combined results from all experiments\n")
        
        f.write("\n## How to Analyze Results\n\n")
        f.write("1. **Excel Analysis**: Open `experiment_results.xlsx` to view:\n")
        f.write("   - Summary sheet: Key metrics for all experiments\n")
        f.write("   - Detailed_Metrics: Per-epoch metrics\n")
        f.write("   - Comparison: Ranked comparison of optimizers\n")
        f.write("   - Statistical_Analysis: Statistical metrics\n")
        f.write("   - Computational_Metrics: Speed and memory usage\n")
        f.write("   - Optimizer_Metrics: Optimizer-specific metrics\n\n")
        
        f.write("2. **Visual Analysis**: Check the `plots/` directory for:\n")
        f.write("   - `loss_curves.png`: Training and test loss curves\n")
        f.write("   - `accuracy_curves.png`: Training and test accuracy curves\n")
        f.write("   - `f1_curves.png`: Test F1 score curves\n")
        f.write("   - `time_comparison.png`: Training time comparison\n")
        f.write("   - `memory_usage.png`: Memory usage comparison\n")
        f.write("   - `gradient_stats.png`: Gradient statistics\n\n")
        
        f.write("3. **Statistical Analysis**: Use `experiment_results.json` for:\n")
        f.write("   - Statistical tests (t-tests, ANOVA)\n")
        f.write("   - Correlation analysis\n")
        f.write("   - Regression analysis\n\n")
        
        f.write("## Recommendations for Paper\n\n")
        f.write("1. **Best Performing Optimizer**: Check the 'Comparison' sheet in Excel\n")
        f.write("2. **Statistical Significance**: Use statistical tests from JSON data\n")
        f.write("3. **Computational Efficiency**: Analyze speed and memory metrics\n")
        f.write("4. **Convergence Analysis**: Examine loss and accuracy curves\n")
        f.write("5. **Robustness**: Check performance across different architectures\n\n")
        
        f.write("## Citation\n\n")
        f.write("If you use these results in your paper, please cite:\n\n")
        f.write("```bibtex\n")
        f.write("@article{optimizer_comparison_2024,\n")
        f.write("  title = {Comprehensive Comparison of Optimizers on MNIST and CIFAR10},\n")
        f.write("  author = {Your Name},\n")
        f.write("  journal = {Your Journal},\n")
        f.write("  year = {2024},\n")
        f.write("  note = {Experimental results generated using Optimizer Comparison Framework}\n")
        f.write("}\n")
        f.write("```\n")
    
    print(f"\nFinal summary saved to: {summary_file}")
    
    # Print quick summary
    print("\nExperiment Results Summary:")
    print("-"*40)
    for result in all_results:
        status_symbol = "✓" if result['status'] == "SUCCESS" else "✗"
        print(f"{status_symbol} {result['name']}: {result['status']}")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print("\nTo analyze results:")
    print(f"1. Check individual experiment directories in '{output_dir}/'")
    print(f"2. Open Excel files for comprehensive analysis")
    print(f"3. Review markdown reports for detailed insights")
    print(f"4. Examine plots for visual comparisons")
    print(f"\nFinal summary report: {summary_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())