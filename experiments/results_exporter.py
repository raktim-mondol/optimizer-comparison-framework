"""
Results exporter for saving experiment results to Excel and JSON formats.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import torch


class ResultsExporter:
    """
    Exports experiment results to various formats (Excel, JSON, plots).
    """
    
    def __init__(self, output_dir='results'):
        """
        Initialize results exporter.
        
        Args:
            output_dir: Directory to save exported results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def export_to_excel(self, results: Dict[str, Dict], filename: str = 'experiment_results.xlsx'):
        """
        Export results to Excel file with multiple sheets.
        
        Args:
            results: Dictionary mapping experiment names to results
            filename: Output Excel filename
        """
        excel_path = self.output_dir / filename
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            self._create_summary_sheet(results, writer)
            
            # Sheet 2: Detailed metrics per experiment
            self._create_detailed_metrics_sheet(results, writer)
            
            # Sheet 3: Comparative analysis
            self._create_comparison_sheet(results, writer)
            
            # Sheet 4: Statistical analysis
            self._create_statistical_sheet(results, writer)
            
            # Sheet 5: Computational metrics
            self._create_computational_sheet(results, writer)
            
            # Sheet 6: Optimizer-specific metrics
            self._create_optimizer_metrics_sheet(results, writer)
        
        print(f"Results exported to Excel: {excel_path}")
        return excel_path
    
    def _create_summary_sheet(self, results: Dict[str, Dict], writer: pd.ExcelWriter):
        """Create summary sheet with key metrics."""
        summary_data = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                summary_data.append({
                    'Experiment': exp_name,
                    'Status': 'ERROR',
                    'Error': result['error'],
                    'Best Test Accuracy (%)': None,
                    'Final Test Accuracy (%)': None,
                    'Final Test F1': None,
                    'Total Training Time (s)': None,
                    'Avg Epoch Time (s)': None,
                    'Model Parameters': None,
                    'Model FLOPs': None,
                    'Peak GPU Memory (GB)': None,
                })
            else:
                summary_data.append({
                    'Experiment': exp_name,
                    'Status': 'COMPLETED',
                    'Error': None,
                    'Best Test Accuracy (%)': result.get('best_test_accuracy', 0),
                    'Final Test Accuracy (%)': result.get('test_accuracy', [0])[-1] if result.get('test_accuracy') else 0,
                    'Final Test F1': result.get('test_f1', [0])[-1] if result.get('test_f1') else 0,
                    'Total Training Time (s)': result.get('total_training_time', 0),
                    'Avg Epoch Time (s)': np.mean(result.get('epoch_times', [0])),
                    'Model Parameters': result.get('model_params', 0),
                    'Model FLOPs': result.get('model_flops', 0),
                    'Peak GPU Memory (GB)': max(result.get('memory_usage', [0])) if result.get('memory_usage') else 0,
                })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format the sheet
        worksheet = writer.sheets['Summary']
        for idx, col in enumerate(df_summary.columns):
            column_width = max(df_summary[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = column_width
    
    def _create_detailed_metrics_sheet(self, results: Dict[str, Dict], writer: pd.ExcelWriter):
        """Create detailed metrics sheet."""
        detailed_data = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            epochs = result.get('epoch', list(range(1, len(result.get('train_loss', [])) + 1)))
            train_losses = result.get('train_loss', [])
            train_accs = result.get('train_accuracy', [])
            test_losses = result.get('test_loss', [])
            test_accs = result.get('test_accuracy', [])
            test_f1s = result.get('test_f1', [])
            epoch_times = result.get('epoch_times', [])
            learning_rates = result.get('learning_rates', [])
            
            for epoch, train_loss, train_acc, test_loss, test_acc, test_f1, epoch_time, lr in zip(
                epochs, train_losses, train_accs, test_losses, test_accs, test_f1s, epoch_times, learning_rates
            ):
                detailed_data.append({
                    'Experiment': exp_name,
                    'Epoch': epoch,
                    'Train Loss': train_loss,
                    'Train Accuracy (%)': train_acc,
                    'Test Loss': test_loss,
                    'Test Accuracy (%)': test_acc,
                    'Test F1 Score': test_f1,
                    'Epoch Time (s)': epoch_time,
                    'Learning Rate': lr,
                })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            df_detailed.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
            
            # Format the sheet
            worksheet = writer.sheets['Detailed_Metrics']
            for idx, col in enumerate(df_detailed.columns):
                column_width = max(df_detailed[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = column_width
    
    def _create_comparison_sheet(self, results: Dict[str, Dict], writer: pd.ExcelWriter):
        """Create comparison sheet with ranked results."""
        comparison_data = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            # Extract experiment components
            parts = exp_name.split('_')
            dataset = parts[0] if len(parts) > 0 else 'unknown'
            model = parts[1] if len(parts) > 1 else 'unknown'
            optimizer = parts[2] if len(parts) > 2 else 'unknown'
            
            comparison_data.append({
                'Dataset': dataset,
                'Model': model,
                'Optimizer': optimizer,
                'Best Test Accuracy (%)': result.get('best_test_accuracy', 0),
                'Final Test Accuracy (%)': result.get('test_accuracy', [0])[-1] if result.get('test_accuracy') else 0,
                'Final Test F1': result.get('test_f1', [0])[-1] if result.get('test_f1') else 0,
                'Total Training Time (s)': result.get('total_training_time', 0),
                'Avg Epoch Time (s)': np.mean(result.get('epoch_times', [0])),
                'Peak GPU Memory (GB)': max(result.get('memory_usage', [0])) if result.get('memory_usage') else 0,
                'Model Parameters': result.get('model_params', 0),
                'Model FLOPs': result.get('model_flops', 0),
                'Convergence Epoch': self._find_convergence_epoch(result.get('test_accuracy', []), threshold=95),
            })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Sort by accuracy
            df_comparison_sorted = df_comparison.sort_values('Best Test Accuracy (%)', ascending=False)
            df_comparison_sorted.to_excel(writer, sheet_name='Comparison', index=False)
            
            # Format the sheet
            worksheet = writer.sheets['Comparison']
            for idx, col in enumerate(df_comparison_sorted.columns):
                column_width = max(df_comparison_sorted[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = column_width
    
    def _create_statistical_sheet(self, results: Dict[str, Dict], writer: pd.ExcelWriter):
        """Create statistical analysis sheet."""
        statistical_data = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            # Calculate statistical metrics
            train_accs = result.get('train_accuracy', [])
            test_accs = result.get('test_accuracy', [])
            train_losses = result.get('train_loss', [])
            test_losses = result.get('test_loss', [])
            
            if train_accs and test_accs:
                # Accuracy statistics
                final_train_acc = train_accs[-1]
                final_test_acc = test_accs[-1]
                generalization_gap = final_train_acc - final_test_acc
                
                # Loss statistics
                final_train_loss = train_losses[-1] if train_losses else 0
                final_test_loss = test_losses[-1] if test_losses else 0
                
                # Convergence metrics
                convergence_epoch_95 = self._find_convergence_epoch(test_accs, threshold=95)
                convergence_epoch_90 = self._find_convergence_epoch(test_accs, threshold=90)
                convergence_epoch_80 = self._find_convergence_epoch(test_accs, threshold=80)
                
                statistical_data.append({
                    'Experiment': exp_name,
                    'Final Train Accuracy (%)': final_train_acc,
                    'Final Test Accuracy (%)': final_test_acc,
                    'Generalization Gap (%)': generalization_gap,
                    'Final Train Loss': final_train_loss,
                    'Final Test Loss': final_test_loss,
                    'Convergence Epoch (95%)': convergence_epoch_95,
                    'Convergence Epoch (90%)': convergence_epoch_90,
                    'Convergence Epoch (80%)': convergence_epoch_80,
                    'Accuracy Std (Last 5 Epochs)': np.std(test_accs[-5:]) if len(test_accs) >= 5 else np.std(test_accs),
                    'Loss Std (Last 5 Epochs)': np.std(test_losses[-5:]) if len(test_losses) >= 5 else np.std(test_losses),
                })
        
        if statistical_data:
            df_statistical = pd.DataFrame(statistical_data)
            df_statistical.to_excel(writer, sheet_name='Statistical_Analysis', index=False)
            
            # Format the sheet
            worksheet = writer.sheets['Statistical_Analysis']
            for idx, col in enumerate(df_statistical.columns):
                column_width = max(df_statistical[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = column_width
    
    def _create_computational_sheet(self, results: Dict[str, Dict], writer: pd.ExcelWriter):
        """Create computational metrics sheet."""
        computational_data = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            epoch_times = result.get('epoch_times', [])
            memory_usage = result.get('memory_usage', [])
            gradient_norms = result.get('gradient_norm', [])
            
            computational_data.append({
                'Experiment': exp_name,
                'Total Training Time (s)': result.get('total_training_time', 0),
                'Avg Epoch Time (s)': np.mean(epoch_times) if epoch_times else 0,
                'Std Epoch Time (s)': np.std(epoch_times) if epoch_times else 0,
                'Peak GPU Memory (GB)': max(memory_usage) if memory_usage and len(memory_usage) > 0 else 0,
                'Avg GPU Memory (GB)': np.mean(memory_usage) if memory_usage else 0,
                'Peak Gradient Norm': max(gradient_norms) if gradient_norms else 0,
                'Avg Gradient Norm': np.mean(gradient_norms) if gradient_norms else 0,
                'Model Parameters': result.get('model_params', 0),
                'Model FLOPs': result.get('model_flops', 0),
                'Parameters per Second': result.get('model_params', 0) / max(np.mean(epoch_times), 1e-6) if epoch_times else 0,
                'FLOPs per Second': result.get('model_flops', 0) / max(np.mean(epoch_times), 1e-6) if epoch_times else 0,
            })
        
        if computational_data:
            df_computational = pd.DataFrame(computational_data)
            df_computational.to_excel(writer, sheet_name='Computational_Metrics', index=False)
            
            # Format the sheet
            worksheet = writer.sheets['Computational_Metrics']
            for idx, col in enumerate(df_computational.columns):
                column_width = max(df_computational[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = column_width
    
    def _create_optimizer_metrics_sheet(self, results: Dict[str, Dict], writer: pd.ExcelWriter):
        """Create optimizer-specific metrics sheet."""
        optimizer_data = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            # Extract optimizer-specific metrics
            gradient_consistency = result.get('optimizer_gradient_consistency', [])
            trust_ratio = result.get('optimizer_trust_ratio', [])
            curvature_stats = result.get('optimizer_curvature_stats', [])
            
            if gradient_consistency:
                optimizer_data.append({
                    'Experiment': exp_name,
                    'Metric Type': 'Gradient Consistency',
                    'Mean': np.mean(gradient_consistency),
                    'Std': np.std(gradient_consistency),
                    'Min': np.min(gradient_consistency),
                    'Max': np.max(gradient_consistency),
                    'Final Value': gradient_consistency[-1] if gradient_consistency else 0,
                })
            
            if trust_ratio:
                optimizer_data.append({
                    'Experiment': exp_name,
                    'Metric Type': 'Trust Ratio',
                    'Mean': np.mean(trust_ratio),
                    'Std': np.std(trust_ratio),
                    'Min': np.min(trust_ratio),
                    'Max': np.max(trust_ratio),
                    'Final Value': trust_ratio[-1] if trust_ratio else 0,
                })
            
            if curvature_stats and isinstance(curvature_stats, list) and curvature_stats:
                # Handle curvature stats (list of dictionaries)
                last_stats = curvature_stats[-1] if isinstance(curvature_stats[-1], dict) else {}
                optimizer_data.append({
                    'Experiment': exp_name,
                    'Metric Type': 'Curvature Mean',
                    'Mean': last_stats.get('mean_curvature', 0),
                    'Std': last_stats.get('std_curvature', 0),
                    'Min': last_stats.get('min_curvature', 0),
                    'Max': last_stats.get('max_curvature', 0),
                    'Final Value': last_stats.get('mean_curvature', 0),
                })
        
        if optimizer_data:
            df_optimizer = pd.DataFrame(optimizer_data)
            df_optimizer.to_excel(writer, sheet_name='Optimizer_Metrics', index=False)
            
            # Format the sheet
            worksheet = writer.sheets['Optimizer_Metrics']
            for idx, col in enumerate(df_optimizer.columns):
                column_width = max(df_optimizer[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = column_width
    
    def _find_convergence_epoch(self, accuracies: List[float], threshold: float) -> int:
        """Find epoch where accuracy first reaches threshold."""
        for epoch, acc in enumerate(accuracies, 1):
            if acc >= threshold:
                return epoch
        return len(accuracies)
    
    def export_to_json(self, results: Dict[str, Dict], filename: str = 'experiment_results.json'):
        """
        Export results to JSON file.
        
        Args:
            results: Dictionary mapping experiment names to results
            filename: Output JSON filename
        """
        json_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for exp_name, result in results.items():
            serializable_results[exp_name] = self._make_serializable(result)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results exported to JSON: {json_path}")
        return json_path
    
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
    
    def generate_plots(self, results: Dict[str, Dict], plot_types: List[str] = None):
        """
        Generate plots from experiment results.
        
        Args:
            results: Dictionary mapping experiment names to results
            plot_types: List of plot types to generate. Options:
                       'loss', 'accuracy', 'f1', 'time', 'memory', 'gradients', 'all'
        """
        if plot_types is None:
            plot_types = ['all']
        
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        if 'all' in plot_types:
            plot_types = ['loss', 'accuracy', 'f1', 'time', 'memory', 'gradients']
        
        for plot_type in plot_types:
            if plot_type == 'loss':
                self._plot_loss_curves(results, plots_dir)
            elif plot_type == 'accuracy':
                self._plot_accuracy_curves(results, plots_dir)
            elif plot_type == 'f1':
                self._plot_f1_curves(results, plots_dir)
            elif plot_type == 'time':
                self._plot_time_comparison(results, plots_dir)
            elif plot_type == 'memory':
                self._plot_memory_usage(results, plots_dir)
            elif plot_type == 'gradients':
                self._plot_gradient_stats(results, plots_dir)
    
    def _plot_loss_curves(self, results: Dict[str, Dict], plots_dir: Path):
        """Plot training and test loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            epochs = list(range(1, len(result.get('train_loss', [])) + 1))
            train_losses = result.get('train_loss', [])
            test_losses = result.get('test_loss', [])
            
            if train_losses:
                axes[0].plot(epochs, train_losses, label=exp_name, marker='o', markersize=3, linewidth=1)
            if test_losses:
                axes[1].plot(epochs, test_losses, label=exp_name, marker='o', markersize=3, linewidth=1)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss Curves')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('Test Loss Curves')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_curves(self, results: Dict[str, Dict], plots_dir: Path):
        """Plot training and test accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            epochs = list(range(1, len(result.get('train_accuracy', [])) + 1))
            train_accs = result.get('train_accuracy', [])
            test_accs = result.get('test_accuracy', [])
            
            if train_accs:
                axes[0].plot(epochs, train_accs, label=exp_name, marker='o', markersize=3, linewidth=1)
            if test_accs:
                axes[1].plot(epochs, test_accs, label=exp_name, marker='o', markersize=3, linewidth=1)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Accuracy (%)')
        axes[0].set_title('Training Accuracy Curves')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Test Accuracy (%)')
        axes[1].set_title('Test Accuracy Curves')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_f1_curves(self, results: Dict[str, Dict], plots_dir: Path):
        """Plot test F1 score curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            epochs = list(range(1, len(result.get('test_f1', [])) + 1))
            test_f1s = result.get('test_f1', [])
            
            if test_f1s:
                ax.plot(epochs, test_f1s, label=exp_name, marker='o', markersize=3, linewidth=1)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test F1 Score')
        ax.set_title('Test F1 Score Curves')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'f1_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_time_comparison(self, results: Dict[str, Dict], plots_dir: Path):
        """Plot time comparison bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        exp_names = []
        total_times = []
        avg_epoch_times = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            exp_names.append(exp_name)
            total_time = result.get('total_training_time', 0)
            # Handle case where total_time might be a list
            if isinstance(total_time, list) and len(total_time) > 0:
                total_time = total_time[0]
            total_times.append(total_time)
            avg_epoch_times.append(np.mean(result.get('epoch_times', [0])))
        
        if exp_names:
            x = np.arange(len(exp_names))
            width = 0.35
            
            axes[0].bar(x - width/2, total_times, width, label='Total Time')
            axes[0].set_xlabel('Experiment')
            axes[0].set_ylabel('Total Training Time (s)')
            axes[0].set_title('Total Training Time Comparison')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(exp_names, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].bar(x - width/2, avg_epoch_times, width, label='Avg Epoch Time')
            axes[1].set_xlabel('Experiment')
            axes[1].set_ylabel('Average Epoch Time (s)')
            axes[1].set_title('Average Epoch Time Comparison')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(exp_names, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'time_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, results: Dict[str, Dict], plots_dir: Path):
        """Plot memory usage comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        exp_names = []
        peak_memory = []
        avg_memory = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            memory_usage = result.get('memory_usage', [])
            if memory_usage:
                exp_names.append(exp_name)
                peak_memory.append(max(memory_usage) if memory_usage and len(memory_usage) > 0 else 0)
                avg_memory.append(np.mean(memory_usage))
        
        if exp_names:
            x = np.arange(len(exp_names))
            width = 0.35
            
            ax.bar(x - width/2, peak_memory, width, label='Peak Memory')
            ax.bar(x + width/2, avg_memory, width, label='Average Memory')
            ax.set_xlabel('Experiment')
            ax.set_ylabel('GPU Memory Usage (GB)')
            ax.set_title('Memory Usage Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(exp_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'memory_usage.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_gradient_stats(self, results: Dict[str, Dict], plots_dir: Path):
        """Plot gradient statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            gradient_norms = result.get('gradient_norm', [])
            if gradient_norms:
                epochs = list(range(1, len(gradient_norms) + 1))
                axes[0].plot(epochs, gradient_norms, label=exp_name, marker='o', markersize=3, linewidth=1)
        
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Gradient Norm')
        axes[0].set_title('Gradient Norm Over Training')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Plot gradient mean/std if available
        for exp_name, result in results.items():
            if 'error' in result:
                continue
            
            gradient_means = result.get('gradient_mean', [])
            if gradient_means:
                epochs = list(range(1, len(gradient_means) + 1))
                axes[1].plot(epochs, gradient_means, label=f'{exp_name} (mean)', marker='o', markersize=3, linewidth=1)
        
        axes[1].set_xlabel('Batch')
        axes[1].set_ylabel('Gradient Mean')
        axes[1].set_title('Gradient Mean Over Training')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'gradient_stats.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: Dict[str, Dict], report_file: str = 'experiment_report.md'):
        """
        Generate a comprehensive markdown report.
        
        Args:
            results: Dictionary mapping experiment names to results
            report_file: Output markdown filename
        """
        report_path = self.output_dir / report_file
        
        with open(report_path, 'w') as f:
            f.write("# Experiment Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Experiment | Best Test Acc (%) | Final Test Acc (%) | Total Time (s) | Params | FLOPs |\n")
            f.write("|------------|-------------------|-------------------|----------------|--------|-------|\n")
            
            for exp_name, result in results.items():
                if 'error' in result:
                    f.write(f"| {exp_name} | ERROR | ERROR | ERROR | ERROR | ERROR |\n")
                else:
                    best_acc = result.get('best_test_accuracy', 0)
                    final_acc = result.get('test_accuracy', [0])[-1] if result.get('test_accuracy') else 0
                    total_time = result.get('total_training_time', 0)
                    # Handle case where total_time might be a list
                    if isinstance(total_time, list) and len(total_time) > 0:
                        total_time = total_time[0]
                    params = result.get('model_params', 0)
                    flops = result.get('model_flops', 0)
                    
                    f.write(f"| {exp_name} | {best_acc:.2f} | {final_acc:.2f} | {total_time:.2f} | {params:,} | {flops:,} |\n")
            
            f.write("\n")
            
            # Best performers
            f.write("## Best Performers\n\n")
            
            # By accuracy
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                best_by_acc = max(valid_results.items(), 
                                key=lambda x: x[1].get('best_test_accuracy', 0))
                f.write(f"### Best by Accuracy\n")
                f.write(f"- **Experiment**: {best_by_acc[0]}\n")
                f.write(f"- **Best Accuracy**: {best_by_acc[1].get('best_test_accuracy', 0):.2f}%\n")
                f.write(f"- **Final Accuracy**: {best_by_acc[1].get('test_accuracy', [0])[-1]:.2f}%\n")
                best_acc_time = best_by_acc[1].get('total_training_time', 0)
                if isinstance(best_acc_time, list) and len(best_acc_time) > 0:
                    best_acc_time = best_acc_time[0]
                f.write(f"- **Training Time**: {best_acc_time:.2f}s\n\n")
                
                # By speed
                best_by_speed = min(valid_results.items(),
                                  key=lambda x: x[1].get('total_training_time', float('inf')))
                f.write(f"### Fastest Training\n")
                f.write(f"- **Experiment**: {best_by_speed[0]}\n")
                best_speed_time = best_by_speed[1].get('total_training_time', 0)
                if isinstance(best_speed_time, list) and len(best_speed_time) > 0:
                    best_speed_time = best_speed_time[0]
                f.write(f"- **Training Time**: {best_speed_time:.2f}s\n")
                f.write(f"- **Best Accuracy**: {best_by_speed[1].get('best_test_accuracy', 0):.2f}%\n")
                f.write(f"- **Final Accuracy**: {best_by_speed[1].get('test_accuracy', [0])[-1]:.2f}%\n\n")
                
                # By memory efficiency
                if any('memory_usage' in v for v in valid_results.values()):
                    best_by_memory = min(valid_results.items(),
                                       key=lambda x: max(x[1].get('memory_usage', [0])) if x[1].get('memory_usage') else 0)
                    f.write(f"### Most Memory Efficient\n")
                    f.write(f"- **Experiment**: {best_by_memory[0]}\n")
                    f.write(f"- **Peak Memory**: {max(best_by_memory[1].get('memory_usage', [0])) if best_by_memory[1].get('memory_usage') else 0:.2f} GB\n")
                    best_memory_time = best_by_memory[1].get('total_training_time', 0)
                    if isinstance(best_memory_time, list) and len(best_memory_time) > 0:
                        best_memory_time = best_memory_time[0]
                    f.write(f"- **Training Time**: {best_memory_time:.2f}s\n")
                    f.write(f"- **Best Accuracy**: {best_by_memory[1].get('best_test_accuracy', 0):.2f}%\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the experiment results:\n\n")
            
            if valid_results:
                # Find best optimizer for each dataset/model combination
                dataset_model_combinations = {}
                for exp_name, result in valid_results.items():
                    parts = exp_name.split('_')
                    if len(parts) >= 3:
                        dataset_model = f"{parts[0]}_{parts[1]}"
                        optimizer = parts[2]
                        accuracy = result.get('best_test_accuracy', 0)
                        
                        if dataset_model not in dataset_model_combinations:
                            dataset_model_combinations[dataset_model] = {}
                        
                        dataset_model_combinations[dataset_model][optimizer] = accuracy
                
                f.write("### Best Optimizer per Dataset-Model Combination\n\n")
                for dataset_model, optimizers in dataset_model_combinations.items():
                    best_optimizer = max(optimizers.items(), key=lambda x: x[1])
                    f.write(f"- **{dataset_model}**: {best_optimizer[0]} ({best_optimizer[1]:.2f}% accuracy)\n")
                
                f.write("\n")
                
                # General recommendations
                f.write("### General Recommendations\n\n")
                f.write("1. **For maximum accuracy**: Use the optimizer with the highest test accuracy\n")
                f.write("2. **For fastest training**: Use the optimizer with the lowest training time\n")
                f.write("3. **For memory efficiency**: Use the optimizer with the lowest memory usage\n")
                f.write("4. **For stable training**: Monitor gradient norms and choose optimizer with stable gradients\n")
                f.write("5. **For generalization**: Choose optimizer with smallest generalization gap (train - test accuracy)\n")
            
            # Plots section
            f.write("\n## Generated Plots\n\n")
            f.write("The following plots have been generated:\n\n")
            f.write("- `loss_curves.png`: Training and test loss curves\n")
            f.write("- `accuracy_curves.png`: Training and test accuracy curves\n")
            f.write("- `f1_curves.png`: Test F1 score curves\n")
            f.write("- `time_comparison.png`: Training time comparison\n")
            f.write("- `memory_usage.png`: Memory usage comparison\n")
            f.write("- `gradient_stats.png`: Gradient statistics\n")
            f.write("\nPlots are saved in the `plots` directory.\n")
            
            # Files section
            f.write("\n## Generated Files\n\n")
            f.write("- `experiment_results.xlsx`: Excel file with all results\n")
            f.write("- `experiment_results.json`: JSON file with raw data\n")
            f.write("- `experiment_report.md`: This report\n")
            f.write("- `plots/`: Directory containing all generated plots\n")
            f.write("- `checkpoints/`: Directory containing model checkpoints\n")
        
        print(f"Report generated: {report_path}")
        return report_path