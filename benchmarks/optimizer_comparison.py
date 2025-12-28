"""
Optimizer comparison on synthetic functions.
Tests optimizers on standard optimization test functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import json
import time

# Import optimizers
import sys
sys.path.append('..')
from optimizers.amcas import AMCAS
from optimizers.ultron import ULTRON
from optimizers.ultron_v2 import ULTRON_V2


class OptimizerComparison:
    """
    Compares optimizers on synthetic test functions.
    """
    
    def __init__(self, output_dir='optimizer_comparison'):
        """
        Initialize optimizer comparison.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Available optimizers
        self.optimizer_registry = {
            'AMCAS': AMCAS,
            'ULTRON': ULTRON,
            'ULTRON_V2': ULTRON_V2,
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
            'ULTRON_V2': {
                'betas': (0.9, 0.999),
                'clip_threshold': 1.0,
                'normalize_gradients': True,
                'normalization_strategy': 'rms',
                'adaptive_clipping': True,
                'clip_alpha': 0.99,
                'clip_percentile': 95.0,
                'state_precision': 'fp32',
                'momentum_correction': True,
            },
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
        
        # Test functions
        self.test_functions = {
            'rosenbrock': self.rosenbrock_function,
            'rastrigin': self.rastrigin_function,
            'ackley': self.ackley_function,
            'sphere': self.sphere_function,
            'beale': self.beale_function,
            'goldstein_price': self.goldstein_price_function,
        }
    
    def rosenbrock_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rosenbrock function (banana function).
        Global minimum at (1, 1) with value 0.
        """
        x1, x2 = x[0], x[1]
        return 100 * (x2 - x1**2)**2 + (1 - x1)**2
    
    def rastrigin_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rastrigin function.
        Global minimum at (0, 0) with value 0.
        """
        A = 10
        n = len(x)
        return A * n + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x))
    
    def ackley_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ackley function.
        Global minimum at (0, 0) with value 0.
        """
        a = 20
        b = 0.2
        c = 2 * torch.pi
        n = len(x)
        
        sum1 = torch.sum(x**2)
        sum2 = torch.sum(torch.cos(c * x))
        
        term1 = -a * torch.exp(-b * torch.sqrt(sum1 / n))
        term2 = -torch.exp(sum2 / n)
        
        return term1 + term2 + a + torch.exp(torch.tensor(1.0))
    
    def sphere_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sphere function.
        Global minimum at (0, 0) with value 0.
        """
        return torch.sum(x**2)
    
    def beale_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Beale function.
        Global minimum at (3, 0.5) with value 0.
        """
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1*x2)**2
        term2 = (2.25 - x1 + x1*x2**2)**2
        term3 = (2.625 - x1 + x1*x2**3)**2
        return term1 + term2 + term3
    
    def goldstein_price_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Goldstein-Price function.
        Global minimum at (0, -1) with value 3.
        """
        x1, x2 = x[0], x[1]
        
        term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        
        return term1 * term2
    
    def test_optimizer_on_function(self, optimizer_name: str, function_name: str, 
                                   initial_point: torch.Tensor, num_iterations: int = 1000,
                                   learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Test an optimizer on a specific function.
        
        Args:
            optimizer_name: Name of the optimizer to test
            function_name: Name of the test function
            initial_point: Initial point for optimization
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary with optimization results
        """
        print(f"  Testing {optimizer_name} on {function_name} function...")
        
        # Get function
        if function_name not in self.test_functions:
            raise ValueError(f"Unknown function: {function_name}. Available: {list(self.test_functions.keys())}")
        
        function = self.test_functions[function_name]
        
        # Create parameter
        x = torch.nn.Parameter(initial_point.clone())
        
        # Create optimizer
        optimizer_class = self.optimizer_registry[optimizer_name]
        optimizer_params = self.default_optimizer_params.get(optimizer_name, {}).copy()
        
        if optimizer_name == 'SGD+Momentum':
            optimizer = optimizer_class([x], lr=learning_rate, **optimizer_params)
        else:
            optimizer = optimizer_class([x], lr=learning_rate, **optimizer_params)
        
        # Track optimization progress
        history = {
            'iterations': [],
            'values': [],
            'parameters': [],
            'gradients': [],
            'convergence': False,
            'final_value': 0,
            'final_parameters': None,
            'iterations_to_convergence': None,
        }
        
        # Optimization loop
        for i in range(num_iterations):
            optimizer.zero_grad()
            value = function(x)
            value.backward()
            optimizer.step()
            
            # Record history
            history['iterations'].append(i)
            history['values'].append(value.item())
            history['parameters'].append(x.detach().clone().numpy())
            history['gradients'].append(x.grad.detach().clone().numpy() if x.grad is not None else np.zeros_like(x.detach().numpy()))
            
            # Check convergence
            if value.item() < 1e-6 and history['iterations_to_convergence'] is None:
                history['convergence'] = True
                history['iterations_to_convergence'] = i + 1
        
        # Final results
        history['final_value'] = history['values'][-1]
        history['final_parameters'] = history['parameters'][-1]
        
        return history
    
    def compare_optimizers_on_function(self, function_name: str, initial_point: torch.Tensor,
                                      num_iterations: int = 1000, learning_rate: float = 0.01) -> Dict[str, Dict]:
        """
        Compare all optimizers on a specific function.
        
        Args:
            function_name: Name of the test function
            initial_point: Initial point for optimization
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizers
            
        Returns:
            Dictionary mapping optimizer names to optimization histories
        """
        print(f"\nComparing optimizers on {function_name} function...")
        print(f"Initial point: {initial_point.tolist()}")
        print(f"Iterations: {num_iterations}, Learning rate: {learning_rate}")
        
        results = {}
        
        for optimizer_name in self.optimizer_registry.keys():
            try:
                history = self.test_optimizer_on_function(
                    optimizer_name, function_name, initial_point, num_iterations, learning_rate
                )
                results[optimizer_name] = history
                
                print(f"  {optimizer_name}: Final value = {history['final_value']:.6f}, "
                      f"Converged = {history['convergence']}, "
                      f"Iterations to converge = {history['iterations_to_convergence']}")
                
            except Exception as e:
                print(f"  Error testing {optimizer_name}: {e}")
                results[optimizer_name] = {'error': str(e)}
        
        return results
    
    def run_all_comparisons(self, num_iterations: int = 1000) -> Dict[str, Dict]:
        """
        Run comparisons on all test functions.
        
        Args:
            num_iterations: Number of optimization iterations
            
        Returns:
            Dictionary mapping function names to optimizer comparison results
        """
        print(f"\n{'='*80}")
        print("Running Optimizer Comparisons on Synthetic Functions")
        print(f"{'='*80}")
        
        all_results = {}
        
        # Define initial points for each function
        initial_points = {
            'rosenbrock': torch.tensor([-1.5, 2.0]),
            'rastrigin': torch.tensor([5.0, 5.0]),
            'ackley': torch.tensor([5.0, 5.0]),
            'sphere': torch.tensor([3.0, 3.0]),
            'beale': torch.tensor([1.0, 1.0]),
            'goldstein_price': torch.tensor([1.0, 1.0]),
        }
        
        # Learning rates for each function
        learning_rates = {
            'rosenbrock': 0.001,
            'rastrigin': 0.01,
            'ackley': 0.01,
            'sphere': 0.1,
            'beale': 0.01,
            'goldstein_price': 0.001,
        }
        
        for function_name in self.test_functions.keys():
            print(f"\n{'='*60}")
            print(f"Function: {function_name}")
            print(f"{'='*60}")
            
            initial_point = initial_points.get(function_name, torch.tensor([1.0, 1.0]))
            lr = learning_rates.get(function_name, 0.01)
            
            results = self.compare_optimizers_on_function(
                function_name, initial_point, num_iterations, lr
            )
            all_results[function_name] = results
            
            # Save individual function results
            self._save_function_results(function_name, results)
        
        # Generate comparison report
        self._generate_comparison_report(all_results, num_iterations)
        
        return all_results
    
    def _save_function_results(self, function_name: str, results: Dict[str, Dict]):
        """Save results for a specific function."""
        result_file = self.output_dir / f'{function_name}_comparison.json'
        
        # Convert to serializable format
        serializable_results = {}
        for optimizer_name, history in results.items():
            if 'error' in history:
                serializable_results[optimizer_name] = {'error': history['error']}
            else:
                serializable_results[optimizer_name] = {
                    'iterations': history['iterations'],
                    'values': history['values'],
                    'parameters': [p.tolist() for p in history['parameters']],
                    'gradients': [g.tolist() for g in history['gradients']],
                    'convergence': history['convergence'],
                    'final_value': history['final_value'],
                    'final_parameters': history['final_parameters'].tolist() if history['final_parameters'] is not None else None,
                    'iterations_to_convergence': history['iterations_to_convergence'],
                }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"  Results saved to: {result_file}")
    
    def _generate_comparison_report(self, all_results: Dict[str, Dict], num_iterations: int):
        """Generate comparison report for all functions."""
        report_path = self.output_dir / 'optimizer_comparison_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Optimizer Comparison Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Iterations per function: {num_iterations}\n\n")
            
            # Summary table for each function
            for function_name, results in all_results.items():
                f.write(f"## {function_name.title()} Function\n\n")
                
                f.write("| Optimizer | Final Value | Converged | Iterations to Converge | Final Parameters |\n")
                f.write("|-----------|-------------|-----------|------------------------|------------------|\n")
                
                for optimizer_name, history in results.items():
                    if 'error' in history:
                        f.write(f"| {optimizer_name} | ERROR | ERROR | ERROR | ERROR |\n")
                    else:
                        final_value = history['final_value']
                        converged = history['convergence']
                        iterations = history['iterations_to_convergence']
                        params = history['final_parameters']
                        
                        f.write(f"| {optimizer_name} | {final_value:.6f} | {converged} | {iterations if converged else 'N/A'} | {params} |\n")
                
                f.write("\n")
            
            # Overall performance comparison
            f.write("## Overall Performance Comparison\n\n")
            
            # Calculate scores for each optimizer
            optimizer_scores = {}
            for optimizer_name in self.optimizer_registry.keys():
                optimizer_scores[optimizer_name] = {
                    'total_functions': 0,
                    'converged_functions': 0,
                    'total_iterations': 0,
                    'average_final_value': 0,
                }
            
            for function_name, results in all_results.items():
                for optimizer_name, history in results.items():
                    if 'error' in history:
                        continue
                    
                    optimizer_scores[optimizer_name]['total_functions'] += 1
                    if history['convergence']:
                        optimizer_scores[optimizer_name]['converged_functions'] += 1
                        optimizer_scores[optimizer_name]['total_iterations'] += history['iterations_to_convergence']
                    optimizer_scores[optimizer_name]['average_final_value'] += history['final_value']
            
            # Normalize scores
            for optimizer_name in optimizer_scores.keys():
                if optimizer_scores[optimizer_name]['total_functions'] > 0:
                    optimizer_scores[optimizer_name]['average_final_value'] /= optimizer_scores[optimizer_name]['total_functions']
                    if optimizer_scores[optimizer_name]['converged_functions'] > 0:
                        optimizer_scores[optimizer_name]['average_iterations'] = (
                            optimizer_scores[optimizer_name]['total_iterations'] / 
                            optimizer_scores[optimizer_name]['converged_functions']
                        )
                    else:
                        optimizer_scores[optimizer_name]['average_iterations'] = float('inf')
            
            # Sort by convergence rate
            sorted_by_convergence = sorted(
                optimizer_scores.items(),
                key=lambda x: (x[1]['converged_functions'], -x[1]['average_iterations']),
                reverse=True
            )
            
            f.write("### Convergence Performance\n\n")
            f.write("| Optimizer | Converged Functions | Total Functions | Convergence Rate | Avg Iterations to Converge |\n")
            f.write("|-----------|-------------------|----------------|------------------|---------------------------|\n")
            
            for optimizer_name, scores in sorted_by_convergence:
                convergence_rate = scores['converged_functions'] / scores['total_functions'] if scores['total_functions'] > 0 else 0
                avg_iterations = scores['average_iterations'] if scores['average_iterations'] != float('inf') else 'N/A'
                
                f.write(f"| {optimizer_name} | {scores['converged_functions']} | {scores['total_functions']} | {convergence_rate:.2%} | {avg_iterations} |\n")
            
            f.write("\n")
            
            # Sort by final value (lower is better)
            sorted_by_value = sorted(
                optimizer_scores.items(),
                key=lambda x: x[1]['average_final_value']
            )
            
            f.write("### Final Value Performance (Lower is Better)\n\n")
            f.write("| Optimizer | Average Final Value |\n")
            f.write("|-----------|-------------------|\n")
            
            for optimizer_name, scores in sorted_by_value:
                f.write(f"| {optimizer_name} | {scores['average_final_value']:.6f} |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if sorted_by_convergence:
                best_convergence = sorted_by_convergence[0][0]
                best_value = sorted_by_value[0][0]
                
                f.write(f"1. **Best convergence rate**: {best_convergence}\n")
                f.write(f"2. **Best final values**: {best_value}\n")
                f.write("3. **For difficult optimization landscapes**: Choose optimizers with good convergence rates\n")
                f.write("4. **For smooth convex functions**: SGD with momentum often works well\n")
                f.write("5. **For non-convex functions with many local minima**: Adaptive methods like Adam/AMCAS perform better\n")
                f.write("6. **For noisy gradients**: RMSprop and Adagrad can be more stable\n")
                f.write("7. **For saddle points**: Adaptive methods with momentum help escape saddle points\n")
            
            # Generate plots
            f.write("\n## Generated Plots\n\n")
            f.write("The following plots have been generated:\n\n")
            for function_name in all_results.keys():
                f.write(f"- `{function_name}_convergence.png`: Convergence curves for {function_name} function\n")
                f.write(f"- `{function_name}_trajectory.png`: Optimization trajectories for {function_name} function\n")
            
            # Files generated
            f.write("\n## Generated Files\n\n")
            f.write("The following files have been generated:\n\n")
            for function_name in all_results.keys():
                f.write(f"- `{function_name}_comparison.json`: Detailed results for {function_name} function\n")
            f.write(f"- `optimizer_comparison_report.md`: This report\n")
            f.write(f"- `plots/`: Directory containing all generated plots\n")
        
        print(f"\nComparison report saved to: {report_path}")
        
        # Generate plots
        self._generate_comparison_plots(all_results)
        
        return report_path
    
    def _generate_comparison_plots(self, all_results: Dict[str, Dict]):
        """Generate comparison plots for all functions."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Colors for different optimizers
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimizer_registry)))
            optimizer_colors = {name: color for name, color in zip(self.optimizer_registry.keys(), colors)}
            
            for function_name, results in all_results.items():
                # Skip if all results have errors
                if all('error' in r for r in results.values()):
                    continue
                
                # Plot 1: Convergence curves
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for optimizer_name, history in results.items():
                    if 'error' in history:
                        continue
                    
                    iterations = history['iterations']
                    values = history['values']
                    
                    ax.semilogy(iterations, values, label=optimizer_name, 
                               color=optimizer_colors.get(optimizer_name, 'black'),
                               linewidth=2, alpha=0.8)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Function Value (log scale)')
                ax.set_title(f'{function_name.title()} Function - Convergence Curves')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / f'{function_name}_convergence.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Plot 2: Optimization trajectories (for 2D functions)
                if function_name in ['rosenbrock', 'rastrigin', 'ackley', 'sphere', 'beale', 'goldstein_price']:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Create contour plot of the function
                    x = np.linspace(-5, 5, 100)
                    y = np.linspace(-5, 5, 100)
                    X, Y = np.meshgrid(x, y)
                    
                    # Evaluate function on grid
                    Z = np.zeros_like(X)
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
                            Z[i, j] = self.test_functions[function_name](point).item()
                    
                    # Plot contours
                    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
                    plt.colorbar(contour, ax=ax)
                    
                    # Plot optimization trajectories
                    for optimizer_name, history in results.items():
                        if 'error' in history:
                            continue
                        
                        parameters = history['parameters']
                        if len(parameters) > 0 and len(parameters[0]) == 2:
                            trajectory_x = [p[0] for p in parameters]
                            trajectory_y = [p[1] for p in parameters]
                            
                            ax.plot(trajectory_x, trajectory_y, label=optimizer_name,
                                   color=optimizer_colors.get(optimizer_name, 'black'),
                                   linewidth=2, alpha=0.8, marker='o', markersize=3)
                            
                            # Mark start and end points
                            ax.scatter(trajectory_x[0], trajectory_y[0], 
                                     color=optimizer_colors.get(optimizer_name, 'black'),
                                     s=100, marker='s', edgecolors='white', linewidth=2)
                            ax.scatter(trajectory_x[-1], trajectory_y[-1],
                                     color=optimizer_colors.get(optimizer_name, 'black'),
                                     s=100, marker='*', edgecolors='white', linewidth=2)
                    
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title(f'{function_name.title()} Function - Optimization Trajectories')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / f'{function_name}_trajectory.png', dpi=150, bbox_inches='tight')
                    plt.close()
            
            # Plot 3: Overall performance comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Calculate metrics for each optimizer
            convergence_rates = []
            avg_iterations = []
            avg_final_values = []
            optimizer_names = []
            
            for optimizer_name in self.optimizer_registry.keys():
                total_functions = 0
                converged_functions = 0
                total_iterations = 0
                total_final_value = 0
                
                for function_name, results in all_results.items():
                    if optimizer_name in results and 'error' not in results[optimizer_name]:
                        total_functions += 1
                        history = results[optimizer_name]
                        total_final_value += history['final_value']
                        
                        if history['convergence']:
                            converged_functions += 1
                            total_iterations += history['iterations_to_convergence']
                
                if total_functions > 0:
                    optimizer_names.append(optimizer_name)
                    convergence_rates.append(converged_functions / total_functions)
                    avg_iterations.append(total_iterations / converged_functions if converged_functions > 0 else float('inf'))
                    avg_final_values.append(total_final_value / total_functions)
            
            # Plot convergence rates
            if optimizer_names:
                x = np.arange(len(optimizer_names))
                axes[0, 0].bar(x, convergence_rates, color='skyblue')
                axes[0, 0].set_xlabel('Optimizer')
                axes[0, 0].set_ylabel('Convergence Rate')
                axes[0, 0].set_title('Convergence Rate Comparison')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(optimizer_names, rotation=45, ha='right')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot average iterations to converge (finite values only)
                finite_indices = [i for i, val in enumerate(avg_iterations) if val != float('inf')]
                if finite_indices:
                    finite_names = [optimizer_names[i] for i in finite_indices]
                    finite_values = [avg_iterations[i] for i in finite_indices]
                    
                    x_finite = np.arange(len(finite_names))
                    axes[0, 1].bar(x_finite, finite_values, color='lightcoral')
                    axes[0, 1].set_xlabel('Optimizer')
                    axes[0, 1].set_ylabel('Average Iterations to Converge')
                    axes[0, 1].set_title('Convergence Speed Comparison')
                    axes[0, 1].set_xticks(x_finite)
                    axes[0, 1].set_xticklabels(finite_names, rotation=45, ha='right')
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Plot average final values
                axes[1, 0].bar(x, avg_final_values, color='lightgreen')
                axes[1, 0].set_xlabel('Optimizer')
                axes[1, 0].set_ylabel('Average Final Value')
                axes[1, 0].set_title('Final Value Comparison (Lower is Better)')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(optimizer_names, rotation=45, ha='right')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot combined score (lower is better)
                # Score = avg_final_value * (1 / convergence_rate) * avg_iterations
                combined_scores = []
                for i in range(len(optimizer_names)):
                    if avg_iterations[i] != float('inf'):
                        score = avg_final_values[i] * (1 / convergence_rates[i]) * avg_iterations[i]
                        combined_scores.append(score)
                    else:
                        combined_scores.append(float('inf'))
                
                finite_scores = [s for s in combined_scores if s != float('inf')]
                finite_names_scores = [optimizer_names[i] for i, s in enumerate(combined_scores) if s != float('inf')]
                
                if finite_scores:
                    x_scores = np.arange(len(finite_names_scores))
                    axes[1, 1].bar(x_scores, finite_scores, color='gold')
                    axes[1, 1].set_xlabel('Optimizer')
                    axes[1, 1].set_ylabel('Combined Score')
                    axes[1, 1].set_title('Overall Performance Score (Lower is Better)')
                    axes[1, 1].set_xticks(x_scores)
                    axes[1, 1].set_xticklabels(finite_names_scores, rotation=45, ha='right')
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'overall_performance.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"Comparison plots saved to: {plots_dir}/")
            
        except ImportError:
            print("Warning: matplotlib not installed, skipping plot generation")
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")


def main():
    """Main function to run optimizer comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare optimizers on synthetic functions')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of iterations per function (default: 1000)')
    parser.add_argument('--output', type=str, default='optimizer_comparison',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create comparison
    comparison = OptimizerComparison(output_dir=args.output)
    
    # Run comparisons
    results = comparison.run_all_comparisons(num_iterations=args.iterations)
    
    print(f"\nOptimizer comparison completed!")
    print(f"Results saved to: {args.output}/")


if __name__ == '__main__':
    main()