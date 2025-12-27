import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import json
from optimizers.amcas import AMCAS
from optimizers.utils import get_optimizer_statistics


class SyntheticBenchmark:
    """Benchmark AMCAS against other optimizers on synthetic functions."""
    
    def __init__(self, output_dir='benchmark_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def rosenbrock_function(self, x, y, a=1, b=100):
        """Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2"""
        return (a - x)**2 + b * (y - x**2)**2
    
    def quadratic_bowl(self, x, center=0.0, scale=1.0):
        """Quadratic bowl: f(x) = scale * ||x - center||^2"""
        return scale * torch.sum((x - center)**2)
    
    def saddle_function(self, x, y):
        """Saddle point function: f(x,y) = x^2 - y^2"""
        return x**2 - y**2
    
    def test_rosenbrock(self, num_iterations=1000, lr=0.001):
        """Test on Rosenbrock function (non-convex, hard to optimize)."""
        print("Testing on Rosenbrock function...")
        
        # Initialize parameters
        x = torch.tensor([0.0], requires_grad=True)
        y = torch.tensor([0.0], requires_grad=True)
        
        optimizers = {
            'AMCAS': AMCAS([x, y], lr=lr),
            'Adam': torch.optim.Adam([x, y], lr=lr),
            'SGD': torch.optim.SGD([x, y], lr=lr),
            'SGD+Momentum': torch.optim.SGD([x, y], lr=lr, momentum=0.9),
            'RMSprop': torch.optim.RMSprop([x, y], lr=lr),
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            print(f"\nTesting {name}...")
            
            # Reset parameters
            x.data = torch.tensor([0.0])
            y.data = torch.tensor([0.0])
            x.grad = None
            y.grad = None
            
            losses = []
            times = []
            
            start_time = time.time()
            for i in range(num_iterations):
                optimizer.zero_grad()
                loss = self.rosenbrock_function(x, y)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                times.append(time.time() - start_time)
                
                if i % 100 == 0:
                    print(f"  Iteration {i}: loss = {loss.item():.6f}, x = {x.item():.6f}, y = {y.item():.6f}")
            
            results[name] = {
                'losses': losses,
                'times': times,
                'final_loss': losses[-1],
                'final_x': x.item(),
                'final_y': y.item(),
                'convergence_iter': self._find_convergence_iter(losses, threshold=1e-3),
            }
        
        self._plot_results(results, 'Rosenbrock Function')
        self._save_results(results, 'rosenbrock')
        return results
    
    def test_quadratic_bowl(self, num_iterations=500, lr=0.01):
        """Test on quadratic bowl (convex, easy to optimize)."""
        print("\nTesting on Quadratic Bowl function...")
        
        # Initialize parameters (10-dimensional)
        x = torch.randn(10, requires_grad=True)
        center = torch.zeros(10)
        
        optimizers = {
            'AMCAS': AMCAS([x], lr=lr),
            'Adam': torch.optim.Adam([x], lr=lr),
            'SGD': torch.optim.SGD([x], lr=lr),
            'SGD+Momentum': torch.optim.SGD([x], lr=lr, momentum=0.9),
            'RMSprop': torch.optim.RMSprop([x], lr=lr),
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            print(f"\nTesting {name}...")
            
            # Reset parameters
            x.data = torch.randn(10)
            x.grad = None
            
            losses = []
            times = []
            
            start_time = time.time()
            for i in range(num_iterations):
                optimizer.zero_grad()
                loss = self.quadratic_bowl(x, center=center)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                times.append(time.time() - start_time)
            
            results[name] = {
                'losses': losses,
                'times': times,
                'final_loss': losses[-1],
                'final_distance': torch.norm(x - center).item(),
                'convergence_iter': self._find_convergence_iter(losses, threshold=1e-4),
            }
        
        self._plot_results(results, 'Quadratic Bowl')
        self._save_results(results, 'quadratic_bowl')
        return results
    
    def test_saddle_point(self, num_iterations=500, lr=0.01):
        """Test on saddle point function."""
        print("\nTesting on Saddle Point function...")
        
        # Initialize parameters
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([1.0], requires_grad=True)
        
        optimizers = {
            'AMCAS': AMCAS([x, y], lr=lr),
            'Adam': torch.optim.Adam([x, y], lr=lr),
            'SGD': torch.optim.SGD([x, y], lr=lr),
            'SGD+Momentum': torch.optim.SGD([x, y], lr=lr, momentum=0.9),
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            print(f"\nTesting {name}...")
            
            # Reset parameters
            x.data = torch.tensor([1.0])
            y.data = torch.tensor([1.0])
            x.grad = None
            y.grad = None
            
            losses = []
            times = []
            
            start_time = time.time()
            for i in range(num_iterations):
                optimizer.zero_grad()
                loss = self.saddle_function(x, y)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                times.append(time.time() - start_time)
            
            results[name] = {
                'losses': losses,
                'times': times,
                'final_loss': losses[-1],
                'final_x': x.item(),
                'final_y': y.item(),
                'convergence_iter': self._find_convergence_iter(losses, threshold=1e-3),
            }
        
        self._plot_results(results, 'Saddle Point Function')
        self._save_results(results, 'saddle_point')
        return results
    
    def test_high_condition_number(self, num_iterations=1000, lr=0.001):
        """Test on function with high condition number (ill-conditioned)."""
        print("\nTesting on High Condition Number function...")
        
        # Create ill-conditioned quadratic: f(x) = x^T A x where A has high condition number
        torch.manual_seed(42)
        n = 20
        # Create diagonal matrix with eigenvalues from 1 to 1000
        eigenvalues = torch.linspace(1, 1000, n)
        A = torch.diag(eigenvalues)
        
        # Initialize parameters
        x = torch.randn(n, requires_grad=True)
        x_opt = torch.zeros(n)  # Optimal point at origin
        
        optimizers = {
            'AMCAS': AMCAS([x], lr=lr),
            'Adam': torch.optim.Adam([x], lr=lr),
            'SGD': torch.optim.SGD([x], lr=lr),
            'SGD+Momentum': torch.optim.SGD([x], lr=lr, momentum=0.9),
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            print(f"\nTesting {name}...")
            
            # Reset parameters
            x.data = torch.randn(n)
            x.grad = None
            
            losses = []
            times = []
            
            start_time = time.time()
            for i in range(num_iterations):
                optimizer.zero_grad()
                loss = x @ A @ x  # Quadratic form
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                times.append(time.time() - start_time)
                
                if i % 200 == 0:
                    print(f"  Iteration {i}: loss = {loss.item():.6f}")
            
            results[name] = {
                'losses': losses,
                'times': times,
                'final_loss': losses[-1],
                'final_distance': torch.norm(x - x_opt).item(),
                'condition_number': torch.max(eigenvalues) / torch.min(eigenvalues),
                'convergence_iter': self._find_convergence_iter(losses, threshold=1e-3),
            }
        
        self._plot_results(results, 'High Condition Number Function')
        self._save_results(results, 'high_condition')
        return results
    
    def test_noisy_gradient(self, num_iterations=500, lr=0.01, noise_std=0.1):
        """Test with noisy gradients."""
        print("\nTesting with Noisy Gradients...")
        
        # Simple quadratic with gradient noise
        x = torch.tensor([2.0], requires_grad=True)
        target = torch.tensor([0.0])
        
        optimizers = {
            'AMCAS': AMCAS([x], lr=lr),
            'Adam': torch.optim.Adam([x], lr=lr),
            'SGD': torch.optim.SGD([x], lr=lr),
            'SGD+Momentum': torch.optim.SGD([x], lr=lr, momentum=0.9),
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            print(f"\nTesting {name}...")
            
            # Reset parameters
            x.data = torch.tensor([2.0])
            x.grad = None
            
            losses = []
            times = []
            gradient_norms = []
            
            start_time = time.time()
            for i in range(num_iterations):
                optimizer.zero_grad()
                loss = (x - target).pow(2).sum()
                loss.backward()
                
                # Add noise to gradient
                with torch.no_grad():
                    noise = torch.randn_like(x.grad) * noise_std
                    x.grad += noise
                    gradient_norms.append(torch.norm(x.grad).item())
                
                optimizer.step()
                
                losses.append(loss.item())
                times.append(time.time() - start_time)
            
            results[name] = {
                'losses': losses,
                'times': times,
                'gradient_norms': gradient_norms,
                'final_loss': losses[-1],
                'final_x': x.item(),
                'loss_std': np.std(losses[-100:]),
                'grad_norm_std': np.std(gradient_norms[-100:]),
                'convergence_iter': self._find_convergence_iter(losses, threshold=1e-3),
            }
        
        self._plot_results(results, 'Noisy Gradient Test')
        self._save_results(results, 'noisy_gradient')
        return results
    
    def _find_convergence_iter(self, losses, threshold=1e-3):
        """Find iteration where loss first goes below threshold."""
        for i, loss in enumerate(losses):
            if loss < threshold:
                return i
        return len(losses)
    
    def _plot_results(self, results, title):
        """Plot comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Loss vs iterations
        ax = axes[0, 0]
        for name, data in results.items():
            ax.plot(data['losses'], label=name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Loss Convergence')
        
        # Plot 2: Loss vs time
        ax = axes[0, 1]
        for name, data in results.items():
            ax.plot(data['times'], data['losses'], label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Loss vs Time')
        
        # Plot 3: Final loss comparison
        ax = axes[1, 0]
        names = list(results.keys())
        final_losses = [results[name]['final_loss'] for name in names]
        convergence_iters = [results[name].get('convergence_iter', len(results[name]['losses'])) 
                            for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, final_losses, width, label='Final Loss')
        ax.bar(x + width/2, convergence_iters, width, label='Convergence Iter')
        ax.set_xlabel('Optimizer')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Final Performance')
        
        # Plot 4: Gradient norms (if available)
        ax = axes[1, 1]
        for name, data in results.items():
            if 'gradient_norms' in data:
                ax.plot(data['gradient_norms'], label=name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        if any('gradient_norms' in data for data in results.values()):
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Gradient Norms')
        else:
            ax.text(0.5, 0.5, 'No gradient norm data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png', dpi=150)
        plt.close()
    
    def _save_results(self, results, filename):
        """Save results to JSON file."""
        # Convert torch tensors to Python floats
        serializable_results = {}
        for name, data in results.items():
            serializable_results[name] = {}
            for key, value in data.items():
                if isinstance(value, list):
                    # Convert list of torch tensors or floats
                    serializable_results[name][key] = [float(v) if isinstance(v, (torch.Tensor, np.generic)) else v 
                                                      for v in value]
                elif isinstance(value, (torch.Tensor, np.generic)):
                    serializable_results[name][key] = float(value)
                else:
                    serializable_results[name][key] = value
        
        with open(self.output_dir / f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
    
    def run_all_benchmarks(self):
        """Run all benchmark tests."""
        print("=" * 60)
        print("Running Synthetic Benchmark Tests")
        print("=" * 60)
        
        all_results = {}
        
        # Run all tests
        all_results['rosenbrock'] = self.test_rosenbrock()
        all_results['quadratic_bowl'] = self.test_quadratic_bowl()
        all_results['saddle_point'] = self.test_saddle_point()
        all_results['high_condition'] = self.test_high_condition_number()
        all_results['noisy_gradient'] = self.test_noisy_gradient()
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, all_results):
        """Generate a summary report of all benchmarks."""
        print("\n" + "=" * 60)
        print("Benchmark Summary Report")
        print("=" * 60)
        
        summary = {}
        for test_name, results in all_results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            test_summary = {}
            for optimizer_name, data in results.items():
                final_loss = data['final_loss']
                convergence_iter = data.get('convergence_iter', 'N/A')
                print(f"{optimizer_name:15} Final Loss: {final_loss:.6e} Converged at: {convergence_iter}")
                
                test_summary[optimizer_name] = {
                    'final_loss': final_loss,
                    'convergence_iter': convergence_iter,
                }
            
            # Find best optimizer for this test
            best_optimizer = min(results.items(), 
                                key=lambda x: x[1]['final_loss'])[0]
            print(f"Best optimizer: {best_optimizer}")
            
            summary[test_name] = {
                'best_optimizer': best_optimizer,
                'results': test_summary,
            }
        
        # Save summary
        with open(self.output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Summary saved to benchmark_results/summary.json")
        print("=" * 60)


def main():
    """Run benchmark tests."""
    benchmark = SyntheticBenchmark()
    results = benchmark.run_all_benchmarks()
    
    print("\nBenchmark completed successfully!")
    print(f"Results saved to: {benchmark.output_dir}/")
    
    # Print final comparison
    print("\nFinal Comparison (lower is better):")
    print("-" * 60)
    
    # Collect final losses for each optimizer across all tests
    optimizer_losses = {}
    for test_name, test_results in results.items():
        for optimizer_name, data in test_results.items():
            if optimizer_name not in optimizer_losses:
                optimizer_losses[optimizer_name] = []
            optimizer_losses[optimizer_name].append(data['final_loss'])
    
    # Calculate average rank across all tests
    optimizer_ranks = {}
    for optimizer_name, losses in optimizer_losses.items():
        avg_loss = np.mean(losses)
        optimizer_ranks[optimizer_name] = avg_loss
    
    # Sort by average loss
    sorted_ranks = sorted(optimizer_ranks.items(), key=lambda x: x[1])
    
    print("\nAverage Final Loss Across All Tests:")
    for optimizer_name, avg_loss in sorted_ranks:
        print(f"{optimizer_name:15} {avg_loss:.6e}")
    
    best_optimizer = sorted_ranks[0][0]
    print(f"\nBest overall optimizer: {best_optimizer}")


if __name__ == '__main__':
    main()