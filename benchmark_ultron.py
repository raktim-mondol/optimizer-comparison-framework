#!/usr/bin/env python3
"""
Comprehensive benchmark for ULTRON optimizer.
Compares ULTRON against other optimizers on various test functions and tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt

# Import optimizers
from optimizers.ultron import ULTRON
from optimizers.amcas import AMCAS


class ULTRONBenchmark:
    """Benchmark ULTRON against other optimizers."""
    
    def __init__(self, output_dir='ultron_benchmark_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Available optimizers
        self.optimizer_registry = {
            'ULTRON': ULTRON,
            'AMCAS': AMCAS,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'SGD+Momentum': lambda params, lr, **kwargs: torch.optim.SGD(params, lr=lr, momentum=0.9, **kwargs),
            'RMSprop': torch.optim.RMSprop,
            'Adagrad': torch.optim.Adagrad,
        }
        
        # Default parameters for each optimizer
        self.default_params = {
            'ULTRON': {'lr': 0.001, 'clip_threshold': 0.1, 'normalize_gradients': True},
            'AMCAS': {'lr': 0.001, 'betas': (0.9, 0.999)},
            'Adam': {'lr': 0.001, 'betas': (0.9, 0.999)},
            'AdamW': {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
            'SGD': {'lr': 0.01},
            'SGD+Momentum': {'lr': 0.01, 'momentum': 0.9},
            'RMSprop': {'lr': 0.001},
            'Adagrad': {'lr': 0.01},
        }
    
    def rosenbrock_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2"""
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rastrigin_function(self, x: torch.Tensor) -> torch.Tensor:
        """Rastrigin function: f(x) = A*n + Σ(x_i^2 - A*cos(2πx_i))"""
        A = 10
        n = len(x)
        return A * n + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x))
    
    def quadratic_bowl(self, x: torch.Tensor, center: float = 0.0) -> torch.Tensor:
        """Quadratic bowl: f(x) = ||x - center||^2"""
        return torch.sum((x - center)**2)
    
    def test_function_optimization(self, function_name: str, 
                                  initial_point: torch.Tensor,
                                  num_iterations: int = 1000,
                                  learning_rate: float = 0.001) -> Dict[str, Dict]:
        """
        Test optimizers on a specific function.
        
        Args:
            function_name: Name of the test function
            initial_point: Initial point for optimization
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizers
            
        Returns:
            Dictionary mapping optimizer names to results
        """
        print(f"\nTesting on {function_name} function...")
        
        # Define function based on name
        if function_name == 'rosenbrock':
            def func(x, y):
                return self.rosenbrock_function(x, y)
            dim = 2
        elif function_name == 'rastrigin':
            def func(x):
                return self.rastrigin_function(x)
            dim = len(initial_point)
        elif function_name == 'quadratic':
            def func(x):
                return self.quadratic_bowl(x)
            dim = len(initial_point)
        else:
            raise ValueError(f"Unknown function: {function_name}")
        
        results = {}
        for optimizer_name in self.optimizer_registry.keys():
            print(f"  Testing {optimizer_name}...")
            
            # Create parameters
            if function_name == 'rosenbrock':
                x = torch.nn.Parameter(initial_point[0].clone())
                y = torch.nn.Parameter(initial_point[1].clone())
                params = [x, y]
            else:
                x = torch.nn.Parameter(initial_point.clone())
                params = [x]
            
            # Create optimizer
            optimizer_class = self.optimizer_registry[optimizer_name]
            optimizer_params = self.default_params.get(optimizer_name, {}).copy()
            optimizer_params['lr'] = learning_rate
            
            if optimizer_name == 'SGD+Momentum':
                optimizer = optimizer_class(params, **optimizer_params)
            else:
                optimizer = optimizer_class(params, **optimizer_params)
            
            # Track optimization
            losses = []
            times = []
            start_time = time.time()
            
            for i in range(num_iterations):
                optimizer.zero_grad()
                
                if function_name == 'rosenbrock':
                    loss = func(x, y)
                else:
                    loss = func(x)
                
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                times.append(time.time() - start_time)
                
                # Early stopping for quadratic
                if function_name == 'quadratic' and loss.item() < 1e-10:
                    break
            
            # Store results
            if function_name == 'rosenbrock':
                final_point = (x.item(), y.item())
            else:
                final_point = x.detach().numpy().tolist()
            
            results[optimizer_name] = {
                'losses': losses,
                'times': times,
                'final_loss': losses[-1],
                'min_loss': min(losses),
                'final_point': final_point,
                'num_iterations': len(losses),
                'converged': losses[-1] < 1e-3,
            }
        
        return results
    
    def test_neural_network(self, model: nn.Module,
                           train_loader: Any,
                           test_loader: Any,
                           num_epochs: int = 5,
                           learning_rate: float = 0.001) -> Dict[str, Dict]:
        """
        Test optimizers on a neural network task.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizers
            
        Returns:
            Dictionary mapping optimizer names to results
        """
        print("\nTesting on neural network task...")
        
        results = {}
        for optimizer_name in self.optimizer_registry.keys():
            print(f"  Testing {optimizer_name}...")
            
            # Create fresh model copy
            model_copy = type(model)(*model.init_args) if hasattr(model, 'init_args') else type(model)()
            
            # Create optimizer
            optimizer_class = self.optimizer_registry[optimizer_name]
            optimizer_params = self.default_params.get(optimizer_name, {}).copy()
            optimizer_params['lr'] = learning_rate
            
            if optimizer_name == 'SGD+Momentum':
                optimizer = optimizer_class(model_copy.parameters(), **optimizer_params)
            else:
                optimizer = optimizer_class(model_copy.parameters(), **optimizer_params)
            
            # Track training
            train_losses = []
            test_accuracies = []
            times = []
            
            start_time = time.time()
            for epoch in range(num_epochs):
                # Training
                model_copy.train()
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model_copy(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                train_losses.append(epoch_loss / len(train_loader))
                
                # Testing
                model_copy.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        output = model_copy(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                
                test_accuracies.append(100. * correct / total)
                times.append(time.time() - start_time)
                
                print(f"    Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, "
                      f"Accuracy = {test_accuracies[-1]:.2f}%")
            
            # Store results
            results[optimizer_name] = {
                'train_losses': train_losses,
                'test_accuracies': test_accuracies,
                'times': times,
                'final_train_loss': train_losses[-1],
                'final_test_accuracy': test_accuracies[-1],
                'best_test_accuracy': max(test_accuracies),
                'total_time': times[-1],
            }
        
        return results
    
    def test_memory_efficiency(self, model_size: Tuple[int, int] = (1000, 2000)) -> Dict[str, float]:
        """
        Test memory efficiency of optimizers.
        
        Args:
            model_size: Size of test model (input_dim, hidden_dim)
            
        Returns:
            Dictionary mapping optimizer names to memory usage in MB
        """
        print("\nTesting memory efficiency...")
        
        # Create a simple model
        input_dim, hidden_dim = model_size
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )
        
        # Create dummy data
        dummy_input = torch.randn(32, input_dim)
        dummy_target = torch.randint(0, 10, (32,))
        
        memory_usage = {}
        for optimizer_name in self.optimizer_registry.keys():
            print(f"  Testing {optimizer_name}...")
            
            # Create fresh model
            model_copy = type(model)()
            
            # Create optimizer
            optimizer_class = self.optimizer_registry[optimizer_name]
            optimizer_params = self.default_params.get(optimizer_name, {}).copy()
            
            if optimizer_name == 'SGD+Momentum':
                optimizer = optimizer_class(model_copy.parameters(), **optimizer_params)
            else:
                optimizer = optimizer_class(model_copy.parameters(), **optimizer_params)
            
            # Do one training step to initialize optimizer state
            optimizer.zero_grad()
            output = model_copy(dummy_input)
            loss = F.cross_entropy(output, dummy_target)
            loss.backward()
            optimizer.step()
            
            # Calculate state size
            state_size = 0
            for param in model_copy.parameters():
                if param.grad is not None:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state_size += value.numel() * value.element_size()
            
            memory_usage[optimizer_name] = state_size / (1024 * 1024)  # Convert to MB
            print(f"    Memory usage: {memory_usage[optimizer_name]:.2f} MB")
        
        return memory_usage
    
    def test_computational_speed(self, num_iterations: int = 1000) -> Dict[str, float]:
        """
        Test computational speed of optimizers.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary mapping optimizer names to time per iteration in milliseconds
        """
        print("\nTesting computational speed...")
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )
        
        # Create dummy data
        dummy_input = torch.randn(32, 100)
        dummy_target = torch.randint(0, 10, (32,))
        
        speeds = {}
        for optimizer_name in self.optimizer_registry.keys():
            print(f"  Testing {optimizer_name}...")
            
            # Create fresh model
            model_copy = type(model)()
            
            # Create optimizer
            optimizer_class = self.optimizer_registry[optimizer_name]
            optimizer_params = self.default_params.get(optimizer_name, {}).copy()
            
            if optimizer_name == 'SGD+Momentum':
                optimizer = optimizer_class(model_copy.parameters(), **optimizer_params)
            else:
                optimizer = optimizer_class(model_copy.parameters(), **optimizer_params)
            
            # Warmup
            for _ in range(10):
                optimizer.zero_grad()
                output = model_copy(dummy_input)
                loss = F.cross_entropy(output, dummy_target)
                loss.backward()
                optimizer.step()
            
            # Benchmark
            start_time = time.time()
            for i in range(num_iterations):
                optimizer.zero_grad()
                output = model_copy(dummy_input)
                loss = F.cross_entropy(output, dummy_target)
                loss.backward()
                optimizer.step()
            
            total_time = time.time() - start_time
            time_per_iter = total_time / num_iterations * 1000  # Convert to ms
            
            speeds[optimizer_name] = time_per_iter
            print(f"    Time per iteration: {time_per_iter:.3f} ms")
        
        return speeds
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite."""
        print("=" * 80)
        print("ULTRON Comprehensive Benchmark Suite")
        print("=" * 80)
        
        all_results = {}
        
        # 1. Test on optimization functions
        print("\n1. Testing on optimization functions...")
        
        # Rosenbrock function
        rosenbrock_results = self.test_function_optimization(
            'rosenbrock',
            torch.tensor([-1.5, 2.0]),
            num_iterations=1000,
            learning_rate=0.001
        )
        all_results['rosenbrock'] = rosenbrock_results
        
        # Rastrigin function (2D)
        rastrigin_results = self.test_function_optimization(
            'rastrigin',
            torch.tensor([5.0, 5.0]),
            num_iterations=1000,
            learning_rate=0.01
        )
        all_results['rastrigin'] = rastrigin_results
        
        # Quadratic bowl (10D)
        quadratic_results = self.test_function_optimization(
            'quadratic',
            torch.randn(10) * 5,
            num_iterations=500,
            learning_rate=0.1
        )
        all_results['quadratic'] = quadratic_results
        
        # 2. Test memory efficiency
        print("\n2. Testing memory efficiency...")
        memory_results = self.test_memory_efficiency()
        all_results['memory'] = memory_results
        
        # 3. Test computational speed
        print("\n3. Testing computational speed...")
        speed_results = self.test_computational_speed(num_iterations=500)
        all_results['speed'] = speed_results
        
        # 4. Test on MNIST (if data is available)
        try:
            print("\n4. Testing on MNIST dataset...")
            import torchvision
            import torchvision.transforms as transforms
            
            # Load MNIST data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1000, shuffle=False)
            
            # Simple CNN for MNIST
            class MNISTCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.init_args = ()
                    self.conv1 = nn.Conv2d(1, 32, 3, 1)
                    self.conv2 = nn.Conv2d(32, 64, 3, 1)
                    self.dropout1 = nn.Dropout2d(0.25)
                    self.dropout2 = nn.Dropout2d(0.5)
                    self.fc1 = nn.Linear(9216, 128)
                    self.fc2 = nn.Linear(128, 10)
                
                def forward(self, x):
                    x = self.conv1(x)
                    x = F.relu(x)
                    x = self.conv2(x)
                    x = F.relu(x)
                    x = F.max_pool2d(x, 2)
                    x = self.dropout1(x)
                    x = torch.flatten(x, 1)
                    x = self.fc1(x)
                    x = F.relu(x)
                    x = self.dropout2(x)
                    x = self.fc2(x)
                    return F.log_softmax(x, dim=1)
            
            mnist_results = self.test_neural_network(
                MNISTCNN(),
                train_loader,
                test_loader,
                num_epochs=3,
                learning_rate=0.001
            )
            all_results['mnist'] = mnist_results
            
        except Exception as e:
            print(f"  Skipping MNIST test: {e}")
        
        # Save results
        self._save_results(all_results)
        
        # Generate report
        self._generate_report(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON file."""
        # Convert to serializable format
        serializable_results = {}
        for test_name, test_results in results.items():
            if test_name in ['memory', 'speed']:
                serializable_results[test_name] = test_results
            else:
                serializable_results[test_name] = {}
                for optimizer_name, optimizer_results in test_results.items():
                    serializable_results[test_name][optimizer_name] = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else 
                            [float(x) if isinstance(x, (torch.Tensor, np.generic)) else x 
                             for x in v] if isinstance(v, list) else
                            float(v) if isinstance(v, (torch.Tensor, np.generic)) else v)
                        for k, v in optimizer_results.items()
                    }
        
        result_file = self.output_dir / 'benchmark_results.json'
        with open(result_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {result_file}")
    
    def _generate_report(self, results: Dict[str, Any]):
        """Generate benchmark report."""
        report_file = self.output_dir / 'benchmark_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# ULTRON Optimizer Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Performance summary
            f.write("## Performance Summary\n\n")
            
            # Function optimization results
            if 'rosenbrock' in results:
                f.write("### Function Optimization\n\n")
                f.write("| Optimizer | Rosenbrock Final Loss | Rastrigin Final Loss | Quadratic Final Loss |\n")
                f.write("|-----------|----------------------|---------------------|---------------------|\n")
                
                for optimizer_name in self.optimizer_registry.keys():
                    rosenbrock_loss = results['rosenbrock'].get(optimizer_name, {}).get('final_loss', 'N/A')
                    rastrigin_loss = results['rastrigin'].get(optimizer_name, {}).get('final_loss', 'N/A')
                    quadratic_loss = results['quadratic'].get(optimizer_name, {}).get('final_loss', 'N/A')
                    
                    if isinstance(rosenbrock_loss, (int, float)):
                        rosenbrock_str = f"{rosenbrock_loss:.6e}"
                    else:
                        rosenbrock_str = str(rosenbrock_loss)
                    
                    if isinstance(rastrigin_loss, (int, float)):
                        rastrigin_str = f"{rastrigin_loss:.6e}"
                    else:
                        rastrigin_str = str(rastrigin_loss)
                    
                    if isinstance(quadratic_loss, (int, float)):
                        quadratic_str = f"{quadratic_loss:.6e}"
                    else:
                        quadratic_str = str(quadratic_loss)
                    
                    f.write(f"| {optimizer_name} | {rosenbrock_str} | {rastrigin_str} | {quadratic_str} |\n")
                
                f.write("\n")
            
            # Memory efficiency
            if 'memory' in results:
                f.write("### Memory Efficiency (Lower is Better)\n\n")
                f.write("| Optimizer | Memory Usage (MB) |\n")
                f.write("|-----------|------------------|\n")
                
                for optimizer_name, memory_usage in results['memory'].items():
                    f.write(f"| {optimizer_name} | {memory_usage:.2f} |\n")
                
                f.write("\n")
            
            # Computational speed
            if 'speed' in results:
                f.write("### Computational Speed (Lower is Better)\n\n")
                f.write("| Optimizer | Time per Iteration (ms) |\n")
                f.write("|-----------|------------------------|\n")
                
                for optimizer_name, speed in results['speed'].items():
                    f.write(f"| {optimizer_name} | {speed:.3f} |\n")
                
                f.write("\n")
            
            # MNIST results
            if 'mnist' in results:
                f.write("### MNIST Classification\n\n")
                f.write("| Optimizer | Final Accuracy (%) | Best Accuracy (%) | Training Time (s) |\n")
                f.write("|-----------|-------------------|------------------|------------------|\n")
                
                for optimizer_name, mnist_results in results['mnist'].items():
                    final_acc = mnist_results.get('final_test_accuracy', 'N/A')
                    best_acc = mnist_results.get('best_test_accuracy', 'N/A')
                    total_time = mnist_results.get('total_time', 'N/A')
                    
                    f.write(f"| {optimizer_name} | {final_acc:.2f} | {best_acc:.2f} | {total_time:.2f} |\n")
                
                f.write("\n")
            
            # Overall recommendations
            f.write("## Recommendations\n\n")
            
            # Find best optimizer for each category
            categories = {}
            if 'rosenbrock' in results:
                best_rosenbrock = min(results['rosenbrock'].items(), 
                                    key=lambda x: x[1].get('final_loss', float('inf')))[0]
                categories['Rosenbrock'] = best_rosenbrock
            
            if 'memory' in results:
                best_memory = min(results['memory'].items(), key=lambda x: x[1])[0]
                categories['Memory Efficiency'] = best_memory
            
            if 'speed' in results:
                best_speed = min(results['speed'].items(), key=lambda x: x[1])[0]
                categories['Computational Speed'] = best_speed
            
            if 'mnist' in results:
                best_mnist = max(results['mnist'].items(), 
                               key=lambda x: x[1].get('best_test_accuracy', 0))[0]
                categories['MNIST Accuracy'] = best_mnist
            
            f.write("### Best Optimizer by Category\n\n")
            for category, best_optimizer in categories.items():
                f.write(f"- **{category}**: {best_optimizer}\n")
            
            f.write("\n### When to Use ULTRON\n\n")
            f.write("1. **When memory is limited**: ULTRON has minimal state requirements\n")
            f.write("2. **For fast iterations**: ULTRON's sign-based updates are computationally cheap\n")
            f.write("3. **For stable training**: Built-in gradient clipping prevents exploding gradients\n")
            f.write("4. **For large models**: Lower memory overhead scales better with model size\n")
            f.write("5. **When Adam/AMCAS are too heavy**: ULTRON provides a lightweight alternative\n")
            
            f.write("\n### When to Use Other Optimizers\n\n")
            f.write("1. **For maximum accuracy**: Adam/AMCAS may perform better on complex tasks\n")
            f.write("2. **For convex problems**: SGD with momentum often converges faster\n")
            f.write("3. **For sparse gradients**: Adagrad/RMSprop can be more effective\n")
            f.write("4. **When memory is not a concern**: Use Adam/AMCAS for best overall performance\n")
        
        print(f"Report generated: {report_file}")
        
        # Generate plots
        self._generate_plots(results)
    
    def _generate_plots(self, results: Dict[str, Any]):
        """Generate benchmark plots."""
        try:
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Colors for different optimizers
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimizer_registry)))
            optimizer_colors = {name: color for name, color in zip(self.optimizer_registry.keys(), colors)}
            
            # Plot 1: Function optimization convergence
            if 'rosenbrock' in results:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                for idx, (func_name, func_results) in enumerate([
                    ('rosenbrock', results['rosenbrock']),
                    ('rastrigin', results['rastrigin']),
                    ('quadratic', results['quadratic'])
                ]):
                    ax = axes[idx]
                    for optimizer_name, optimizer_results in func_results.items():
                        losses = optimizer_results.get('losses', [])
                        if losses:
                            ax.plot(losses, label=optimizer_name, 
                                   color=optimizer_colors.get(optimizer_name, 'black'),
                                   linewidth=2, alpha=0.8)
                    
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Loss')
                    ax.set_yscale('log')
                    ax.set_title(f'{func_name.title()} Function')
                    ax.grid(True, alpha=0.3)
                    
                    if idx == 0:
                        ax.legend(fontsize=9)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'function_convergence.png', dpi=150)
                plt.close()
            
            # Plot 2: Memory and speed comparison
            if 'memory' in results and 'speed' in results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Memory plot
                memory_data = results['memory']
                names = list(memory_data.keys())
                memory_values = list(memory_data.values())
                
                x = np.arange(len(names))
                ax1.bar(x, memory_values, color='skyblue')
                ax1.set_xlabel('Optimizer')
                ax1.set_ylabel('Memory Usage (MB)')
                ax1.set_title('Memory Efficiency')
                ax1.set_xticks(x)
                ax1.set_xticklabels(names, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                
                # Speed plot
                speed_data = results['speed']
                speed_values = [speed_data.get(name, 0) for name in names]
                
                ax2.bar(x, speed_values, color='lightcoral')
                ax2.set_xlabel('Optimizer')
                ax2.set_ylabel('Time per Iteration (ms)')
                ax2.set_title('Computational Speed')
                ax2.set_xticks(x)
                ax2.set_xticklabels(names, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'memory_speed_comparison.png', dpi=150)
                plt.close()
            
            # Plot 3: MNIST training curves
            if 'mnist' in results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                for optimizer_name, optimizer_results in results['mnist'].items():
                    train_losses = optimizer_results.get('train_losses', [])
                    test_accuracies = optimizer_results.get('test_accuracies', [])
                    
                    if train_losses:
                        epochs = range(1, len(train_losses) + 1)
                        ax1.plot(epochs, train_losses, label=optimizer_name,
                               color=optimizer_colors.get(optimizer_name, 'black'),
                               linewidth=2, alpha=0.8)
                    
                    if test_accuracies:
                        epochs = range(1, len(test_accuracies) + 1)
                        ax2.plot(epochs, test_accuracies, label=optimizer_name,
                               color=optimizer_colors.get(optimizer_name, 'black'),
                               linewidth=2, alpha=0.8)
                
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Training Loss')
                ax1.set_title('MNIST Training Loss')
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)
                
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Test Accuracy (%)')
                ax2.set_title('MNIST Test Accuracy')
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'mnist_training.png', dpi=150)
                plt.close()
            
            print(f"Plots saved to: {plots_dir}/")
            
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")


def main():
    """Run the benchmark."""
    benchmark = ULTRONBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n" + "=" * 80)
    print("Benchmark completed successfully!")
    print("=" * 80)
    
    # Print quick summary
    print("\nQuick Summary:")
    print("-" * 40)
    
    if 'rosenbrock' in results:
        best_rosenbrock = min(results['rosenbrock'].items(), 
                            key=lambda x: x[1].get('final_loss', float('inf')))[0]
        print(f"Best for Rosenbrock: {best_rosenbrock}")
    
    if 'memory' in results:
        best_memory = min(results['memory'].items(), key=lambda x: x[1])[0]
        print(f"Most memory efficient: {best_memory}")
    
    if 'speed' in results:
        best_speed = min(results['speed'].items(), key=lambda x: x[1])[0]
        print(f"Fastest: {best_speed}")
    
    if 'mnist' in results:
        best_mnist = max(results['mnist'].items(), 
                       key=lambda x: x[1].get('best_test_accuracy', 0))[0]
        print(f"Best MNIST accuracy: {best_mnist}")
    
    print("\nSee ultron_benchmark_results/ for detailed results and plots.")


if __name__ == '__main__':
    main()
