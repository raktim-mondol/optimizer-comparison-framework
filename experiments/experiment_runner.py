"""
Experiment runner for comprehensive optimizer comparison.
Handles all combinations of datasets, models, and optimizers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
import traceback
import sys

# Import optimizers
sys.path.append('..')
from optimizers.amcas import AMCAS
from optimizers.ultron import ULTRON

# Import models
from models.cnn_mnist import get_mnist_model
from models.cnn_cifar10 import get_cifar10_model
from models.vit_mnist import VisionTransformerMNISTSmall, VisionTransformerMNISTMedium, VisionTransformerMNISTLarge
from models.vit_cifar10 import VisionTransformerCIFAR10Small, VisionTransformerCIFAR10Medium, VisionTransformerCIFAR10Large

# Import metrics collector
from .metrics_collector import MetricsCollector


class ExperimentRunner:
    """
    Main experiment runner for comparing optimizers across datasets and models.
    """
    
    def __init__(self, output_dir='results', device=None):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
            device: PyTorch device (cpu or cuda)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.metrics_collector = MetricsCollector()
        
        # Available optimizers
        self.optimizer_registry = {
            'AMCAS': AMCAS,
            'ULTRON': ULTRON,
            'Adam': optim.Adam,
            'AdamW': optim.AdamW,
            'SGD': optim.SGD,
            'SGD+Momentum': lambda params, lr, **kwargs: optim.SGD(params, lr=lr, momentum=0.9, **kwargs),
            'RMSprop': optim.RMSprop,
            'Adagrad': optim.Adagrad,
            'Adadelta': optim.Adadelta,
            'NAdam': optim.NAdam,
            'RAdam': optim.RAdam,
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
        
        print(f"Experiment runner initialized with device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_config(self, config_path: str) -> Dict:
        """
        Load experiment configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def save_config(self, config: Dict, experiment_name: str):
        """
        Save experiment configuration.
        
        Args:
            config: Configuration dictionary
            experiment_name: Name of the experiment
        """
        config_dir = self.output_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / f'{experiment_name}_config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_dataset(self, dataset_name: str, batch_size: int = 64, 
                   data_augmentation: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Load dataset by name.
        
        Args:
            dataset_name: 'mnist' or 'cifar10'
            batch_size: Batch size for data loaders
            data_augmentation: Whether to use data augmentation
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        if dataset_name.lower() == 'mnist':
            return self._load_mnist(batch_size, data_augmentation)
        elif dataset_name.lower() == 'cifar10':
            return self._load_cifar10(batch_size, data_augmentation)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: 'mnist', 'cifar10'")
    
    def _load_mnist(self, batch_size: int, data_augmentation: bool) -> Tuple[DataLoader, DataLoader]:
        """Load MNIST dataset."""
        if data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def _load_cifar10(self, batch_size: int, data_augmentation: bool) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR10 dataset."""
        if data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def get_model(self, model_name: str, dataset_name: str) -> nn.Module:
        """
        Get model by name and dataset.
        
        Args:
            model_name: Name of the model
            dataset_name: 'mnist' or 'cifar10'
            
        Returns:
            PyTorch model
        """
        if dataset_name.lower() == 'mnist':
            return get_mnist_model(model_name)
        elif dataset_name.lower() == 'cifar10':
            return get_cifar10_model(model_name)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def get_optimizer(self, optimizer_name: str, model: nn.Module, lr: float, 
                     optimizer_params: Optional[Dict] = None) -> optim.Optimizer:
        """
        Get optimizer by name.
        
        Args:
            optimizer_name: Name of the optimizer
            model: PyTorch model
            lr: Learning rate
            optimizer_params: Additional optimizer parameters
            
        Returns:
            PyTorch optimizer
        """
        if optimizer_name not in self.optimizer_registry:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(self.optimizer_registry.keys())}")
        
        optimizer_class = self.optimizer_registry[optimizer_name]
        
        # Merge default parameters with provided parameters
        params = self.default_optimizer_params.get(optimizer_name, {}).copy()
        if optimizer_params:
            params.update(optimizer_params)
        
        # Special handling for SGD+Momentum
        if optimizer_name == 'SGD+Momentum':
            return optimizer_class(model.parameters(), lr=lr, **params)
        
        return optimizer_class(model.parameters(), lr=lr, **params)
    
    def train_epoch(self, model: nn.Module, device: torch.device, 
                    train_loader: DataLoader, optimizer: optim.Optimizer, 
                    criterion: nn.Module, epoch: int, metrics_collector: MetricsCollector) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        batch_metrics = []
        total_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Show batch progress every 10 batches or on last batch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                batch_acc = 100. * predicted.eq(target).sum().item() / target.size(0)
                batch_loss = loss.item()
                progress_pct = ((batch_idx + 1) / total_batches) * 100
                print(f"  Batch {batch_idx+1}/{total_batches} ({progress_pct:.1f}%): "
                      f"Loss={batch_loss:.4f}, Acc={batch_acc:.1f}%")
            
            # Collect batch metrics
            batch_metrics.append({
                'batch_loss': loss.item(),
                'batch_accuracy': 100. * predicted.eq(target).sum().item() / target.size(0),
            })
            
            # Collect optimizer-specific metrics if available
            if hasattr(optimizer, 'get_gradient_consistency'):
                metrics_collector.add_optimizer_metric('gradient_consistency', 
                                                     optimizer.get_gradient_consistency())
            if hasattr(optimizer, 'get_curvature_stats'):
                metrics_collector.add_optimizer_metric('curvature_stats', 
                                                     optimizer.get_curvature_stats())
            if hasattr(optimizer, 'get_trust_ratio'):
                metrics_collector.add_optimizer_metric('trust_ratio', 
                                                     optimizer.get_trust_ratio())
        
        avg_loss = train_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'batch_metrics': batch_metrics,
        }
    
    def evaluate(self, model: nn.Module, device: torch.device, 
                test_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
        }
    
    def run_experiment(self, config: Dict, experiment_name: str = None) -> Dict:
        """
        Run a single experiment based on configuration.
        
        Args:
            config: Experiment configuration dictionary with keys:
                - dataset: Dataset name ('mnist' or 'cifar10')
                - model: Model name
                - optimizer: Optimizer name
                - epochs: Number of training epochs (default: 10)
                - batch_size: Batch size (default: 64)
                - learning_rate: Learning rate (default: 0.001)
                - data_augmentation: Use data augmentation (default: False)
                - use_scheduler: Use learning rate scheduler (default: False)
                - patience: Early stopping patience in epochs (default: 10)
                - min_delta: Minimum improvement for early stopping (default: 0.001)
                - seed: Random seed (default: 42)
                - optimizer_params: Optimizer-specific parameters
            experiment_name: Name for the experiment (optional)
            
        Returns:
            Dictionary with experiment results
        """
        if experiment_name is None:
            experiment_name = f"{config['dataset']}_{config['model']}_{config['optimizer']}"
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*60}")
        
        # Extract configuration
        dataset_name = config['dataset']
        model_name = config['model']
        optimizer_name = config['optimizer']
        num_epochs = config.get('epochs', 10)
        batch_size = config.get('batch_size', 64)
        learning_rate = config.get('learning_rate', 0.001)
        data_augmentation = config.get('data_augmentation', False)
        use_scheduler = config.get('use_scheduler', False)
        optimizer_params = config.get('optimizer_params', {})
        
        # Set random seed for reproducibility
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        
        # Load dataset
        print(f"Loading {dataset_name} dataset...")
        train_loader, test_loader = self.get_dataset(dataset_name, batch_size, data_augmentation)
        
        # Create model
        print(f"Creating {model_name} model for {dataset_name}...")
        model = self.get_model(model_name, dataset_name)
        model = model.to(self.device)
        
        # Create optimizer
        print(f"Creating {optimizer_name} optimizer with LR={learning_rate}...")
        optimizer = self.get_optimizer(optimizer_name, model, learning_rate, optimizer_params)
        
        # Create loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create learning rate scheduler if requested
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs * len(train_loader)
            )
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector()
        metrics_collector.start_timer('total_training')
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': [],
            'epoch_times': [],
            'learning_rates': [],
            'memory_usage': [],
            'optimizer_metrics': {},
        }
        
        # Track best model and early stopping
        best_accuracy = 0
        best_model_state = None
        epochs_without_improvement = 0
        patience = config.get('patience', 10)  # Default patience of 10 epochs
        min_delta = config.get('min_delta', 0.001)  # Minimum improvement to reset patience
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\n[Epoch {epoch}/{num_epochs}]")
            print("-" * 40)
            
            # Start epoch timer
            epoch_start_time = time.time()
            metrics_collector.start_timer('epoch_training')
            
            # Train for one epoch
            print(f"Training...")
            train_metrics = self.train_epoch(model, self.device, train_loader, 
                                           optimizer, criterion, epoch, metrics_collector)
            
            # Evaluate on test set
            print(f"Evaluating...")
            test_metrics = self.evaluate(model, self.device, test_loader, criterion)
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            metrics_collector.stop_timer('epoch_training')
            
            # Record metrics
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            history['test_loss'].append(test_metrics['test_loss'])
            history['test_accuracy'].append(test_metrics['test_accuracy'])
            history['test_precision'].append(test_metrics['test_precision'])
            history['test_recall'].append(test_metrics['test_recall'])
            history['test_f1'].append(test_metrics['test_f1'])
            history['epoch_times'].append(epoch_time)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Record memory usage
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9  # GB
                history['memory_usage'].append(memory_allocated)
            
            # Check for improvement
            current_accuracy = test_metrics['test_accuracy']
            if current_accuracy > best_accuracy + min_delta:
                best_accuracy = current_accuracy
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
                improvement_symbol = ">>"
            else:
                epochs_without_improvement += 1
                improvement_symbol = ".."
            
            # Print progress summary
            print(f"\n[Epoch {epoch} Summary]:")
            print(f"  [{improvement_symbol}] Best: {best_accuracy:.2f}% | Current: {current_accuracy:.2f}% | No improvement: {epochs_without_improvement}")
            print(f"  [TRAIN] Loss={train_metrics['train_loss']:.4f}, Acc={train_metrics['train_accuracy']:.2f}%")
            print(f"  [TEST]  Loss={test_metrics['test_loss']:.4f}, Acc={test_metrics['test_accuracy']:.2f}%, F1={test_metrics['test_f1']:.4f}")
            print(f"  [TIME]  {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
            if self.device.type == 'cuda':
                print(f"  [GPU]   Memory: {memory_allocated:.2f} GB")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"\n[EARLY STOPPING] Triggered after {epoch} epochs!")
                print(f"   No improvement for {epochs_without_improvement} consecutive epochs (patience={patience})")
                print(f"   [BEST] Accuracy achieved: {best_accuracy:.2f}%")
                break
        
        # Stop total training timer
        metrics_collector.stop_timer('total_training')
        
        # Collect final metrics
        final_metrics = metrics_collector.get_metrics()
        history.update(final_metrics)
        
        # Add model information
        history['model_params'] = sum(p.numel() for p in model.parameters())
        history['model_flops'] = getattr(model, 'get_flops', lambda: 0)()
        history['best_test_accuracy'] = best_accuracy
        
        # Save model checkpoint
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f'{experiment_name}_best.pth'
        torch.save({
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs,
            'best_accuracy': best_accuracy,
            'config': config,
        }, checkpoint_path)
        
        print(f"\n{'='*60}")
        print(f"[COMPLETED] Experiment Finished Successfully!")
        print(f"{'='*60}")
        print(f"[RESULTS] Best Test Accuracy: {best_accuracy:.2f}%")
        print(f"[TIME] Total Training Time: {final_metrics.get('total_training_total', 0):.2f}s")
        print(f"[CHECKPOINT] Model saved to: {checkpoint_path}")
        print(f"[EPOCHS] Trained: {len(history['train_loss'])}/{num_epochs}")
        if len(history['train_loss']) < num_epochs:
            print(f"   (Early stopping saved {num_epochs - len(history['train_loss'])} epochs)")
        print(f"{'='*60}")
        
        return history
    
    def run_batch_experiments(self, configs: List[Dict], parallel: bool = False) -> Dict[str, Dict]:
        """
        Run multiple experiments.
        
        Args:
            configs: List of experiment configurations
            parallel: Whether to run experiments in parallel (not implemented yet)
            
        Returns:
            Dictionary mapping experiment names to results
        """
        results = {}
        
        for i, config in enumerate(configs):
            experiment_name = config.get('name', f'experiment_{i+1}')
            print(f"\n{'='*60}")
            print(f"Running experiment {i+1}/{len(configs)}: {experiment_name}")
            print(f"{'='*60}")
            
            try:
                result = self.run_experiment(config, experiment_name)
                results[experiment_name] = result
                
                # Save intermediate results
                self._save_experiment_result(experiment_name, config, result)
                
            except Exception as e:
                print(f"Error running experiment {experiment_name}: {e}")
                traceback.print_exc()
                results[experiment_name] = {'error': str(e)}
        
        return results
    
    def _save_experiment_result(self, experiment_name: str, config: Dict, result: Dict):
        """Save individual experiment result."""
        result_dir = self.output_dir / 'raw_results'
        result_dir.mkdir(exist_ok=True)
        
        # Save result as JSON
        result_path = result_dir / f'{experiment_name}_result.json'
        with open(result_path, 'w') as f:
            # Convert any numpy arrays to lists for JSON serialization
            serializable_result = self._make_serializable(result)
            json.dump({
                'experiment_name': experiment_name,
                'config': config,
                'result': serializable_result,
            }, f, indent=2, default=str)
        
        print(f"Result saved to: {result_path}")
    
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
    
    def generate_comparison_report(self, results: Dict[str, Dict], output_file: str = 'comparison_report.md'):
        """
        Generate a comparison report from multiple experiment results.
        
        Args:
            results: Dictionary mapping experiment names to results
            output_file: Output markdown file name
        """
        report_path = self.output_dir / output_file
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Optimizer Comparison Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
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
                    params = result.get('model_params', 0)
                    flops = result.get('model_flops', 0)
                    
                    f.write(f"| {exp_name} | {best_acc:.2f} | {final_acc:.2f} | {total_time:.2f} | {params:,} | {flops:,} |\n")
            
            f.write("\n")
            
            # Detailed results for each experiment
            for exp_name, result in results.items():
                if 'error' in result:
                    f.write(f"\n## {exp_name} - ERROR\n\n")
                    f.write(f"Error: {result['error']}\n")
                else:
                    f.write(f"\n## {exp_name}\n\n")
                    
                    # Key metrics
                    f.write("### Key Metrics\n\n")
                    f.write(f"- **Best Test Accuracy**: {result.get('best_test_accuracy', 0):.2f}%\n")
                    f.write(f"- **Final Test Accuracy**: {result.get('test_accuracy', [0])[-1]:.2f}%\n")
                    f.write(f"- **Final Test F1 Score**: {result.get('test_f1', [0])[-1]:.4f}\n")
                    f.write(f"- **Total Training Time**: {result.get('total_training_time', 0):.2f}s\n")
                    f.write(f"- **Average Epoch Time**: {np.mean(result.get('epoch_times', [0])):.2f}s\n")
                    f.write(f"- **Model Parameters**: {result.get('model_params', 0):,}\n")
                    f.write(f"- **Model FLOPs**: {result.get('model_flops', 0):,}\n")
                    
                    # Memory usage if available
                    if 'memory_usage' in result and result['memory_usage']:
                        f.write(f"- **Peak GPU Memory**: {max(result['memory_usage']):.2f} GB\n")
                        f.write(f"- **Average GPU Memory**: {np.mean(result['memory_usage']):.2f} GB\n")
        
        print(f"Comparison report saved to: {report_path}")
        return report_path