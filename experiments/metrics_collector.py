"""
Comprehensive metrics collector for tracking training and evaluation metrics.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import psutil
import gc


class MetricsCollector:
    """
    Collects comprehensive metrics during training and evaluation.
    
    Tracks:
    - Training metrics (loss, accuracy)
    - Evaluation metrics (precision, recall, F1, AUC-ROC)
    - Computational metrics (time, memory, FLOPs)
    - Optimizer-specific metrics (gradient consistency, curvature stats, etc.)
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = defaultdict(list)
        self.timers = {}
        self.memory_tracker = MemoryTracker()
        self.gradient_tracker = GradientTracker()
        
    def start_timer(self, name: str):
        """Start a timer with given name."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop timer and return elapsed time in seconds."""
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' not started")
        
        elapsed = time.time() - self.timers[name]
        self.add_metric(f"{name}_time", elapsed)
        return elapsed
    
    def add_metric(self, name: str, value: Any):
        """Add a metric value."""
        self.metrics[name].append(value)
    
    def add_batch_metrics(self, batch_idx: int, loss: float, accuracy: float, 
                         grad_norm: Optional[float] = None):
        """Add batch-level metrics."""
        self.add_metric('batch_loss', loss)
        self.add_metric('batch_accuracy', accuracy)
        self.add_metric('batch_idx', batch_idx)
        
        if grad_norm is not None:
            self.add_metric('batch_gradient_norm', grad_norm)
        
        # Track memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
            self.add_metric('batch_gpu_memory_allocated', memory_allocated)
            self.add_metric('batch_gpu_memory_reserved', memory_reserved)
        
        # Track CPU memory
        cpu_memory = psutil.Process().memory_info().rss / 1e9  # GB
        self.add_metric('batch_cpu_memory', cpu_memory)
    
    def add_epoch_metrics(self, epoch: int, train_loss: float, train_accuracy: float,
                          test_loss: float, test_accuracy: float, test_precision: float,
                          test_recall: float, test_f1: float, epoch_time: float):
        """Add epoch-level metrics."""
        self.add_metric('epoch', epoch)
        self.add_metric('train_loss', train_loss)
        self.add_metric('train_accuracy', train_accuracy)
        self.add_metric('test_loss', test_loss)
        self.add_metric('test_accuracy', test_accuracy)
        self.add_metric('test_precision', test_precision)
        self.add_metric('test_recall', test_recall)
        self.add_metric('test_f1', test_f1)
        self.add_metric('epoch_time', epoch_time)
        
        # Calculate generalization gap
        generalization_gap = train_accuracy - test_accuracy
        self.add_metric('generalization_gap', generalization_gap)
        
        # Track memory at epoch end
        self.memory_tracker.track()
        memory_stats = self.memory_tracker.get_stats()
        for key, value in memory_stats.items():
            self.add_metric(f'epoch_{key}', value)
    
    def add_optimizer_metric(self, name: str, value: Any):
        """Add optimizer-specific metric."""
        self.add_metric(f'optimizer_{name}', value)
    
    def add_gradient_stats(self, model: torch.nn.Module):
        """Add gradient statistics for all parameters."""
        grad_stats = self.gradient_tracker.compute_gradient_stats(model)
        for key, value in grad_stats.items():
            self.add_metric(f'gradient_{key}', value)
    
    def add_model_stats(self, model: torch.nn.Module):
        """Add model statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.add_metric('total_parameters', total_params)
        self.add_metric('trainable_parameters', trainable_params)
        
        # Estimate FLOPs if available
        if hasattr(model, 'get_flops'):
            try:
                flops = model.get_flops()
                self.add_metric('model_flops', flops)
            except:
                pass
    
    def compute_auc_roc(self, all_preds: List[np.ndarray], all_targets: List[np.ndarray], 
                       num_classes: int = 10) -> Dict[str, float]:
        """
        Compute AUC-ROC scores.
        
        Args:
            all_preds: List of predicted probabilities (shape: [n_samples, n_classes])
            all_targets: List of true labels
            num_classes: Number of classes
            
        Returns:
            Dictionary with AUC-ROC scores
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
        
        if not all_preds or not all_targets:
            return {}
        
        # Convert to numpy arrays
        preds_array = np.vstack(all_preds) if isinstance(all_preds[0], np.ndarray) else np.array(all_preds)
        targets_array = np.hstack(all_targets) if isinstance(all_targets[0], np.ndarray) else np.array(all_targets)
        
        # Binarize labels for multi-class ROC
        targets_binary = label_binarize(targets_array, classes=range(num_classes))
        
        auc_scores = {}
        try:
            # Compute AUC for each class
            for i in range(num_classes):
                auc = roc_auc_score(targets_binary[:, i], preds_array[:, i])
                auc_scores[f'auc_class_{i}'] = auc
            
            # Compute macro-average AUC
            auc_scores['auc_macro'] = np.mean([auc_scores[f'auc_class_{i}'] for i in range(num_classes)])
            
            # Compute micro-average AUC (if predictions are probabilities)
            auc_scores['auc_micro'] = roc_auc_score(targets_binary.ravel(), preds_array.ravel())
            
        except ValueError as e:
            # AUC calculation may fail if only one class present
            print(f"Warning: AUC calculation failed: {e}")
        
        return auc_scores
    
    def compute_confusion_matrix(self, all_preds: List, all_targets: List, 
                                num_classes: int = 10) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            Confusion matrix of shape (num_classes, num_classes)
        """
        from sklearn.metrics import confusion_matrix
        
        preds_array = np.hstack(all_preds) if isinstance(all_preds[0], np.ndarray) else np.array(all_preds)
        targets_array = np.hstack(all_targets) if isinstance(all_targets[0], np.ndarray) else np.array(all_targets)
        
        return confusion_matrix(targets_array, preds_array, labels=range(num_classes))
    
    def compute_calibration_error(self, all_preds: List[np.ndarray], all_targets: List, 
                                 num_bins: int = 10) -> Dict[str, float]:
        """
        Compute calibration error (ECE and MCE).
        
        Returns:
            Dictionary with Expected Calibration Error (ECE) and 
            Maximum Calibration Error (MCE)
        """
        if not all_preds or not all_targets:
            return {'ece': 0.0, 'mce': 0.0}
        
        # Convert to numpy arrays
        preds_array = np.vstack(all_preds) if isinstance(all_preds[0], np.ndarray) else np.array(all_preds)
        targets_array = np.hstack(all_targets) if isinstance(all_targets[0], np.ndarray) else np.array(all_targets)
        
        # Get predicted probabilities for true classes
        n_samples = len(targets_array)
        pred_probs = preds_array[np.arange(n_samples), targets_array]
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = np.logical_and(pred_probs >= bin_lower, pred_probs < bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = np.mean(targets_array[in_bin] == np.argmax(preds_array[in_bin], axis=1))
                
                # Average confidence in this bin
                avg_confidence_in_bin = np.mean(pred_probs[in_bin])
                
                # Calibration error for this bin
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                
                # Update ECE and MCE
                ece += prop_in_bin * bin_error
                mce = max(mce, bin_error)
        
        return {'ece': ece, 'mce': mce}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics with summary statistics.
        
        Returns:
            Dictionary with all metrics and their summary statistics
        """
        result = {}
        
        # Add raw metrics
        for key, values in self.metrics.items():
            result[key] = values
            
            # Add summary statistics for numeric lists
            if values and isinstance(values[0], (int, float, np.number)):
                result[f'{key}_mean'] = np.mean(values)
                result[f'{key}_std'] = np.std(values)
                result[f'{key}_min'] = np.min(values)
                result[f'{key}_max'] = np.max(values)
                result[f'{key}_median'] = np.median(values)
        
        # Add timer summaries
        for timer_name, start_time in self.timers.items():
            if f"{timer_name}_time" in result:
                result[f"{timer_name}_total"] = sum(result[f"{timer_name}_time"])
        
        # Add memory statistics
        memory_stats = self.memory_tracker.get_summary_stats()
        result.update(memory_stats)
        
        # Add gradient statistics
        gradient_stats = self.gradient_tracker.get_summary_stats()
        result.update(gradient_stats)
        
        return result
    
    def reset(self):
        """Reset all collected metrics."""
        self.metrics.clear()
        self.timers.clear()
        self.memory_tracker.reset()
        self.gradient_tracker.reset()
    
    def save_to_file(self, filepath: str):
        """
        Save metrics to file.
        
        Args:
            filepath: Path to save metrics JSON file
        """
        import json
        import numpy as np
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and value and isinstance(value[0], (np.integer, np.floating)):
                serializable_metrics[key] = [float(v) if isinstance(v, np.generic) else v for v in value]
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
    
    def load_from_file(self, filepath: str):
        """
        Load metrics from file.
        
        Args:
            filepath: Path to load metrics JSON file from
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_metrics = json.load(f)
        
        self.metrics.update(loaded_metrics)


class MemoryTracker:
    """Tracks memory usage during training."""
    
    def __init__(self):
        self.memory_readings = []
        self.gpu_memory_readings = []
    
    def track(self):
        """Record current memory usage."""
        # CPU memory
        cpu_memory = psutil.Process().memory_info().rss / 1e9  # GB
        self.memory_readings.append(cpu_memory)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9  # GB
            self.gpu_memory_readings.append(gpu_memory)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        if self.memory_readings:
            stats['cpu_memory_gb'] = self.memory_readings[-1]
            stats['cpu_memory_mean_gb'] = np.mean(self.memory_readings)
            stats['cpu_memory_max_gb'] = np.max(self.memory_readings)
        
        if self.gpu_memory_readings:
            stats['gpu_memory_gb'] = self.gpu_memory_readings[-1]
            stats['gpu_memory_mean_gb'] = np.mean(self.gpu_memory_readings)
            stats['gpu_memory_max_gb'] = np.max(self.gpu_memory_readings)
        
        return stats
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary memory statistics."""
        stats = {}
        
        if self.memory_readings:
            stats['cpu_memory_peak_gb'] = np.max(self.memory_readings)
            stats['cpu_memory_avg_gb'] = np.mean(self.memory_readings)
            stats['cpu_memory_std_gb'] = np.std(self.memory_readings)
        
        if self.gpu_memory_readings:
            stats['gpu_memory_peak_gb'] = np.max(self.gpu_memory_readings)
            stats['gpu_memory_avg_gb'] = np.mean(self.gpu_memory_readings)
            stats['gpu_memory_std_gb'] = np.std(self.gpu_memory_readings)
        
        return stats
    
    def reset(self):
        """Reset memory tracker."""
        self.memory_readings.clear()
        self.gpu_memory_readings.clear()


class GradientTracker:
    """Tracks gradient statistics during training."""
    
    def __init__(self):
        self.gradient_norms = []
        self.gradient_means = []
        self.gradient_stds = []
    
    def compute_gradient_stats(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Compute gradient statistics for all parameters.
        
        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        all_gradients = []
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                all_gradients.extend(param.grad.data.cpu().numpy().flatten())
        
        total_norm = total_norm ** 0.5
        
        stats = {
            'gradient_norm': total_norm,
        }
        
        if all_gradients:
            stats['gradient_mean'] = np.mean(all_gradients)
            stats['gradient_std'] = np.std(all_gradients)
            stats['gradient_min'] = np.min(all_gradients)
            stats['gradient_max'] = np.max(all_gradients)
        
        # Store for summary statistics
        self.gradient_norms.append(total_norm)
        if all_gradients:
            self.gradient_means.append(stats['gradient_mean'])
            self.gradient_stds.append(stats['gradient_std'])
        
        return stats
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary gradient statistics."""
        stats = {}
        
        if self.gradient_norms:
            stats['gradient_norm_mean'] = np.mean(self.gradient_norms)
            stats['gradient_norm_std'] = np.std(self.gradient_norms)
            stats['gradient_norm_max'] = np.max(self.gradient_norms)
            stats['gradient_norm_min'] = np.min(self.gradient_norms)
        
        if self.gradient_means:
            stats['gradient_mean_mean'] = np.mean(self.gradient_means)
            stats['gradient_mean_std'] = np.std(self.gradient_means)
        
        if self.gradient_stds:
            stats['gradient_std_mean'] = np.mean(self.gradient_stds)
            stats['gradient_std_std'] = np.std(self.gradient_stds)
        
        return stats
    
    def reset(self):
        """Reset gradient tracker."""
        self.gradient_norms.clear()
        self.gradient_means.clear()
        self.gradient_stds.clear()