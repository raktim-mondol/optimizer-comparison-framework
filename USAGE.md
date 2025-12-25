# Optimizer Comparison Framework - Usage Guide

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run All Experiments for Paper Publication
```bash
python run_full_experiments.py
```

This will run:
1. MNIST CNN comparison with all optimizers
2. MNIST ViT comparison with all optimizers  
3. CIFAR10 CNN comparison with all optimizers
4. CIFAR10 ViT comparison with all optimizers
5. Generate comprehensive Excel reports
6. Create visualization plots
7. Generate markdown reports

### Run Specific Experiments
```bash
# Run all experiments with default settings
python main.py

# Run with specific datasets
python main.py --datasets mnist cifar10

# Run with specific architectures
python main.py --architectures cnn vit

# Run with specific optimizers
python main.py --optimizers AMCAS Adam SGD

# Run with custom parameters
python main.py --epochs 20 --batch-size 128 --learning-rate 0.01

# Run with full benchmark suite (speed + memory)
python main.py --full

# Run on GPU
python main.py --gpu 0
```

## Output Files

After running experiments, you'll get:

### Excel Reports (`results/experiment_results.xlsx`)
- **Summary**: Key metrics for all experiments
- **Detailed_Metrics**: Per-epoch metrics
- **Comparison**: Ranked comparison of optimizers
- **Statistical_Analysis**: Statistical metrics
- **Computational_Metrics**: Speed and memory usage
- **Optimizer_Metrics**: Optimizer-specific metrics

### JSON Data (`results/experiment_results.json`)
Raw experiment data for further analysis.

### Markdown Report (`results/experiment_report.md`)
Comprehensive report with:
- Summary table of all experiments
- Best performers by accuracy, speed, and memory
- Recommendations for optimizer selection

### Visualization Plots (`results/plots/`)
- `loss_curves.png`: Training and test loss curves
- `accuracy_curves.png`: Training and test accuracy curves  
- `f1_curves.png`: Test F1 score curves
- `time_comparison.png`: Training time comparison
- `memory_usage.png`: Memory usage comparison
- `gradient_stats.png`: Gradient statistics

### Summary (`results/summary.txt`)
Quick summary of experiment results.

## Experiment Configurations

### Datasets
- **MNIST**: 60,000 training images, 10,000 test images, 10 classes
- **CIFAR10**: 50,000 training images, 10,000 test images, 10 classes

### Architectures
**MNIST Models:**
- `simple_cnn`: Simple CNN (2 conv layers, 2 FC layers)
- `cnn_v2`: Enhanced CNN with batch normalization
- `cnn_v3`: Deeper CNN with residual connections
- `vit_small`: Small Vision Transformer
- `vit_medium`: Medium Vision Transformer
- `vit_large`: Large Vision Transformer

**CIFAR10 Models:**
- `resnet`: ResNet-18 adapted for CIFAR10
- `cnn`: Custom CNN (3 conv blocks, 3 FC layers)
- `vgg`: VGG-style network
- `vit_small`: Small Vision Transformer
- `vit_medium`: Medium Vision Transformer
- `vit_large`: Large Vision Transformer
- `hybrid`: Hybrid CNN-ViT model

### Optimizers
- **AMCAS**: Adaptive Momentum with Curvature-Aware Scaling
- **Adam**: Adaptive Moment Estimation
- **AdamW**: Adam with decoupled weight decay
- **SGD**: Stochastic Gradient Descent
- **SGD+Momentum**: SGD with momentum (0.9)
- **RMSprop**: Root Mean Square Propagation
- **Adagrad**: Adaptive Gradient Algorithm
- **Adadelta**: Adaptive Learning Rate Method
- **NAdam**: Nesterov-accelerated Adaptive Moment Estimation
- **RAdam**: Rectified Adam

## Metrics Collected

### Classification Metrics
- Accuracy
- Precision (macro-average)
- Recall (macro-average)
- F1-score (macro-average)
- AUC-ROC (per class and macro-average)
- Confusion matrix
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)

### Computational Metrics
- Training time per epoch
- Total training time
- Inference time
- GPU/CPU memory usage
- Model parameters count
- FLOPs estimation
- Gradient statistics (norm, mean, std)

### Optimizer-Specific Metrics (AMCAS)
- Gradient consistency
- Curvature statistics (mean, std, min, max)
- Trust region ratio

## Customization

### Adding New Optimizers
1. Add optimizer implementation to `optimizers/` directory
2. Register optimizer in `experiment_runner.py` optimizer registry
3. Add default parameters in `default_optimizer_params`

### Adding New Models
1. Add model implementation to `models/` directory
2. Register model in appropriate factory function (`get_mnist_model()` or `get_cifar10_model()`)
3. Add model to configuration YAML files

### Custom Experiment Configurations
Create YAML configuration files in `experiments/configs/`:
```yaml
experiment_name: "custom_experiment"
dataset: "mnist"
model: "simple_cnn"
epochs: 10
batch_size: 64
learning_rate: 0.001
data_augmentation: false
use_scheduler: false
seed: 42

optimizers:
  - name: "AMCAS"
    class: "AMCAS"
    params:
      betas: [0.9, 0.999]
      gamma: 0.1
      lambda_consistency: 0.01
      
  - name: "Adam"
    class: "Adam"
    params:
      betas: [0.9, 0.999]
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{optimizer_comparison_framework_2024,
  title = {Optimizer Comparison Framework for MNIST and CIFAR10},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/optimizer-comparison}
}
```

## License

MIT License