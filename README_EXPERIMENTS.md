# Optimizer Comparison Framework for MNIST and CIFAR10

A comprehensive experimental framework for comparing the AMCAS optimizer against popular existing optimizers (SGD, Adam, RMSprop, etc.) on MNIST and CIFAR10 datasets using both CNN and Vision Transformer (ViT) architectures.

## Features

- **Comprehensive Optimizer Comparison**: Compare AMCAS with 9 other optimizers (Adam, AdamW, SGD, SGD+Momentum, RMSprop, Adagrad, Adadelta, NAdam, RAdam)
- **Multiple Architectures**: CNN and Vision Transformer (ViT) models for both MNIST and CIFAR10
- **Automated Experiment Pipeline**: Run all experiments with a single command
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix
- **Benchmarking**: Speed and memory profiling for each optimizer
- **Excel Reporting**: All results exported to Excel with multiple sheets
- **Visualization**: Automatic generation of comparison plots
- **Reproducibility**: Seed control and detailed logging

## Project Structure

```
DL_Theory/
├── main.py                      # Main entry point for running all experiments
├── requirements.txt             # Python dependencies
├── README_EXPERIMENTS.md        # This file
│
├── optimizers/                  # Optimizer implementations
│   ├── amcas.py                 # AMCAS optimizer (user's proposed optimizer)
│   ├── base.py                  # Base optimizer class
│   └── utils.py                 # Utility functions
│
├── models/                      # Model architectures
│   ├── cnn_mnist.py            # CNN models for MNIST
│   ├── cnn_cifar10.py          # CNN models for CIFAR10
│   ├── vit_mnist.py            # ViT models for MNIST
│   └── vit_cifar10.py          # ViT models for CIFAR10
│
├── experiments/                 # Experiment management
│   ├── experiment_runner.py    # Runs experiments and collects metrics
│   ├── metrics_collector.py    # Comprehensive metrics collection
│   ├── results_exporter.py     # Exports results to Excel/JSON/plots
│   └── configs/                # Experiment configurations
│       ├── mnist_cnn.yaml      # MNIST CNN experiments
│       ├── mnist_vit.yaml      # MNIST ViT experiments
│       ├── cifar10_cnn.yaml    # CIFAR10 CNN experiments
│       └── cifar10_vit.yaml    # CIFAR10 ViT experiments
│
├── benchmarks/                  # Benchmarking tools
│   ├── comprehensive_benchmark.py  # Main benchmark runner
│   ├── memory_profiler.py      # Memory usage profiling
│   ├── speed_benchmark.py     # Training/inference speed benchmarking
│   └── optimizer_comparison.py # Optimizer comparison on synthetic functions
│
└── scripts/
    └── run_all_experiments.py # Alternative script for running experiments
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DL_Theory
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

Run all experiments with default settings:
```bash
python main.py
```

This will run:
- All optimizer comparisons on MNIST with CNN
- All optimizer comparisons on MNIST with ViT
- All optimizer comparisons on CIFAR10 with CNN
- All optimizer comparisons on CIFAR10 with ViT
- Generate Excel reports with all results
- Create visualization plots
- Generate summary reports

## Usage Examples

### Run all experiments with default settings
```bash
python main.py
```

### Run specific configuration file
```bash
python main.py --config experiments/configs/mnist_cnn.yaml
```

### Run with full benchmark suite (speed + memory tests)
```bash
python main.py --full
```

### Run only specific datasets
```bash
python main.py --datasets mnist cifar10
```

### Run only specific architectures
```bash
python main.py --architectures cnn vit
```

### Run only specific optimizers
```bash
python main.py --optimizers AMCAS Adam SGD
```

### Run with custom parameters
```bash
python main.py --epochs 20 --batch-size 128 --learning-rate 0.01 --data-augmentation
```

### Run on specific GPU
```bash
python main.py --gpu 0
```

### Run with custom output directory
```bash
python main.py --output my_results
```

## Experiment Configurations

The framework supports the following experiment combinations:

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
- **AMCAS**: Adaptive Momentum with Curvature-Aware Scaling (user's proposed optimizer)
- **Adam**: Adaptive Moment Estimation
- **AdamW**: Adam with decoupled weight decay
- **SGD**: Stochastic Gradient Descent
- **SGD+Momentum**: SGD with momentum (0.9)
- **RMSprop**: Root Mean Square Propagation
- **Adagrad**: Adaptive Gradient Algorithm
- **Adadelta**: Adaptive Learning Rate Method
- **NAdam**: Nesterov-accelerated Adaptive Moment Estimation
- **RAdam**: Rectified Adam

## Output Files

After running experiments, the following files are generated:

### Excel Reports (`results/experiment_results.xlsx`)
- **Summary**: Key metrics for all experiments
- **Detailed_Metrics**: Per-epoch metrics for all experiments
- **Comparison**: Ranked comparison of optimizers
- **Statistical_Analysis**: Statistical metrics and convergence analysis
- **Computational_Metrics**: Speed and memory usage metrics
- **Optimizer_Metrics**: Optimizer-specific metrics (gradient consistency, trust ratio, etc.)

### JSON Data (`results/experiment_results.json`)
Raw experiment data in JSON format for further analysis.

### Markdown Report (`results/experiment_report.md`)
Comprehensive report with:
- Summary table of all experiments
- Best performers by accuracy, speed, and memory efficiency
- Recommendations for optimizer selection
- Links to generated plots

### Visualization Plots (`results/plots/`)
- `loss_curves.png`: Training and test loss curves
- `accuracy_curves.png`: Training and test accuracy curves
- `f1_curves.png`: Test F1 score curves
- `time_comparison.png`: Training time comparison
- `memory_usage.png`: Memory usage comparison
- `gradient_stats.png`: Gradient statistics

### Summary (`results/summary.txt`)
Quick summary of experiment results and generated files.

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

## Benchmarking

The framework includes comprehensive benchmarking:

### Speed Benchmarking
- Measures training and inference speed for each optimizer
- Warmup iterations to stabilize measurements
- Batch size customization
- GPU/CPU performance comparison

### Memory Profiling
- Tracks GPU and CPU memory usage during training
- Peak memory consumption
- Average memory usage
- Memory efficiency comparison

### Synthetic Function Optimization
- Compares optimizers on standard test functions
- Convergence speed analysis
- Robustness to different initializations

## Customization

### Adding New Optimizers
1. Add optimizer implementation to `optimizers/` directory
2. Register optimizer in `experiment_runner.py` optimizer registry
3. Add default parameters in `default_optimizer_params`

### Adding New Models
1. Add model implementation to `models/` directory
2. Register model in appropriate factory function (`get_mnist_model()` or `get_cifar10_model()`)
3. Add model to configuration YAML files

### Adding New Datasets
1. Add dataset loading function to `experiment_runner.py`
2. Update `get_dataset()` method
3. Create appropriate data transforms

### Custom Experiment Configurations
Create YAML configuration files in `experiments/configs/` with the following structure:
```yaml
experiment_name: "custom_experiment"
dataset: "mnist"  # or "cifar10"
model: "simple_cnn"  # model name from registry
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

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- See `requirements.txt` for full dependencies

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