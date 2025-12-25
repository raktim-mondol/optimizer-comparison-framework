"""
Simple Example of AMCAS Optimizer Usage

This example shows basic usage of the AMCAS optimizer on a simple neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizers.amcas import AMCAS


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    """Main function demonstrating AMCAS usage."""
    print("AMCAS Optimizer Simple Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and dummy data
    model = SimpleNet()
    dummy_input = torch.randn(32, 10)  # Batch of 32 samples, 10 features each
    dummy_target = torch.randn(32, 1)   # Random targets
    
    # Create AMCAS optimizer
    optimizer = AMCAS(
        model.parameters(),
        lr=0.001,                    # Learning rate
        betas=(0.9, 0.999),           # Coefficients for running averages
        gamma=0.1,                    # Curvature update rate
        lambda_consistency=0.01,     # Gradient consistency sensitivity
        trust_region_params=(0.8, 1.2, 1.5, 0.5),  # Trust region parameters
        eps=1e-8,                     # Numerical stability term
        weight_decay=0.01,           # L2 regularization
        amsgrad=False                 # Whether to use AMSGrad variant
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    print("\nTraining loop:")
    print("-" * 30)
    
    # Training loop
    for epoch in range(5):
        # Forward pass
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Get gradient statistics before step
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(torch.norm(param.grad).item())
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        
        # Optimization step
        optimizer.step()
        
        # Get optimizer statistics
        trust_ratio = optimizer.get_trust_ratio()
        curvature_stats = optimizer.get_curvature_stats()
        gradient_consistency = optimizer.get_gradient_consistency()
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Avg Gradient Norm: {avg_grad_norm:.6f}")
        print(f"  Trust Ratio: {trust_ratio:.6f}")
        print(f"  Gradient Consistency: {gradient_consistency:.6f}")
        print(f"  Mean Curvature: {curvature_stats['mean_curvature']:.6f}")
        print(f"  Curvature Std: {curvature_stats['std_curvature']:.6f}")
        print()
    
    print("\nOptimizer Statistics:")
    print("-" * 30)
    
    # Get detailed optimizer statistics
    from optimizers.utils import get_optimizer_statistics
    stats = get_optimizer_statistics(optimizer)
    
    print(f"Total parameters: {stats['total_parameters']}")
    print(f"State size: {stats['state_size_bytes'] / 1024:.2f} KB")
    print(f"Learning rates: {stats['learning_rates']}")
    print(f"Momentum values: {stats.get('momentum_values', 'N/A')}")
    print(f"Average trust ratio: {stats.get('avg_trust_ratio', 'N/A'):.6f}")
    
    print("\nComparison with other optimizers:")
    print("-" * 30)
    
    # Compare with Adam
    model_adam = SimpleNet()
    optimizer_adam = torch.optim.Adam(
        model_adam.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Compare with SGD with momentum
    model_sgd = SimpleNet()
    optimizer_sgd = torch.optim.SGD(
        model_sgd.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.01
    )
    
    # Train all three for comparison
    models = {
        'AMCAS': (model, optimizer),
        'Adam': (model_adam, optimizer_adam),
        'SGD+Momentum': (model_sgd, optimizer_sgd)
    }
    
    final_losses = {}
    for name, (model, optimizer) in models.items():
        # Reset model
        for param in model.parameters():
            param.data = torch.randn_like(param)
        
        # Train for a few steps
        for _ in range(100):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
        
        final_losses[name] = loss.item()
    
    print("\nFinal losses after 100 steps:")
    for name, loss in sorted(final_losses.items(), key=lambda x: x[1]):
        print(f"  {name:15} {loss:.6f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)


def advanced_example():
    """Advanced example showing AMCAS features."""
    print("\n\nAdvanced AMCAS Features Example")
    print("=" * 50)
    
    # Create a more complex model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    
    # AMCAS with custom parameters
    optimizer = AMCAS(
        model.parameters(),
        lr=0.001,
        betas=(0.95, 0.999),      # Higher momentum
        gamma=0.05,                # Slower curvature adaptation
        lambda_consistency=0.001,  # Less sensitive to gradient changes
        trust_region_params=(0.7, 1.3, 2.0, 0.3),  # More aggressive trust region
        eps=1e-6,                  # Smaller epsilon for numerical stability
        weight_decay=0.001,        # L2 regularization
        amsgrad=True               # Use AMSGrad variant
    )
    
    print("AMCAS optimizer created with:")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Betas: {optimizer.param_groups[0]['betas']}")
    print(f"  Gamma: {optimizer.param_groups[0]['gamma']}")
    print(f"  Lambda consistency: {optimizer.param_groups[0]['lambda_consistency']}")
    print(f"  Trust region params: {optimizer.param_groups[0]['trust_region_params']}")
    print(f"  Epsilon: {optimizer.param_groups[0]['eps']}")
    print(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    print(f"  AMSGrad: {optimizer.param_groups[0]['amsgrad']}")
    
    # Demonstrate gradient clipping
    print("\nGradient clipping example:")
    dummy_input = torch.randn(64, 100)
    dummy_target = torch.randn(64, 10)
    criterion = nn.MSELoss()
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    
    # Clip gradients before optimizer step
    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    print(f"  Gradient norm before clipping: {total_norm:.6f}")
    
    optimizer.step()
    print("  Optimization step completed with gradient clipping")
    
    # Show optimizer state
    print("\nOptimizer state after one step:")
    print(f"  Number of parameter groups: {len(optimizer.param_groups)}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Check if AMSGrad max_exp_avg_sq is initialized
    if optimizer.param_groups[0]['amsgrad']:
        print("  AMSGrad: max_exp_avg_sq state initialized")
        # Check one parameter's state
        for name, param in model.named_parameters():
            if param.requires_grad and param in optimizer.state:
                state = optimizer.state[param]
                if 'max_exp_avg_sq' in state:
                    print(f"  Parameter '{name}' has max_exp_avg_sq state")
                    break


def parameter_tuning_tips():
    """Provide tips for tuning AMCAS parameters."""
    print("\n\nAMCAS Parameter Tuning Tips")
    print("=" * 50)
    
    tips = [
        ("Learning Rate (lr)", 
         "Start with 0.001. AMCAS is less sensitive to learning rate than Adam due to trust region adaptation."),
        
        ("Betas", 
         "Default (0.9, 0.999) works well. First beta controls momentum, second controls squared gradient accumulation."),
        
        ("Gamma", 
         "Controls curvature update rate. Default 0.1. Higher values make curvature adapt faster but may be noisy."),
        
        ("Lambda Consistency", 
         "Controls sensitivity to gradient changes. Default 0.01. Higher values make momentum decay faster with noisy gradients."),
        
        ("Trust Region Params", 
         "(eta_low, eta_high, tau_increase, tau_decrease). Default (0.8, 1.2, 1.5, 0.5). Adjust based on problem stability."),
        
        ("Weight Decay", 
         "L2 regularization. Default 0. Can help with generalization."),
        
        ("AMSGrad", 
         "Set to True for non-increasing adaptive learning rates. Helps with convergence guarantees."),
        
        ("Epsilon", 
         "Numerical stability term. Default 1e-8. Don't change unless you have numerical issues.")
    ]
    
    for param, tip in tips:
        print(f"{param}:")
        print(f"  {tip}")
        print()


if __name__ == '__main__':
    main()
    advanced_example()
    parameter_tuning_tips()