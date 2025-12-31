"""Check if torch is available."""
import sys

try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    
    # Test basic operations
    x = torch.randn(5, 3)
    y = torch.randn(5, 3)
    z = x + y
    print(f"[OK] Basic tensor operations work")
    
    # Test NEXUS import
    from optimizers.nexus import NEXUS
    print(f"[OK] NEXUS import successful")
    
    # Test NEXUS creation
    model = torch.nn.Linear(10, 10)
    optimizer = NEXUS(model.parameters(), lr=1e-3)
    print(f"[OK] NEXUS optimizer created")
    
    # Test one step
    x = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"[OK] NEXUS optimization step works")
    
    # Test statistics
    lr_stats = optimizer.get_adaptive_lr_stats()
    print(f"[OK] Adaptive LR stats: {lr_stats['mean_adaptive_lr']:.6f}")
    
    dir_stats = optimizer.get_direction_consistency_stats()
    print(f"[OK] Direction consistency: {dir_stats['mean_consistency']:.6f}")
    
    curv_stats = optimizer.get_curvature_stats()
    print(f"[OK] Curvature: {curv_stats['mean_curvature']:.6f}")
    
    mem_stats = optimizer.get_memory_usage()
    print(f"[OK] Memory usage: {mem_stats['total_mb']:.2f} MB")
    
    print("\n[SUCCESS] All checks passed! NEXUS is working correctly!")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

