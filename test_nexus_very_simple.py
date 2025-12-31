"""Very simple NEXUS test"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing NEXUS import...")
try:
    from optimizers.nexus import NEXUS
    print("[OK] NEXUS imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import NEXUS: {e}")
    sys.exit(1)

print("\nTesting NEXUS creation...")
try:
    import torch
    model = torch.nn.Linear(10, 10)
    optimizer = NEXUS(model.parameters(), lr=1e-3)
    print("[OK] NEXUS optimizer created")
except Exception as e:
    print(f"[ERROR] Failed to create NEXUS: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting NEXUS step...")
try:
    x = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"[OK] NEXUS step completed, loss: {loss.item():.6f}")
except Exception as e:
    print(f"[ERROR] Failed to run NEXUS step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting NEXUS statistics...")
try:
    lr_stats = optimizer.get_adaptive_lr_stats()
    print(f"[OK] Adaptive LR: {lr_stats['mean_adaptive_lr']:.6f}")
    
    dir_stats = optimizer.get_direction_consistency_stats()
    print(f"[OK] Direction Consistency: {dir_stats['mean_consistency']:.6f}")
    
    curv_stats = optimizer.get_curvature_stats()
    print(f"[OK] Curvature: {curv_stats['mean_curvature']:.6f}")
    
    mem_stats = optimizer.get_memory_usage()
    print(f"[OK] Memory: {mem_stats['total_mb']:.2f} MB")
except Exception as e:
    print(f"[ERROR] Failed to get statistics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All NEXUS tests passed!")

