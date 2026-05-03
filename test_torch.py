import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Test tensor ops
    x = torch.rand(5, 5)
    y = torch.matmul(x, x)
    print("CPU Tensor ops successful.")
    
    if torch.backends.mps.is_available():
        x_mps = x.to('mps')
        y_mps = torch.matmul(x_mps, x_mps)
        print("MPS Tensor ops successful.")
        
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
except BaseException as e: # Catch segfaults if possible, though unlikely in pure python
    print(f"FATAL: {e}")
