"""
test_gpu_comprehensive.py — Comprehensive GPU testing to verify actual GPU usage
"""

import xgboost as xgb
import numpy as np
import time

print("=== XGBoost GPU Test ===")
print("XGBoost version:", xgb.__version__)

# Check CUDA availability
try:
    from xgboost.core import _LIB
    print("XGBoost built with CUDA support:", _LIB.get_config()["USE_CUDA"])
except:
    print("Cannot check CUDA support via config")

# Create a larger dataset to ensure GPU utilization
print("\nCreating larger dataset (100K rows, 100 features)...")
X_large = np.random.rand(100000, 100).astype(np.float32)
y_large = np.random.randint(0, 2, 100000)

print("\n=== Testing GPU Training ===")
print("Watch your GPU usage in Task Manager during this test...")
time.sleep(2)

start = time.time()
model_gpu = xgb.XGBClassifier(
    tree_method='hist', 
    device='cuda', 
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)
model_gpu.fit(X_large, y_large)
gpu_time = time.time() - start

print(f"GPU training time: {gpu_time:.2f} seconds")
print("GPU model device:", model_gpu.get_params()['device'])

print("\n=== Testing CPU Training ===")
time.sleep(2)

start = time.time()
model_cpu = xgb.XGBClassifier(
    tree_method='hist', 
    device='cpu', 
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)
model_cpu.fit(X_large, y_large)
cpu_time = time.time() - start

print(f"CPU training time: {cpu_time:.2f} seconds")
print("CPU model device:", model_cpu.get_params()['device'])

print(f"\n=== Speedup ===")
print(f"GPU vs CPU speedup: {cpu_time/gpu_time:.2f}x")

if cpu_time/gpu_time > 1.5:
    print("✅ GPU is working and providing significant speedup")
else:
    print("❌ GPU may not be utilized properly - speedup is minimal")
