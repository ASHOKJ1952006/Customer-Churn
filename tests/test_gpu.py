import xgboost as xgb
import numpy as np

print("XGBoost version:", xgb.__version__)

# Small dummy dataset just to test GPU training works
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

model = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=50)
model.fit(X, y)

print("\nGPU training succeeded. Device used:", model.get_params()['device'])
print("If no error appeared above, your GPU setup is working correctly.")