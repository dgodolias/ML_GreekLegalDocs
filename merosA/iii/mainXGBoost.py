import numpy as np
import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from merosA.MatrixProcessing import MatrixProcessing
import merosA.utils as utils
from merosA.iii.XGBoost import XGBoostScratch # Using the from-scratch version

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting the XGBoost (from Scratch) model")

processor = MatrixProcessing(top_words=10000, skip_top_words=30, skip_least_frequent=20)
x_train, y_train, x_val, y_val, x_test, y_test = processor.load_data()

binary_vectors_train, y_train_raw, \
binary_vectors_val,   y_val_raw,   \
binary_vectors_test,  y_test_raw   = utils.preprocess_data(
    processor, 
    x_train,  y_train, 
    x_val,    y_val, 
    x_test,   y_test
)

# XGBoost from scratch expects labels as 0 or 1, which y_train_raw etc. already are.

# Our XGBoost (from Scratch) model
# Parameters can be adjusted here.
# Note: Training from scratch can be slow, especially with many estimators or deep trees.
# Consider starting with fewer estimators for testing.
xgboost_scratch_model = XGBoostScratch(
    n_estimators=200,      # Reduced for faster initial testing
    learning_rate=0.1,
    max_depth=3,          # Shallow trees are common in boosting
    reg_lambda=1.0,       # L2 regularization
    gamma=0.1,            # Min loss reduction for split
    min_samples_split=5,
    min_impurity_decrease=0.01 # Corresponds to min_gain_to_split
)

# Using raw labels (0 or 1) for XGBoost
y_pred_train, y_pred_val, y_pred_test = utils.runMyModel_scratch( # New utility function
    xgboost_scratch_model,
    binary_vectors_train, y_train_raw,
    binary_vectors_val,   y_val_raw,
    binary_vectors_test,  y_test_raw,
    model_name="XGBoost (Scratch)"
)

print("\nFinal Test Metrics for XGBoost (from Scratch):")
utils.compute_and_print_metrics(y_test_raw, y_pred_test, model_name="XGBoost (Scratch)")

print("\nDone with XGBoost (from Scratch)!")
