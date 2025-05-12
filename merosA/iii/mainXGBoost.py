import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from merosA.MatrixProcessing import MatrixProcessing
import merosA.utils as utils
from merosA.iii.XGBoost import MultiClassXGBoost

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting the XGBoost (from Scratch) model for Greek Legal Code")

levels = ['volume', 'chapter', 'subject']
models = {}

for level in levels:
    print(f"\n=== Processing {level.capitalize()} Level ===")

    # Define output path for metrics
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_file_prefix = os.path.join(output_dir, f"{level}_xgboost_scratch")
    
    processor = MatrixProcessing(level=level, top_words=10000, skip_top_words=30, skip_least_frequent=20)
    x_train, y_train, x_val, y_val, x_test, y_test = processor.load_data()

    binary_vectors_train, y_train_raw, \
    binary_vectors_val, y_val_raw, \
    binary_vectors_test, y_test_raw = utils.preprocess_data(
        processor,
        x_train, y_train,
        x_val, y_val,
        x_test, y_test
    )

    n_classes = len(np.unique(y_train_raw))

    xgboost_model = MultiClassXGBoost(
        n_classes=n_classes,
        n_estimators=3,
        learning_rate=0.1,
        max_depth=3,
        reg_lambda=1.0,
        gamma=0.1,
        min_samples_split=5,
        min_impurity_decrease=0.01
    )

    xgboost_model.fit(binary_vectors_train, y_train_raw)

    y_pred_train = xgboost_model.predict(binary_vectors_train)
    y_pred_val = xgboost_model.predict(binary_vectors_val)
    y_pred_test = xgboost_model.predict(binary_vectors_test)

    print("\nFinal Metrics for XGBoost (from Scratch):")
    train_save_path = f"{output_file_prefix}_train.txt"
    val_save_path = f"{output_file_prefix}_val.txt"
    test_save_path = f"{output_file_prefix}_test.txt"

    utils.compute_and_print_metrics(y_train_raw, y_pred_train, "TRAINING data", "XGBoost (Scratch)", save_path=train_save_path)
    utils.compute_and_print_metrics(y_val_raw, y_pred_val, "VALIDATION data", "XGBoost (Scratch)", save_path=val_save_path)
    utils.compute_and_print_metrics(y_test_raw, y_pred_test, "TEST data", "XGBoost (Scratch)", save_path=test_save_path)

print("\nDone with XGBoost (from Scratch)!")