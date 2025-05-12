import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from MatrixProcessing import MatrixProcessing
import utils
from LogisticRegression import MultiClassLogisticRegression

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting logistic regression pipeline for Greek Legal Code...")

levels = ['volume', 'chapter', 'subject']
models = {}

for level in levels:
    print(f"\n=== Processing {level.capitalize()} Level ===")
    
    # Define output path for metrics
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_file_prefix = os.path.join(output_dir, f"{level}_logregr")

    # Initialize processor for the current level
    processor = MatrixProcessing(level=level, top_words=10000, skip_top_words=30, skip_least_frequent=20)
    x_train, y_train, x_val, y_val, x_test, y_test = processor.load_data()

    # Preprocess data
    binary_vectors_train, y_train_raw, \
    binary_vectors_val, y_val_raw, \
    binary_vectors_test, y_test_raw = utils.preprocess_data(
        processor,
        x_train, y_train,
        x_val, y_val,
        x_test, y_test
    )

    # Initialize and train the model
    logRegr = MultiClassLogisticRegression(lr=0.01, n_iter=300, tolerance=1e-6, alpha=0.01)
    y_pred_train, y_pred_val, y_pred_test, custom_iteration_data = utils.runMyLogisticRegression(
        logRegr,
        binary_vectors_train, y_train_raw,
        binary_vectors_val, y_val_raw,
        binary_vectors_test, y_test_raw,
        batch_size=256,
        output_file_prefix=output_file_prefix  # Pass the prefix
    )
    models[level] = logRegr

print("\nDone!")