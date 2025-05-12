import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from merosA.MatrixProcessing import MatrixProcessing
import merosA.utils as utils
from LogisticRegression import LogisticRegression


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting logistic regression pipeline...")


processor = MatrixProcessing(top_words=10000, skip_top_words=30, skip_least_frequent=20)
x_train, y_train, x_val, y_val, x_test, y_test = processor.load_data()


binary_vectors_train, y_train_raw, \
binary_vectors_val,   y_val_raw,   \
binary_vectors_test,  y_test_raw   = utils.preprocess_data(
    processor,
    x_train, y_train,
    x_val,   y_val,
    x_test,  y_test
)

#Our Logistic Regression model
logRegr = LogisticRegression(lr=0.01, n_iter=1000, tolerance=1e-6, alpha=0.01)

y_pred_train, y_pred_val, y_pred_test,custom_iteration_data = utils.runMyLogisticRegression(
    logRegr,
    binary_vectors_train, y_train_raw,
    binary_vectors_val,   y_val_raw,
    binary_vectors_test,  y_test_raw,
    batch_size=256
)


print("\nDone!")
