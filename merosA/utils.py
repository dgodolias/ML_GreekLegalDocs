import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

def preprocess_data(processor, x_train, y_train, x_val, y_val, x_test, y_test):
    """Preprocess GLC data by converting to binary vectors and selecting top features."""
    binary_vectors_train = x_train  # Already processed by MatrixProcessing
    binary_vectors_val = x_val
    binary_vectors_test = x_test

    y_train = np.array(y_train).flatten()
    y_val = np.array(y_val).flatten()
    y_test = np.array(y_test).flatten()

    top_k = int(processor.vocab_size / 10)
    top_word_indices = processor.calculate_information_gain(binary_vectors_train, y_train, top_k=top_k)

    binary_vectors_train = binary_vectors_train[:, top_word_indices]
    binary_vectors_val = binary_vectors_val[:, top_word_indices]
    binary_vectors_test = binary_vectors_test[:, top_word_indices]

    return binary_vectors_train, y_train, binary_vectors_val, y_val, binary_vectors_test, y_test

def runMyLogisticRegression(logRegr, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=256):
    """Train and evaluate the custom multi-class Logistic Regression."""
    from tqdm import tqdm
    import time
    
    # Get number of classes for more accurate progress tracking
    n_classes = len(np.unique(y_train))
    
    # Initialize progress bar with proper total (iterations × classes)
    print("Training Multi-Class Logistic Regression (Custom)...")
    
    # Store the original fit method for the MultiClassLogisticRegression
    original_multi_fit = logRegr.fit
    
    # Create a wrapper that captures progress
    def fit_with_progress_tracking(*args, **kwargs):
        # Show progress bar only for the final metrics
        pbar = tqdm(total=100, desc="Finalizing Model", ncols=100)
        
        # Call the original fit method
        result = original_multi_fit(*args, **kwargs)
        
        # Complete the progress bar when done
        pbar.update(100)
        pbar.close()
        
        return result
        
    # Replace the fit method temporarily
    logRegr.fit = fit_with_progress_tracking
    
    # Train the model
    iteration_data = logRegr.fit(X_train, y_train, X_val, y_val, batch_size=batch_size)
    
    # Reset original method
    logRegr.fit = original_multi_fit
    
    # Make predictions
    y_pred_train = logRegr.predict(X_train)
    y_pred_val = logRegr.predict(X_val)
    y_pred_test = logRegr.predict(X_test)

    # Display only the final metrics table
    print("\n==== Final metrics table for CUSTOM Multi-Class Logistic Regression ====")
    compute_and_print_metrics(y_train, y_pred_train, "TRAINING data", "Multi-Class Logistic Regression")
    compute_and_print_metrics(y_val, y_pred_val, "VALIDATION data", "Multi-Class Logistic Regression")
    compute_and_print_metrics(y_test, y_pred_test, "TEST data", "Multi-Class Logistic Regression")

    return y_pred_train, y_pred_val, y_pred_test, iteration_data

def compute_accuracy(y_true, y_pred):
    """Compute accuracy for multi-class classification."""
    return np.mean(y_true == y_pred)

def compute_metrics_matrix(y_true, y_pred):
    """Compute precision, recall, and F1-score for each class, plus micro/macro averages."""
    classes = np.unique(y_true)
    n_classes = len(classes)
    metrics = []

    # Header
    metrics.append(["Category", "Precision", "Recall", "F1-Score"])

    # Per-class metrics
    for cls in classes:
        TP = np.sum((y_true == cls) & (y_pred == cls))
        FP = np.sum((y_true != cls) & (y_pred == cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics.append([f"Class {cls}", round(precision, 4), round(recall, 4), round(f1, 4)])

    # Micro-average
    TP_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
    FP_total = np.sum([np.sum((y_true != cls) & (y_pred == cls)) for cls in classes])
    FN_total = np.sum([np.sum((y_true == cls) & (y_pred != cls)) for cls in classes])
    micro_prec = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
    micro_rec = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0
    metrics.append(["Micro-Average", round(micro_prec, 4), round(micro_rec, 4), round(micro_f1, 4)])

    # Macro-average
    precisions = [float(row[1]) for row in metrics[1:n_classes+1]]
    recalls = [float(row[2]) for row in metrics[1:n_classes+1]]
    f1s = [float(row[3]) for row in metrics[1:n_classes+1]]
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    metrics.append(["Macro-Average", round(macro_prec, 4), round(macro_rec, 4), round(macro_f1, 4)])

    return np.array(metrics)

def compute_and_print_metrics(y_true, y_pred, dataset_description="Data", model_name="Model"):
    """Compute and print accuracy and detailed metrics for multi-class classification."""
    acc = compute_accuracy(y_true, y_pred)
    matrix = compute_metrics_matrix(y_true, y_pred)
    print(f"\n==== Metrics for {model_name} on {dataset_description} ====")
    print(f"Accuracy = {acc:.4f}")
    print(f"Model Evaluation Metrics ({dataset_description}):")
    print("+" + "-" * 50 + "+")
    for row in matrix:
        print("| {:<12} | {:<12} | {:<12} | {:<12} |".format(*row))
    print("+" + "-" * 50 + "+\n")

# Other functions (runMyAdaBoost, runMyXGBoost, etc.) remain unused for this task but kept for compatibility
def runMyAdaBoost(adaboost, X_train, y_train_transformed, X_val, y_val_transformed, X_test, y_test_transformed):
    print("Training Custom AdaBoost...")
    adaboost.fit(X_train, y_train_transformed)
    y_pred_train = adaboost.predict(X_train)
    y_pred_val = adaboost.predict(X_val)
    y_pred_test = adaboost.predict(X_test)
    y_pred_train = np.where(y_pred_train == -1, 0, 1)
    y_pred_val = np.where(y_pred_val == -1, 0, 1)
    y_pred_test = np.where(y_pred_test == -1, 0, 1)
    return y_pred_train, y_pred_val, y_pred_test

def runMyXGBoost(xgboost_model, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\nTraining Custom XGBoost...")
    xgboost_model.fit(X_train, y_train)
    y_pred_train = xgboost_model.predict(X_train)
    y_pred_val = xgboost_model.predict(X_val)
    y_pred_test = xgboost_model.predict(X_test)
    return y_pred_train, y_pred_val, y_pred_test