from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np # Needed for handling unique labels in classification_report

def run_svm_classification(X_train_features, X_test_features, y_train, y_test, feature_type="BoW"):
    """
    Trains an SVM classifier, makes predictions, and evaluates the model.
    """
    print(f"\n--- Running SVM Classification with {feature_type} ---")
    # 4. Train SVM classifier
    print(f"Training SVM classifier with {feature_type} features...")
    start_time = time.time()
    svm_classifier = SVC(kernel='linear') # Linear kernel is often good for text
    svm_classifier.fit(X_train_features, y_train)
    print(f"SVM classifier trained in {time.time() - start_time:.2f} seconds.")

    # 5. Make predictions
    print("Making predictions on the test set...")
    start_time = time.time()
    y_pred = svm_classifier.predict(X_test_features)
    print(f"Predictions made in {time.time() - start_time:.2f} seconds.")

    # 6. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set with {feature_type}: {accuracy:.4f}")

    # Generate and print classification report
    print(f"\nClassification Report for {feature_type}:")
    # Ensure labels are sorted for consistent report display, especially if they are not 0-indexed integers
    # Get unique labels present in y_test or y_pred to avoid warnings for labels not in predictions
    # However, classification_report handles this well by default if target_names are not provided or labels are specified.
    # For simplicity, we'll let classification_report determine the labels.
    # If y_test contains string labels, convert them to a consistent format if needed or ensure they match y_pred.
    
    # Get unique labels from both y_test and y_pred to handle cases where some classes might not be in y_pred
    # and to ensure correct ordering if labels are not simple integers.
    # Convert to list and sort for consistent output, especially if labels are strings or non-sequential.
    # Note: sklearn's classification_report can often infer labels, but explicitly providing them
    # can be more robust, especially if some classes in y_test are not predicted at all.
    # For integer labels, this is usually fine.
    
    # We need to ensure that the labels used for the report are the ones present in the test data.
    # And that they are sorted for consistent output.
    report_labels = sorted(list(np.unique(np.concatenate((y_test, y_pred)))))
    
    # If your labels are not simple integers (e.g., strings), you might want to create target_names
    # target_names = [f"Class {label}" for label in report_labels]
    # print(classification_report(y_test, y_pred, labels=report_labels, target_names=target_names, zero_division=0))
    
    # For integer labels as in this dataset, directly passing y_test and y_pred is usually sufficient.
    # The `zero_division=0` parameter prevents warnings if a class has no predicted samples (precision/F1 would be 0).
    try:
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
    except ValueError as e:
        print(f"Could not generate classification report: {e}")
        print("This might happen if y_test or y_pred is empty, or if labels are inconsistent.")

    return accuracy