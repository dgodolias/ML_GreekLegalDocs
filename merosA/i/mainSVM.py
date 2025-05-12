import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from MatrixProcessing import MatrixProcessing
from CustomSVC import CustomSVC, CustomOneVsRestClassifier
import utils

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting Custom SVM pipeline for Greek Legal Code...")

levels = ['volume', 'chapter', 'subject']
representations = ['bow', 'tfidf']

for level in levels:
    for representation in representations:
        print(f"\n=== Processing {level.capitalize()} Level with {representation.upper()} ===")
        
        # Initialize processor with representation
        processor = MatrixProcessing(level=level, top_words=10000, skip_top_words=30, skip_least_frequent=20, representation=representation)
        x_train, y_train, x_val, y_val, x_test, y_test = processor.load_data()

        # Preprocess data using utils
        binary_vectors_train, y_train_raw, \
        binary_vectors_val, y_val_raw, \
        binary_vectors_test, y_test_raw = utils.preprocess_data(
            processor,
            x_train, y_train,
            x_val, y_val,
            x_test, y_test
        )

        # Initialize Custom SVM with OneVsRest strategy
        n_classes = len(np.unique(y_train_raw))
        
        # Create a custom estimator that includes progress tracking
        def create_svm_with_progress():
            # Create a custom SVC with modified fit method
            svc = CustomSVC(learning_rate=0.01, lambda_param=0.01, n_iters=100)
            original_fit = svc.fit
            
            # Override the fit method to include progress bar
            def fit_with_progress(X, y):
                n_samples, n_features = X.shape
                svc.weights = np.zeros(n_features)
                svc.bias = 0
                
                # Use tqdm for progress tracking during iterations
                for _ in tqdm(range(svc.n_iters), desc="Training SVM", leave=False):
                    for idx, x_i in enumerate(X):
                        condition = y[idx] * (np.dot(x_i, svc.weights) - svc.bias) >= 1
                        if condition:
                            svc.weights -= svc.lr * (2 * svc.lambda_param * svc.weights)
                        else:
                            svc.weights -= svc.lr * (2 * svc.lambda_param * svc.weights - np.dot(x_i, y[idx]))
                            svc.bias -= svc.lr * y[idx]
            
            # Replace the original fit method
            svc.fit = fit_with_progress
            return svc
        
        svm_model = CustomOneVsRestClassifier(
            estimator=create_svm_with_progress,
            n_classes=n_classes
        )

        # Train the model with progress bar for each classifier
        print("Training Custom SVM model...")
        print(f"Training {n_classes} binary classifiers (one-vs-rest)")
        
        # Modify the OneVsRest fit method to show progress across classifiers
        original_ovr_fit = svm_model.fit
        def fit_with_class_progress(X, y):
            for i in tqdm(range(svm_model.n_classes), desc="Classifier Progress", position=0):
                y_binary = (y == i).astype(int)
                print(f"\nTraining classifier for class {i+1}/{svm_model.n_classes}")
                svm_model.classifiers[i].fit(X, y_binary)
        
        # Replace the fit method
        svm_model.fit = fit_with_class_progress
        
        # Train with progress tracking
        svm_model.fit(binary_vectors_train, y_train_raw)

        # Make predictions
        y_pred_train = svm_model.predict(binary_vectors_train)
        y_pred_val = svm_model.predict(binary_vectors_val)
        y_pred_test = svm_model.predict(binary_vectors_test)

        # Evaluate the model
        print("\nFinal Metrics for Custom SVM:")
        utils.compute_and_print_metrics(y_train_raw, y_pred_train, "TRAINING data", f"Custom SVM ({representation.upper()})")
        utils.compute_and_print_metrics(y_val_raw, y_pred_val, "VALIDATION data", f"Custom SVM ({representation.upper()})")
        utils.compute_and_print_metrics(y_test_raw, y_pred_test, "TEST data", f"Custom SVM ({representation.upper()})")

print("\nDone with Custom SVM!")