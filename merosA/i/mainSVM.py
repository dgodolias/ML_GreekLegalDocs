import numpy as np
import os
import sys
from tqdm import tqdm

# Add parent directories to path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from merosA.MatrixProcessing import MatrixProcessing
import merosA.utils as utils
from merosA.i.SVM import MultiClassSVM

# Disable TensorFlow warnings if not being used
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Starting Support Vector Machine (SVM) pipeline for Greek Legal Code...")

# Create outputs directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

levels = ['volume', 'chapter', 'subject']
models = {}

for level in levels:
    print(f"\n=== Processing {level.capitalize()} Level ===")
    
    # Initialize processor for the current level
    processor = MatrixProcessing(level=level, top_words=10000, skip_top_words=30, skip_least_frequent=20)
    x_train, y_train, x_val, y_val, x_test, y_test = processor.load_data()

    # Preprocess data (create binary vectors for BoW)
    binary_vectors_train, y_train_raw, \
    binary_vectors_val, y_val_raw, \
    binary_vectors_test, y_test_raw = utils.preprocess_data(
        processor,
        x_train, y_train,
        x_val, y_val,
        x_test, y_test
    )

    # Run SVM model with two different feature representations: BoW and TF-IDF
    feature_representations = [
        {"name": "BoW", "use_tfidf": False},
        {"name": "TF-IDF", "use_tfidf": True}
    ]
    
    for feature_rep in feature_representations:
        print(f"\n--- Using {feature_rep['name']} feature representation ---")
        
        # Initialize and train the model
        svm_model = MultiClassSVM(
            C=1.0,
            max_iter=1000,
            dual=True,
            use_tfidf=feature_rep["use_tfidf"],
            fit_intercept=True
        )
        
        # Train the model
        svm_model.fit(binary_vectors_train, y_train_raw, binary_vectors_val, y_val_raw)
        
        # Make predictions
        y_pred_train = svm_model.predict(binary_vectors_train)
        y_pred_val = svm_model.predict(binary_vectors_val)
        y_pred_test = svm_model.predict(binary_vectors_test)
        
        # Calculate and print metrics
        print(f"\nFinal Metrics for SVM with {feature_rep['name']} features:")
        utils.compute_and_print_metrics(y_train_raw, y_pred_train, "TRAINING data", f"SVM ({feature_rep['name']})")
        utils.compute_and_print_metrics(y_val_raw, y_pred_val, "VALIDATION data", f"SVM ({feature_rep['name']})")
        utils.compute_and_print_metrics(y_test_raw, y_pred_test, "TEST data", f"SVM ({feature_rep['name']})")
        
        # Save results to output file
        output_file = os.path.join(output_dir, f"{level}_{feature_rep['name'].lower()}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== SVM Results for {level.capitalize()} Level using {feature_rep['name']} ===\n\n")
            
            # Write training metrics
            acc_train = np.mean(y_pred_train == y_train_raw)
            f.write(f"Training Accuracy: {acc_train:.4f}\n")
            
            # Write validation metrics
            acc_val = np.mean(y_pred_val == y_val_raw)
            f.write(f"Validation Accuracy: {acc_val:.4f}\n")
            
            # Write test metrics
            acc_test = np.mean(y_pred_test == y_test_raw)
            f.write(f"Test Accuracy: {acc_test:.4f}\n\n")
            
            # Write detailed test metrics
            matrix = utils.compute_metrics_matrix(y_test_raw, y_pred_test)
            f.write("Detailed Test Metrics:\n")
            f.write("+" + "-" * 50 + "+\n")
            for row in matrix:
                f.write("| {:<12} | {:<12} | {:<12} | {:<12} |\n".format(*row))
            f.write("+" + "-" * 50 + "+\n")
        
        print(f"Results saved to {output_file}")
        
        # Store model for future use if needed
        model_key = f"{level}_{feature_rep['name'].lower()}"
        models[model_key] = svm_model

print("\nDone with SVM implementation!")
