import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sklearn.feature_extraction.text import CountVectorizer
from utils import (
    run_svm_classification, 
    load_and_preprocess_data,
    perform_k_fold_cv_and_report
)
# Note: format_mean_report_to_string is used internally by perform_k_fold_cv_and_report

# --- Configuration ---
DATASET_CONFIG = "volume" # available options: "volume", "chapter", "subject"
USE_SUBSET_FOR_CV_DATA = True
SUBSET_PERCENTAGE = 0.2 
PERFORM_K_FOLD_CV = True
N_SPLITS_CV = 5
RANDOM_STATE = 42
VECTORIZER_MAX_FEATURES = 5000
# --- End Configuration ---

def main_bow():
    print("Starting BoW SVM script...")
    feature_name = "BoW"
    vectorizer_class = CountVectorizer
    vectorizer_params = {'max_features': VECTORIZER_MAX_FEATURES}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, "outputs")
    
    if PERFORM_K_FOLD_CV:
        run_output_dir = os.path.join(output_base_dir, f"bow_{DATASET_CONFIG}_cv_reports")
    else:
        run_output_dir = os.path.join(output_base_dir, f"bow_{DATASET_CONFIG}_single_run_reports")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"INFO: Reports will be saved in '{run_output_dir}'")

    print(f"INFO: Script configured to use a {SUBSET_PERCENTAGE*100:.0f}% subset." if USE_SUBSET_FOR_CV_DATA else "INFO: Using all data.")
    print(f"INFO: K-Fold CV is {'ENABLED' if PERFORM_K_FOLD_CV else 'DISABLED'} with {N_SPLITS_CV} splits.")

    texts_proc, labels_proc, unique_labels_proc, unique_labels_proc_str = load_and_preprocess_data(
        DATASET_CONFIG, USE_SUBSET_FOR_CV_DATA, SUBSET_PERCENTAGE, RANDOM_STATE
    )

    if texts_proc is None:
        print("Failed to load or preprocess data. Exiting.")
        return

    if PERFORM_K_FOLD_CV:
        success = perform_k_fold_cv_and_report(
            texts_to_process=texts_proc, 
            labels_to_process=labels_proc,
            unique_labels_to_process=unique_labels_proc,
            unique_labels_to_process_str=unique_labels_proc_str,
            n_splits_cv=N_SPLITS_CV,
            vectorizer_class=vectorizer_class,
            vectorizer_params=vectorizer_params,
            feature_name=feature_name,
            cv_output_dir=run_output_dir,
            random_state=RANDOM_STATE
        )
        if not success:
            print("K-Fold Cross-Validation did not complete successfully.")
    else: 
        print("\nSplitting data into a single training and testing set...")
        from sklearn.model_selection import train_test_split as sk_train_test_split # Local import for clarity

        X_train_texts, X_test_texts, y_train, y_test = sk_train_test_split(
            texts_proc, labels_proc, test_size=0.2, random_state=RANDOM_STATE,
            stratify=labels_proc if len(set(labels_proc)) > 1 else None
        )
        print(f"Data split. Train: {len(X_train_texts)}, Test: {len(X_test_texts)}")
        if not X_train_texts or len(X_train_texts) == 0:
            print("Error: Training set empty after split.")
            return

        print(f"\n--- Processing with {feature_name} for single split on '{DATASET_CONFIG}' data ---")
        vectorizer = vectorizer_class(**vectorizer_params)
        X_train_transformed = vectorizer.fit_transform(X_train_texts)
        X_test_transformed = vectorizer.transform(X_test_texts)
        
        run_svm_classification(
            X_train_transformed, X_test_transformed, y_train, y_test, 
            feature_type=f"{feature_name} (Single Split - {DATASET_CONFIG})",
            labels=unique_labels_proc, 
            target_names=unique_labels_proc_str,
            output_dir=run_output_dir,
            random_state=RANDOM_STATE
        )

    print(f"\n{feature_name} SVM script finished for {DATASET_CONFIG} dataset.")

if __name__ == "__main__":
    main_bow()