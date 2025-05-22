import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC # Import SVC
from utils import (
    run_classification_and_report, # Updated name
    load_and_preprocess_data,
    perform_k_fold_cv_and_report,
    script_execution_timer
)

# --- Configuration ---
DATASET_CONFIG = "chapter" # available options: "volume", "chapter", "subject"
USE_SUBSET_FOR_CV_DATA = False
SUBSET_PERCENTAGE = 0.2 
PERFORM_K_FOLD_CV = True
N_SPLITS_CV = 5
RANDOM_STATE = 42
VECTORIZER_MAX_FEATURES = 5000
# --- End Configuration ---

@script_execution_timer
def main_svm_bow(): # Renamed for clarity
    model_name_script = "SVM"
    feature_method_name = "BoW"
    
    # For output directory and report naming
    base_run_name = f"{model_name_script}_{feature_method_name}" 
    
    print(f"Starting {model_name_script} with {feature_method_name} script...")
    
    feature_config = {
        'method': 'sklearn_vectorizer',
        'class': CountVectorizer,
        'params': {'max_features': VECTORIZER_MAX_FEATURES}
    }
    
    model_class = SVC
    # Add any specific SVC params here, random_state will be handled by utils if applicable
    model_init_params = {'kernel': 'linear', 'probability': True} # probability=True if you need predict_proba later

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, "outputs")
    
    if PERFORM_K_FOLD_CV:
        run_output_dir = os.path.join(output_base_dir, f"{base_run_name}_{DATASET_CONFIG}_cv_reports")
    else:
        run_output_dir = os.path.join(output_base_dir, f"{base_run_name}_{DATASET_CONFIG}_single_run_reports")
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

    # Descriptive name for reports
    descriptive_run_identifier = f"{model_class.__name__} with {feature_method_name}"

    if PERFORM_K_FOLD_CV:
        success = perform_k_fold_cv_and_report(
            texts_to_process=texts_proc, 
            labels_to_process=labels_proc,
            unique_labels_to_process=unique_labels_proc,
            unique_labels_to_process_str=unique_labels_proc_str,
            n_splits_cv=N_SPLITS_CV,
            feature_config=feature_config, # Pass feature_config
            model_class=model_class,       # Pass model_class
            model_init_params=model_init_params, # Pass model_init_params
            run_identifier_string=f"{descriptive_run_identifier} ({DATASET_CONFIG})", # Updated identifier
            cv_output_dir=run_output_dir,
            random_state=RANDOM_STATE
        )
        if not success:
            print("K-Fold Cross-Validation did not complete successfully.")
    else: 
        print("\nSplitting data into a single training and testing set...")
        from sklearn.model_selection import train_test_split as sk_train_test_split

        X_train_texts, X_test_texts, y_train, y_test = sk_train_test_split(
            texts_proc, labels_proc, test_size=0.2, random_state=RANDOM_STATE,
            stratify=labels_proc if len(set(labels_proc)) > 1 else None
        )
        print(f"Data split. Train: {len(X_train_texts)}, Test: {len(X_test_texts)}")
        if not X_train_texts or len(X_train_texts) == 0: print("Error: Training set empty."); return

        print(f"\n--- Processing with {descriptive_run_identifier} for single split on '{DATASET_CONFIG}' data ---")
        vectorizer = feature_config['class'](**feature_config['params'])
        X_train_transformed = vectorizer.fit_transform(X_train_texts)
        X_test_transformed = vectorizer.transform(X_test_texts)
        
        run_classification_and_report( # Updated function call
            X_train_transformed, X_test_transformed, y_train, y_test, 
            model_class=model_class,
            model_init_params=model_init_params,
            run_identifier_string=f"{descriptive_run_identifier} (Single Split - {DATASET_CONFIG})",
            labels=unique_labels_proc, 
            target_names=unique_labels_proc_str,
            output_dir=run_output_dir,
            random_state=RANDOM_STATE
        )

    print(f"\n{descriptive_run_identifier} script finished for {DATASET_CONFIG} dataset.")

if __name__ == "__main__":
    main_svm_bow()