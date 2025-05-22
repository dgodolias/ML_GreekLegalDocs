import sys
import os
import numpy as np
import joblib # For saving/loading sklearn vectorizers
from sklearn.model_selection import train_test_split as sk_train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC # Import SVC
from utils import (
    run_classification_and_report, 
    load_and_preprocess_data,
    perform_k_fold_cv_and_report,
    script_execution_timer # Added import
)

# --- Configuration ---
DATASET_CONFIG = "chapter" # available options: "volume", "chapter", "subject"
USE_SUBSET_FOR_CV_DATA = False
SUBSET_PERCENTAGE = 0.2 
PERFORM_K_FOLD_CV = True # Set to False to test single run
N_SPLITS_CV = 5
RANDOM_STATE = 42
VECTORIZER_MAX_FEATURES = 5000

# Feature Engineering Saving/Loading
SAVE_TRAINED_FEATURE_MODELS = True
LOAD_TRAINED_FEATURE_MODELS_IF_EXIST = True
# --- End Configuration ---

@script_execution_timer
def main_svm_tfidf(): 
    model_name_script = "SVM"
    feature_method_name = "TF-IDF"
    
    base_run_name = f"{model_name_script}_{feature_method_name.replace('-', '')}" 
    
    print(f"Starting {model_name_script} with {feature_method_name} script...")
    
    feature_config = {
        'method': 'sklearn_vectorizer',
        'class': TfidfVectorizer,
        'params': {'max_features': VECTORIZER_MAX_FEATURES}
    }
    
    model_class = SVC
    model_init_params = {'kernel': 'linear', 'probability': True, 'random_state': RANDOM_STATE} 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, "outputs")

    single_run_feature_models_dir = os.path.join(output_base_dir, "feature_models_single_run")
    if SAVE_TRAINED_FEATURE_MODELS or LOAD_TRAINED_FEATURE_MODELS_IF_EXIST:
        os.makedirs(single_run_feature_models_dir, exist_ok=True)
    
    if PERFORM_K_FOLD_CV:
        run_output_dir = os.path.join(output_base_dir, f"{base_run_name}_{DATASET_CONFIG}_cv_reports")
    else:
        run_output_dir = os.path.join(output_base_dir, f"{base_run_name}_{DATASET_CONFIG}_single_run_reports")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"INFO: Reports will be saved in '{run_output_dir}'")

    actual_use_subset = USE_SUBSET_FOR_CV_DATA
    actual_subset_percentage = SUBSET_PERCENTAGE

    print(f"INFO: Script configured to use a {actual_subset_percentage*100:.0f}% subset." if actual_use_subset else "INFO: Using all data.")
    print(f"INFO: K-Fold CV is {'ENABLED' if PERFORM_K_FOLD_CV else 'DISABLED'}{f' with {N_SPLITS_CV} splits' if PERFORM_K_FOLD_CV else ''}.")
    print(f"INFO: Saving of trained feature models is {'ENABLED' if SAVE_TRAINED_FEATURE_MODELS else 'DISABLED'}.")
    print(f"INFO: Loading of existing feature models is {'ENABLED' if LOAD_TRAINED_FEATURE_MODELS_IF_EXIST else 'DISABLED'}.")

    texts_proc, labels_proc, unique_labels_proc, unique_labels_proc_str = load_and_preprocess_data(
        DATASET_CONFIG, actual_use_subset, actual_subset_percentage, RANDOM_STATE
    )

    if texts_proc is None or len(texts_proc) == 0:
        print("Failed to load or preprocess data, or no data remains after filtering. Exiting.")
        return

    descriptive_run_identifier = f"{model_class.__name__} with {feature_method_name}"
    sanitized_run_id_for_filename = "".join(c if c.isalnum() else "_" for c in descriptive_run_identifier.split('(')[0].strip())


    if PERFORM_K_FOLD_CV:
        if unique_labels_proc and labels_proc is not None:
            counts = np.unique(labels_proc, return_counts=True)[1]
            if np.any(counts < N_SPLITS_CV):
                print(f"CRITICAL ERROR: Not enough samples in some classes for {N_SPLITS_CV}-Fold CV after preprocessing.")
                return
        success = perform_k_fold_cv_and_report(
            texts_to_process=texts_proc, 
            labels_to_process=labels_proc,
            unique_labels_to_process=unique_labels_proc,
            unique_labels_to_process_str=unique_labels_proc_str,
            n_splits_cv=N_SPLITS_CV,
            feature_config=feature_config,      
            model_class=model_class,            
            model_init_params=model_init_params, 
            run_identifier_string=f"{descriptive_run_identifier} ({DATASET_CONFIG})", 
            cv_output_dir=run_output_dir,
            random_state=RANDOM_STATE,
            save_trained_features=SAVE_TRAINED_FEATURE_MODELS,
            load_trained_features_if_exist=LOAD_TRAINED_FEATURE_MODELS_IF_EXIST
        )
        if not success:
            print("K-Fold Cross-Validation did not complete successfully.")
    else: 
        print("\nSplitting data into a single training and testing set (80/20 split)...")
        
        can_stratify_single_split = False
        if labels_proc is not None and len(labels_proc) > 0:
            unique_labels_single, counts_single = np.unique(labels_proc, return_counts=True)
            if len(unique_labels_single) > 1 and all(counts_single >= 2):
                can_stratify_single_split = True
            elif len(unique_labels_single) == 1:
                 can_stratify_single_split = False
            else:
                print("Warning: Some classes have less than 2 samples. Stratification for single split might fail.")

        X_train_texts, X_test_texts, y_train, y_test = sk_train_test_split(
            texts_proc, labels_proc, test_size=0.2, random_state=RANDOM_STATE,
            stratify=labels_proc if can_stratify_single_split else None
        )
        print(f"Data split. Train texts: {len(X_train_texts)}, Test texts: {len(X_test_texts)}")
        if not X_train_texts or len(X_train_texts) == 0: 
            print("Error: Training set empty after split."); return

        print(f"\n--- Processing with {descriptive_run_identifier} for single split on '{DATASET_CONFIG}' data ---")
        
        X_train_transformed, X_test_transformed = None, None
        vectorizer = None
        
        if feature_config['method'] == 'sklearn_vectorizer':
            vectorizer_path_single = os.path.join(
                single_run_feature_models_dir, 
                f"{feature_config['class'].__name__}_{sanitized_run_id_for_filename}_{DATASET_CONFIG}.joblib"
            )
            if LOAD_TRAINED_FEATURE_MODELS_IF_EXIST and os.path.exists(vectorizer_path_single):
                print(f"Loading pre-trained Sklearn Vectorizer for single run from {vectorizer_path_single}...")
                vectorizer = joblib.load(vectorizer_path_single)
                print("Vectorizer loaded.")
            else:
                print(f"Training Sklearn Vectorizer ({feature_config['class'].__name__}) for single run...")
                vectorizer = feature_config['class'](**feature_config['params'])
                vectorizer.fit(X_train_texts)
                print("Vectorizer trained.")
                if SAVE_TRAINED_FEATURE_MODELS:
                    print(f"Saving Vectorizer for single run to {vectorizer_path_single}...")
                    joblib.dump(vectorizer, vectorizer_path_single)
                    print("Vectorizer saved.")
            
            X_train_transformed = vectorizer.transform(X_train_texts)
            X_test_transformed = vectorizer.transform(X_test_texts)
        else:
            raise ValueError(f"Unsupported feature_config method for this script: {feature_config['method']}")

        if X_train_transformed is not None and X_test_transformed is not None:
            run_classification_and_report(
                X_train_transformed, X_test_transformed, y_train, y_test, 
                model_class=model_class,
                model_init_params=model_init_params,
                run_identifier_string=f"{descriptive_run_identifier} (Single Split - {DATASET_CONFIG})",
                labels=unique_labels_proc, 
                target_names=unique_labels_proc_str,
                output_dir=run_output_dir,
                random_state=RANDOM_STATE
            )
        else:
            print("Error: Feature transformation failed for single run.")

    print(f"\n{descriptive_run_identifier} script finished for {DATASET_CONFIG} dataset.")

if __name__ == "__main__":
    main_svm_tfidf()