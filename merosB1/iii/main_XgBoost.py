import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import xgboost as xgb # Import XGBoost
# Word2Vec and generate_word2vec_doc_features are handled within utils
from utils import (
    run_classification_and_report, 
    load_and_preprocess_data,
    perform_k_fold_cv_and_report,
    generate_word2vec_doc_features,
    script_execution_timer # For single run
)
from gensim.models import Word2Vec # For single run Word2Vec training

# --- Configuration ---
DATASET_CONFIG = "volume"  # available options: "volume", "chapter", "subject"
USE_SUBSET_FOR_CV_DATA = False 
SUBSET_PERCENTAGE = 0.2
PERFORM_K_FOLD_CV = True
N_SPLITS_CV = 5
RANDOM_STATE = 42
# Word2Vec specific parameters
WORD2VEC_VECTOR_SIZE = 100
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 2
WORD2VEC_WORKERS = 4 # Number of worker threads to train the model
WORD2VEC_SG = 0 # 0 for CBOW, 1 for Skip-gram
WORD2VEC_EPOCHS = 10 # Number of iterations (epochs) over the corpus
# XGBoost specific parameters
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 3
XGB_USE_LABEL_ENCODER = False # Suppress warning in newer XGBoost versions
XGB_EVAL_METRIC = 'mlogloss' # Evaluation metric for multiclass classification
# --- End Configuration ---

@script_execution_timer
def main_xgboost_word2vec():
    model_name_script = "XGBoost"
    feature_method_name = "Word2Vec"
    
    base_run_name = f"{model_name_script}_{feature_method_name}"
    
    print(f"Starting {model_name_script} with {feature_method_name} script...")
    
    feature_config = {
        'method': 'word2vec',
        'params': {
            'vector_size': WORD2VEC_VECTOR_SIZE,
            'window': WORD2VEC_WINDOW,
            'min_count': WORD2VEC_MIN_COUNT,
            'workers': WORD2VEC_WORKERS,
            'sg': WORD2VEC_SG,
            'epochs': WORD2VEC_EPOCHS
            # 'seed' will be set to RANDOM_STATE in utils for K-Fold Word2Vec training
        }
    }
    
    model_class = xgb.XGBClassifier
    model_init_params = {
        'n_estimators': XGB_N_ESTIMATORS,
        'learning_rate': XGB_LEARNING_RATE,
        'max_depth': XGB_MAX_DEPTH,
        'use_label_encoder': XGB_USE_LABEL_ENCODER,
        'eval_metric': XGB_EVAL_METRIC,
        # 'random_state' will be set by the utility functions if the model accepts it
    }

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

    descriptive_run_identifier = f"{model_class.__name__} with {feature_method_name}"

    if PERFORM_K_FOLD_CV:
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
        if not X_train_texts or len(X_train_texts) == 0: 
            print("Error: Training set empty after split.")
            return

        print(f"\n--- Processing with {descriptive_run_identifier} for single split on '{DATASET_CONFIG}' data ---")
        
        # Train Word2Vec model for single run
        print("Training Word2Vec model for single run...")
        tokenized_X_train_texts = [text.lower().split() for text in X_train_texts]
        w2v_single_run_params = feature_config['params'].copy()
        w2v_single_run_params['seed'] = RANDOM_STATE # for reproducibility
        
        w2v_model_single_run = Word2Vec(sentences=tokenized_X_train_texts, **w2v_single_run_params)
        print("Word2Vec model trained.")

        vector_size = w2v_single_run_params.get('vector_size', WORD2VEC_VECTOR_SIZE) # Use configured default
        X_train_transformed = generate_word2vec_doc_features(X_train_texts, w2v_model_single_run, vector_size)
        X_test_transformed = generate_word2vec_doc_features(X_test_texts, w2v_model_single_run, vector_size)
        
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

    print(f"\n{descriptive_run_identifier} script finished for {DATASET_CONFIG} dataset.")

if __name__ == "__main__":
    main_xgboost_word2vec()