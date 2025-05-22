import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sklearn.linear_model import LogisticRegression
from utils import (
    load_and_preprocess_data,
    run_experiment,
    script_execution_timer
)
# Note: generate_word2vec_doc_features is now called within run_experiment in utils.py
# from gensim.models import Word2Vec # Word2Vec is imported in utils.py

# --- Configuration ---
DATASET_CONFIG = "subject" # available options: "volume", "chapter", "subject"
SUBSET_PERCENTAGE = 1.0 # 1.0 for full data, < 1.0 for subset
N_SPLITS_CV = 1 # 1 for single train/test split, > 1 for K-Fold CV
RANDOM_STATE = 42
SINGLE_SPLIT_TEST_SIZE = 0.2 # Used when N_SPLITS_CV = 1

# Feature Engineering Saving/Loading
SAVE_TRAINED_FEATURE_MODELS = True
LOAD_TRAINED_FEATURE_MODELS_IF_EXIST = True

# Word2Vec specific parameters (used by feature_config)
WORD2VEC_VECTOR_SIZE = 100
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 2
WORD2VEC_WORKERS = 4 # Adjust based on your CPU cores
WORD2VEC_SG = 0 # 0 for CBOW, 1 for Skip-gram
WORD2VEC_EPOCHS = 10
# --- End Configuration ---

@script_execution_timer
def main_logregr_word2vec():
    model_name_script = "LogRegr"
    feature_method_name = "Word2Vec"
    
    base_run_id = f"{model_name_script}_{feature_method_name}_{DATASET_CONFIG}"
    
    print(f"Starting {model_name_script} with {feature_method_name} script for '{DATASET_CONFIG}' config...")
    
    feature_config = {
        'method': 'word2vec',
        'params': {
            'vector_size': WORD2VEC_VECTOR_SIZE,
            'window': WORD2VEC_WINDOW,
            'min_count': WORD2VEC_MIN_COUNT,
            'workers': WORD2VEC_WORKERS,
            'sg': WORD2VEC_SG,
            'epochs': WORD2VEC_EPOCHS
            # 'seed' will be handled by run_experiment for reproducibility
        }
    }
    
    model_class = LogisticRegression
    model_init_params = {'solver': 'liblinear', 'max_iter': 1000, 'random_state': RANDOM_STATE}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_output_base_dir = os.path.join(script_dir, "outputs", base_run_id)
    reports_output_dir = os.path.join(experiment_output_base_dir, "reports")
    feature_models_output_dir = os.path.join(experiment_output_base_dir, "feature_models")

    os.makedirs(reports_output_dir, exist_ok=True)
    os.makedirs(feature_models_output_dir, exist_ok=True)
    print(f"INFO: Experiment outputs will be saved in '{experiment_output_base_dir}'")

    if 0 < SUBSET_PERCENTAGE < 1.0:
        print(f"INFO: Using a {SUBSET_PERCENTAGE*100:.0f}% subset of the data.")
    else:
        print("INFO: Using all available data (after filtering).")
    
    if N_SPLITS_CV == 1:
        print(f"INFO: Performing a single train/test split (test_size={SINGLE_SPLIT_TEST_SIZE}).")
    elif N_SPLITS_CV > 1:
        print(f"INFO: Performing {N_SPLITS_CV}-Fold Cross-Validation.")
    else:
        print("ERROR: N_SPLITS_CV must be >= 1.")
        return

    print(f"INFO: Saving of trained feature models is {'ENABLED' if SAVE_TRAINED_FEATURE_MODELS else 'DISABLED'}.")
    print(f"INFO: Loading of existing feature models is {'ENABLED' if LOAD_TRAINED_FEATURE_MODELS_IF_EXIST else 'DISABLED'}.")

    texts_proc, labels_proc, unique_labels_proc, unique_labels_proc_str = load_and_preprocess_data(
        DATASET_CONFIG, SUBSET_PERCENTAGE, RANDOM_STATE
    )

    if texts_proc is None or len(texts_proc) == 0:
        print("Failed to load or preprocess data. Exiting.")
        return

    success = run_experiment(
        texts_to_process=texts_proc,
        labels_to_process=labels_proc,
        unique_labels_to_process=unique_labels_proc,
        unique_labels_to_process_str=unique_labels_proc_str,
        n_splits=N_SPLITS_CV,
        feature_config=feature_config,
        model_class=model_class,
        model_init_params=model_init_params,
        base_run_identifier=base_run_id,
        reports_output_dir=reports_output_dir,
        feature_models_output_dir=feature_models_output_dir,
        random_state=RANDOM_STATE,
        save_trained_features=SAVE_TRAINED_FEATURE_MODELS,
        load_trained_features_if_exist=LOAD_TRAINED_FEATURE_MODELS_IF_EXIST,
        single_split_test_size=SINGLE_SPLIT_TEST_SIZE
    )

    if success:
        print(f"\nExperiment '{base_run_id}' completed successfully.")
    else:
        print(f"\nExperiment '{base_run_id}' encountered errors.")

    print(f"\n{model_name_script} with {feature_method_name} script finished for {DATASET_CONFIG} dataset.")

if __name__ == "__main__":
    main_logregr_word2vec()