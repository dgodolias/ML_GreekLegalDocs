import functools
import os
import time
import json
import numpy as np
from scipy.stats import t
from datasets import load_dataset
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split as sk_train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
import joblib

def script_execution_timer(func):
    """Decorator to time the execution of a script/function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        script_name = func.__name__
        start_time_sec = time.time()
        start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_sec))
        print(f"\n--- Script '{script_name}' started at: {start_time_readable} ---")
        result = func(*args, **kwargs)
        end_time_sec = time.time()
        end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time_sec))
        duration = end_time_sec - start_time_sec
        print(f"--- Script '{script_name}' finished at: {end_time_readable} ---")
        print(f"--- Total execution time for '{script_name}': {duration:.2f} seconds ---\n")
        return result
    return wrapper

def format_mean_report_to_string(mean_report_dict, report_class_labels_str, include_ci=False):
    """Formats the mean classification report dictionary into a string."""
    if include_ci:
        header = f"{'':<20}{'precision (±MoE) [CI]':>32}{'recall (±MoE) [CI]':>32}{'f1-score (±MoE) [CI]':>32}{'support (avg)':>15}\n\n"
    else:
        header = f"{'':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n\n"
    report_str = header
    for label in report_class_labels_str:
        if label in mean_report_dict:
            metrics = mean_report_dict[label]
            report_str += f"{label:<20}"
            if include_ci and 'precision' in metrics and isinstance(metrics['precision'], dict):
                for metric_name in ['precision', 'recall', 'f1-score']:
                    if metric_name in metrics and isinstance(metrics[metric_name], dict):
                        mean_val = metrics[metric_name].get('mean', 0)
                        moe_val = metrics[metric_name].get('moe', 0)
                        lower_bound = mean_val - moe_val
                        upper_bound = mean_val + moe_val
                        report_str += f"{mean_val:>8.4f} ± {moe_val:<6.4f} [{lower_bound:>6.4f},{upper_bound:>6.4f}]"
                    else:
                        report_str += f"{metrics.get(metric_name, 0):>8.4f} ± {'N/A':<6} [{'N/A':>6},{'N/A':>6}]"
            else:
                report_str += f"{metrics.get('precision', 0):>10.4f}"
                report_str += f"{metrics.get('recall', 0):>10.4f}"
                report_str += f"{metrics.get('f1-score', 0):>10.4f}"
            report_str += f"{metrics.get('support', {}).get('mean', 0):>15.2f}\n"
    report_str += "\n"
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in mean_report_dict:
            metrics = mean_report_dict[avg_type]
            report_str += f"{avg_type:<20}"
            if include_ci and 'precision' in metrics and isinstance(metrics['precision'], dict):
                for metric_name in ['precision', 'recall', 'f1-score']:
                    if metric_name in metrics and isinstance(metrics[metric_name], dict):
                        mean_val = metrics[metric_name].get('mean', 0)
                        moe_val = metrics[metric_name].get('moe', 0)
                        lower_bound = mean_val - moe_val
                        upper_bound = mean_val + moe_val
                        report_str += f"{mean_val:>8.4f} ± {moe_val:<6.4f} [{lower_bound:>6.4f},{upper_bound:>6.4f}]"
                    else:
                        report_str += f"{metrics.get(metric_name, 0):>8.4f} ± {'N/A':<6} [{'N/A':>6},{'N/A':>6}]"
            else:
                report_str += f"{metrics.get('precision', 0):>10.4f}"
                report_str += f"{metrics.get('recall', 0):>10.4f}"
                report_str += f"{metrics.get('f1-score', 0):>10.4f}"
            report_str += f"{metrics.get('support', {}).get('mean', 0):>15.2f}\n"
    return report_str

def load_and_preprocess_data(dataset_config, subset_percentage, random_state=42, min_samples_per_class=2):
    """Loads dataset, extracts text/labels, filters classes with < min_samples_per_class, and optionally creates a subset based on subset_percentage."""
    print(f"Loading dataset 'AI-team-UoA/greek_legal_code' with '{dataset_config}' configuration...")
    start_time = time.time()
    try:
        ds = load_dataset("AI-team-UoA/greek_legal_code", dataset_config, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset with config '{dataset_config}': {e}")
        return None, None, None, None
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")
    dataset_split = ds.get('train')
    if not dataset_split:
        print("Error: 'train' split not found in the dataset.")
        if len(ds.keys()) > 0:
            split_to_use = list(ds.keys())[0]; print(f"Attempting to use: '{split_to_use}'"); dataset_split = ds[split_to_use]
        else: print("No splits available."); return None, None, None, None
    print("Extracting text and labels...")
    start_time = time.time()
    if 'text' not in dataset_split.column_names or 'label' not in dataset_split.column_names:
        print(f"Error: 'text' or 'label' column not found. Available: {dataset_split.column_names}"); return None, None, None, None
    all_texts = dataset_split['text']
    all_labels = np.array(dataset_split['label'])
    print(f"Full dataset extracted in {time.time() - start_time:.2f}s. Samples: {len(all_texts)}")
    if len(all_texts) == 0: print("No data to process initially."); return None, None, None, None
    unique_labels_original, counts_original = np.unique(all_labels, return_counts=True)
    print(f"Original unique labels in full dataset ({dataset_config} config): {len(unique_labels_original)}")
    print(f"Filtering out classes with fewer than {min_samples_per_class} samples...")
    labels_to_keep = unique_labels_original[counts_original >= min_samples_per_class]
    if len(labels_to_keep) < len(unique_labels_original):
        num_filtered_out = len(unique_labels_original) - len(labels_to_keep)
        print(f"Filtered out {num_filtered_out} classes. Keeping {len(labels_to_keep)} classes.")
    else:
        print("No classes needed filtering based on min_samples_per_class.")
    mask = np.isin(all_labels, labels_to_keep)
    texts_filtered = [text for i, text in enumerate(all_texts) if mask[i]]
    labels_filtered = all_labels[mask]
    if len(texts_filtered) == 0:
        print(f"No data remaining after filtering classes with less than {min_samples_per_class} samples."); return None, None, None, None
    print(f"Dataset size after filtering for min_samples_per_class: {len(texts_filtered)} samples.")
    unique_labels_after_filtering = sorted(list(set(labels_filtered)))
    print(f"Unique labels after initial filtering: {len(unique_labels_after_filtering)}")
    texts_to_process, labels_to_process = texts_filtered, labels_filtered
    if 0 < subset_percentage < 1.0:
        print(f"\nSelecting a {subset_percentage*100:.0f}% subset from the filtered data...")
        unique_labels_for_subset_stratify, counts_for_subset_stratify = np.unique(labels_filtered, return_counts=True)
        can_stratify_subset = all(counts_for_subset_stratify >= 1)
        stratify_subset_labels = labels_filtered if len(unique_labels_for_subset_stratify) > 1 and can_stratify_subset else None
        if stratify_subset_labels is None and len(unique_labels_for_subset_stratify) > 1:
            print("Warning: Cannot stratify subset due to low class counts. Proceeding without stratification for subset.")
        try:
            texts_to_process, _, labels_to_process, _ = sk_train_test_split(
                texts_filtered, labels_filtered, train_size=subset_percentage, random_state=random_state, stratify=stratify_subset_labels)
        except ValueError as e:
            print(f"Error during stratified subset split: {e}. Attempting without stratification...")
            texts_to_process, _, labels_to_process, _ = sk_train_test_split(
                texts_filtered, labels_filtered, train_size=subset_percentage, random_state=random_state, stratify=None)
        print(f"Subset size: {len(texts_to_process)}")
        if len(texts_to_process) == 0: print("Error: Subset empty after selection."); return None, None, None, None
    elif subset_percentage >= 1.0:
        print(f"\nUsing 100% of the (filtered) data (subset_percentage={subset_percentage}).")
    else:
        print(f"\nWarning: subset_percentage is {subset_percentage}. Using 100% of the (filtered) data.")
    if not isinstance(labels_to_process, np.ndarray): labels_to_process = np.array(labels_to_process)
    unique_labels_to_process = sorted(list(set(labels_to_process)))
    unique_labels_to_process_str = [f"Class {l}" for l in unique_labels_to_process]
    print(f"Unique labels in data finally selected for processing: {len(unique_labels_to_process)}")
    return texts_to_process, labels_to_process, unique_labels_to_process, unique_labels_to_process_str

def generate_word2vec_doc_features(texts, w2v_model, vector_size):
    features = []
    for text in texts:
        tokens = text.lower().split()
        word_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if not word_vectors:
            features.append(np.zeros(vector_size))
        else:
            features.append(np.mean(word_vectors, axis=0))
    return np.array(features)

def run_experiment(
    texts_to_process, labels_to_process,
    unique_labels_to_process, unique_labels_to_process_str, # unique_labels_to_process are original unique labels
    n_splits,  # This is N_SPLITS_CV from main scripts
    feature_config,
    model_class, model_init_params,
    base_run_identifier, # e.g., "SVM_BoW_chapter" (used for filenames)
    reports_output_dir, # e.g., "outputs/SVM_BoW_chapter/reports"
    feature_models_output_dir, # e.g., "outputs/SVM_BoW_chapter/feature_models"
    random_state=42,
    save_trained_features=True,
    load_trained_features_if_exist=True,
    single_split_test_size=0.2 # Used when n_splits = 1
):
    """
    Runs a classification experiment, either as a single train/test split or K-Fold CV.
    Manages feature extraction, model training, evaluation, and report generation.
    """
    os.makedirs(reports_output_dir, exist_ok=True)
    os.makedirs(feature_models_output_dir, exist_ok=True)

    sanitized_base_run_id = "".join(c if c.isalnum() else "_" for c in base_run_identifier)

    # --- Label Encoding ---
    le = LabelEncoder()

    y_processed_encoded = le.fit_transform(labels_to_process)

    
    num_encoded_classes = len(le.classes_)

    report_labels_for_sklearn = list(range(num_encoded_classes))

    # --- SINGLE TRAIN/TEST SPLIT (n_splits = 1) ---
    if n_splits == 1:
        print(f"\n--- Performing single train/test split ({1-single_split_test_size:.0%}/{single_split_test_size:.0%}) for {base_run_identifier} ---")

        can_stratify_single_split = False
        if y_processed_encoded is not None and len(y_processed_encoded) > 0:
            unique_labels_single_encoded, counts_single_encoded = np.unique(y_processed_encoded, return_counts=True)
            if len(unique_labels_single_encoded) > 1 and all(counts_single_encoded >= 2): # Must have at least 2 samples per class for stratification
                can_stratify_single_split = True
            elif len(unique_labels_single_encoded) == 1:
                can_stratify_single_split = False 
            else: 
                print(f"Warning: Stratification for single split might not be possible or effective. Unique encoded classes: {len(unique_labels_single_encoded)}, counts: {counts_single_encoded}. Proceeding without stratification if necessary.")
        
        try:
            X_train_texts, X_test_texts, y_train, y_test = sk_train_test_split(
                texts_to_process, y_processed_encoded, test_size=single_split_test_size, # Use y_processed_encoded
                random_state=random_state, stratify=y_processed_encoded if can_stratify_single_split else None # Use y_processed_encoded for stratify
            )
        except ValueError as e:
            print(f"Error during stratified split: {e}. Attempting split without stratification.")
            X_train_texts, X_test_texts, y_train, y_test = sk_train_test_split(
                texts_to_process, y_processed_encoded, test_size=single_split_test_size, # Use y_processed_encoded
                random_state=random_state, stratify=None
            )
        # y_train and y_test are now encoded (0 to N-1)

        print(f"Data split. Train texts: {len(X_train_texts)}, Test texts: {len(X_test_texts)}")
        if not X_train_texts or len(X_train_texts) == 0:
            print("Error: Training set empty after split. Aborting single run."); return False

        # Feature Engineering
        feature_extractor = None
        feature_model_filename = f"{feature_config.get('method', 'feature_extractor')}_{sanitized_base_run_id}_single_run.model"
        if feature_config['method'] == 'sklearn_vectorizer':
            feature_model_filename = f"{feature_config['class'].__name__}_{sanitized_base_run_id}_single_run.joblib"
        
        feature_model_path = os.path.join(feature_models_output_dir, feature_model_filename)

        X_train_transformed, X_test_transformed = None, None

        if feature_config['method'] == 'sklearn_vectorizer':
            if load_trained_features_if_exist and os.path.exists(feature_model_path):
                print(f"Loading pre-trained Sklearn Vectorizer from {feature_model_path}...")
                feature_extractor = joblib.load(feature_model_path)
            else:
                print(f"Training Sklearn Vectorizer ({feature_config['class'].__name__})...")
                feature_extractor = feature_config['class'](**feature_config['params'])
                feature_extractor.fit(X_train_texts)
                if save_trained_features:
                    print(f"Saving Vectorizer to {feature_model_path}...")
                    joblib.dump(feature_extractor, feature_model_path)
            X_train_transformed = feature_extractor.transform(X_train_texts)
            X_test_transformed = feature_extractor.transform(X_test_texts)

        elif feature_config['method'] == 'word2vec':
            w2v_params = feature_config['params'].copy()
            w2v_params['seed'] = random_state # Ensure reproducibility for W2V training
            vector_size = w2v_params.get('vector_size', 100)
            if load_trained_features_if_exist and os.path.exists(feature_model_path):
                print(f"Loading pre-trained Word2Vec model from {feature_model_path}...")
                feature_extractor = Word2Vec.load(feature_model_path)
            else:
                print("Training Word2Vec model...")
                tokenized_train_texts = [text.lower().split() for text in X_train_texts]
                feature_extractor = Word2Vec(sentences=tokenized_train_texts, **w2v_params)
                if save_trained_features:
                    print(f"Saving Word2Vec model to {feature_model_path}...")
                    feature_extractor.save(feature_model_path)
            X_train_transformed = generate_word2vec_doc_features(X_train_texts, feature_extractor, vector_size)
            X_test_transformed = generate_word2vec_doc_features(X_test_texts, feature_extractor, vector_size)
        else:
            raise ValueError(f"Unsupported feature_config method: {feature_config['method']}")

        # Model Training
        print(f"Training {model_class.__name__} classifier...")
        current_model_params = model_init_params.copy()
        if 'random_state' not in current_model_params:
            try: 
                _ = model_class(random_state=random_state) # Check if model accepts random_state
                current_model_params['random_state'] = random_state
            except TypeError: 
                pass # Model might not accept random_state
        
        # If model is XGBoost, ensure num_class is set (though it should infer correctly with encoded labels)
        if "XGBClassifier" in str(model_class): # Check if it's an XGBoost model
            current_model_params['num_class'] = num_encoded_classes

        classifier = model_class(**current_model_params)
        classifier.fit(X_train_transformed, y_train) # y_train is encoded

        # Prediction and Evaluation
        print("Making predictions on the test set...")
        y_pred = classifier.predict(X_test_transformed) # y_pred will be encoded
        accuracy = accuracy_score(y_test, y_pred) # y_test is encoded
        print(f"Accuracy on the test set: {accuracy:.4f}")

        # Use report_labels_for_sklearn for 'labels' and unique_labels_to_process_str for 'target_names'
        report_str = classification_report(y_test, y_pred, labels=report_labels_for_sklearn, target_names=unique_labels_to_process_str, zero_division=0)
        report_dict = classification_report(y_test, y_pred, labels=report_labels_for_sklearn, target_names=unique_labels_to_process_str, output_dict=True, zero_division=0)

        report_title = f"Classification Report for {base_run_identifier} (Single Run)"
        print(f"\n{report_title}:\n{report_str}")

        report_txt_path = os.path.join(reports_output_dir, f"classification_report_{sanitized_base_run_id}_single_run.txt")
        with open(report_txt_path, "w", encoding='utf-8') as f:
            f.write(f"{report_title}\nAccuracy: {accuracy:.4f}\n\n{report_str}")
        print(f"Single run classification report (TXT) saved to {report_txt_path}")

        report_json_path = os.path.join(reports_output_dir, f"classification_report_{sanitized_base_run_id}_single_run.json")
        with open(report_json_path, "w", encoding='utf-8') as f:
            json.dump(report_dict, f, indent=4)
        print(f"Single run classification report (JSON) saved to {report_json_path}")
        return True

    # --- K-FOLD CROSS-VALIDATION (n_splits > 1) ---
    elif n_splits > 1:
        print(f"\n--- Performing {n_splits}-Fold Cross-Validation for {base_run_identifier} ---")

        if len(texts_to_process) < n_splits: # Check based on number of text samples
            print(f"Error: Samples ({len(texts_to_process)}) < N_SPLITS ({n_splits}). Cannot perform K-Fold CV."); return False

        min_samples_per_class_for_kfold = n_splits
        class_counts_encoded = np.unique(y_processed_encoded, return_counts=True)[1] # Use encoded labels for counts

        if np.any(class_counts_encoded < min_samples_per_class_for_kfold):
            problematic_encoded_indices = np.where(class_counts_encoded < min_samples_per_class_for_kfold)[0]
            # Map encoded indices back to original labels for user-friendly warning
            problematic_original_labels = le.classes_[problematic_encoded_indices] 
            
            print(f"Critical Warning: Some classes have fewer than {min_samples_per_class_for_kfold} samples (required for n_splits={n_splits} in StratifiedKFold).")
            print(f"Problematic classes (original label: count after encoding):")
            for i, orig_label in enumerate(problematic_original_labels):
                 print(f"  Class {orig_label}: {class_counts_encoded[problematic_encoded_indices[i]]} samples")
            print("StratifiedKFold will likely fail. Consider reducing n_splits, or further data filtering."); return False

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_accuracies = []
        all_fold_report_dicts = []
        texts_to_process_np = np.array(texts_to_process, dtype=object) 

        degrees_freedom = n_splits - 1
        alpha_ci = 0.05
        t_critical = t.ppf(1 - alpha_ci / 2, degrees_freedom) if degrees_freedom > 0 else 0
        if t_critical == 0: print("Warning: N_SPLITS <= 1 or invalid for CI calculation, CI for metrics will not be calculated.")

        feature_models_fold_dir = os.path.join(feature_models_output_dir, "cv_fold_models")
        if save_trained_features or load_trained_features_if_exist:
            os.makedirs(feature_models_fold_dir, exist_ok=True)

        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(texts_to_process_np, y_processed_encoded)): # Use y_processed_encoded for split
            fold_num = fold_idx + 1
            print(f"\nProcessing Fold {fold_num}/{n_splits}...")
            start_fold_time = time.time()

            X_train_fold_texts, X_val_fold_texts = texts_to_process_np[train_indices], texts_to_process_np[val_indices]
            # y_train_fold and y_val_fold will be slices of y_processed_encoded, thus already encoded
            y_train_fold, y_val_fold = y_processed_encoded[train_indices], y_processed_encoded[val_indices]

            feature_extractor_fold = None
            feature_model_fold_filename = f"{feature_config.get('method', 'feature_extractor')}_{sanitized_base_run_id}_fold_{fold_num}.model"
            if feature_config['method'] == 'sklearn_vectorizer':
                 feature_model_fold_filename = f"{feature_config['class'].__name__}_{sanitized_base_run_id}_fold_{fold_num}.joblib"
            
            feature_model_fold_path = os.path.join(feature_models_fold_dir, feature_model_fold_filename)
            
            X_train_fold_transformed, X_val_fold_transformed = None, None

            if feature_config['method'] == 'sklearn_vectorizer':
                if load_trained_features_if_exist and os.path.exists(feature_model_fold_path):
                    print(f"Loading pre-trained Sklearn Vectorizer for fold {fold_num} from {feature_model_fold_path}...")
                    feature_extractor_fold = joblib.load(feature_model_fold_path)
                else:
                    print(f"Training Sklearn Vectorizer ({feature_config['class'].__name__}) for fold {fold_num}...")
                    feature_extractor_fold = feature_config['class'](**feature_config['params'])
                    feature_extractor_fold.fit(X_train_fold_texts)
                    if save_trained_features:
                        print(f"Saving Vectorizer for fold {fold_num} to {feature_model_fold_path}...")
                        joblib.dump(feature_extractor_fold, feature_model_fold_path)
                X_train_fold_transformed = feature_extractor_fold.transform(X_train_fold_texts)
                X_val_fold_transformed = feature_extractor_fold.transform(X_val_fold_texts)

            elif feature_config['method'] == 'word2vec':
                w2v_params_fold = feature_config['params'].copy()
                w2v_params_fold['seed'] = random_state + fold_idx # Vary seed per fold for W2V
                vector_size_fold = w2v_params_fold.get('vector_size', 100)
                if load_trained_features_if_exist and os.path.exists(feature_model_fold_path):
                    print(f"Loading pre-trained Word2Vec model for fold {fold_num} from {feature_model_fold_path}...")
                    feature_extractor_fold = Word2Vec.load(feature_model_fold_path)
                else:
                    print(f"Training Word2Vec model for fold {fold_num}...")
                    tokenized_train_fold_texts = [text.lower().split() for text in X_train_fold_texts]
                    feature_extractor_fold = Word2Vec(sentences=tokenized_train_fold_texts, **w2v_params_fold)
                    if save_trained_features:
                        print(f"Saving Word2Vec model for fold {fold_num} to {feature_model_fold_path}...")
                        feature_extractor_fold.save(feature_model_fold_path)
                X_train_fold_transformed = generate_word2vec_doc_features(X_train_fold_texts, feature_extractor_fold, vector_size_fold)
                X_val_fold_transformed = generate_word2vec_doc_features(X_val_fold_texts, feature_extractor_fold, vector_size_fold)
            
            current_model_params_fold = model_init_params.copy()
            if 'random_state' not in current_model_params_fold:
                try: 
                    _ = model_class(random_state=random_state)
                    current_model_params_fold['random_state'] = random_state
                except TypeError: 
                    pass
            
            if "XGBClassifier" in str(model_class): # Check if it's an XGBoost model
                current_model_params_fold['num_class'] = num_encoded_classes
                # current_model_params_fold.pop('use_label_encoder', None)


            fold_model = model_class(**current_model_params_fold)
            fold_model.fit(X_train_fold_transformed, y_train_fold) # y_train_fold is encoded
            y_pred_fold = fold_model.predict(X_val_fold_transformed) # y_pred_fold is encoded

            fold_accuracy = accuracy_score(y_val_fold, y_pred_fold) # y_val_fold is encoded
            fold_accuracies.append(fold_accuracy)
            print(f"Fold {fold_num} Accuracy: {fold_accuracy:.4f}")

            report_str_fold = classification_report(y_val_fold, y_pred_fold, labels=report_labels_for_sklearn, target_names=unique_labels_to_process_str, zero_division=0)
            report_dict_fold = classification_report(y_val_fold, y_pred_fold, labels=report_labels_for_sklearn, target_names=unique_labels_to_process_str, output_dict=True, zero_division=0)
            all_fold_report_dicts.append(report_dict_fold)

            fold_report_path = os.path.join(reports_output_dir, f"fold_{fold_num}_report_{sanitized_base_run_id}.txt")
            with open(fold_report_path, "w", encoding='utf-8') as f:
                f.write(f"Classification Report for Fold {fold_num} ({base_run_identifier})\nAccuracy: {fold_accuracy:.4f}\n\n{report_str_fold}")
            print(f"Fold {fold_num} report saved. Processed in {time.time() - start_fold_time:.2f}s.")

        # Mean Report Calculation
        mean_report_metrics = {}
        # unique_labels_to_process_str are the keys in report_dict_fold (from target_names)
        report_keys_for_mean_calc = unique_labels_to_process_str + ['macro avg', 'weighted avg'] 
        for key in report_keys_for_mean_calc:
            if all_fold_report_dicts and key in all_fold_report_dicts[0]: # Check if key exists in the first fold's report
                mean_report_metrics[key] = {}
                for metric_name in ['precision', 'recall', 'f1-score']:
                    # Ensure the metric exists for the key in the first fold's report before list comprehension
                    if key in all_fold_report_dicts[0] and metric_name in all_fold_report_dicts[0][key]:
                        metric_values = [fr[key][metric_name] for fr in all_fold_report_dicts if key in fr and metric_name in fr[key]]
                        if not metric_values: continue # Should not happen if first check passes, but good for safety
                        mean_val, std_val = np.mean(metric_values), np.std(metric_values)
                        sem_val, moe_val = 0, 0
                        if n_splits > 1 and t_critical > 0: # degrees_freedom > 0 implies n_splits > 1
                            sem_val = std_val / np.sqrt(n_splits)
                            moe_val = t_critical * sem_val
                        mean_report_metrics[key][metric_name] = {'mean': mean_val, 'moe': moe_val}
                # Ensure 'support' exists for the key in the first fold's report
                if key in all_fold_report_dicts[0] and 'support' in all_fold_report_dicts[0][key]:
                    support_values = [fr[key]['support'] for fr in all_fold_report_dicts if key in fr and 'support' in fr[key]]
                    if support_values: mean_report_metrics[key]['support'] = {'mean': np.mean(support_values)}
        
        formatted_mean_report_str = format_mean_report_to_string(mean_report_metrics, unique_labels_to_process_str, include_ci=(t_critical > 0))
        mean_report_title = f"Mean Classification Report ({n_splits}-Folds, {base_run_identifier})"
        
        print(f"\n{mean_report_title}:\n{formatted_mean_report_str}")

        mean_report_txt_path = os.path.join(reports_output_dir, f"mean_classification_report_{sanitized_base_run_id}.txt")
        with open(mean_report_txt_path, "w", encoding='utf-8') as f:
            f.write(f"{mean_report_title}\n\n{formatted_mean_report_str}")
        print(f"Mean classification report (TXT) with CIs saved to {mean_report_txt_path}")

        mean_report_json_path = os.path.join(reports_output_dir, f"mean_classification_report_{sanitized_base_run_id}.json")
        with open(mean_report_json_path, "w", encoding='utf-8') as f:
            json.dump(mean_report_metrics, f, indent=4)
        print(f"Mean classification report (JSON) saved to {mean_report_json_path}")
        return True
    else: # n_splits < 1
        print(f"Error: n_splits must be 1 for a single run or > 1 for K-Fold CV. Got {n_splits}.")
        return False
