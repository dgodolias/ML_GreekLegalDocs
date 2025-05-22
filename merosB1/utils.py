import functools
import os
import time
import json
import numpy as np
from scipy.stats import t
from datasets import load_dataset
from sklearn.model_selection import train_test_split as sk_train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression # Added
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec # Added for Word2Vec

def script_execution_timer(func):
    """Decorator to time the execution of a script/function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        script_name = func.__name__
        start_time_sec = time.time()
        start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_sec))
        
        print(f"\n--- Script '{script_name}' started at: {start_time_readable} ---")
        
        result = func(*args, **kwargs) # Execute the decorated function
        
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
                    if metric_name in metrics and isinstance(metrics[metric_name], dict): # Check if metric data exists and is dict
                        mean_val = metrics[metric_name].get('mean', 0)
                        moe_val = metrics[metric_name].get('moe', 0)
                        lower_bound = mean_val - moe_val
                        upper_bound = mean_val + moe_val
                        report_str += f"{mean_val:>8.4f} ± {moe_val:<6.4f} [{lower_bound:>6.4f},{upper_bound:>6.4f}]"
                    else:
                        report_str += f"{metrics.get(metric_name, 0):>8.4f} ± {'N/A':<6} [{'N/A':>6},{'N/A':>6}]" # Fallback for non-dict or missing
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
                    if metric_name in metrics and isinstance(metrics[metric_name], dict): # Check if metric data exists and is dict
                        mean_val = metrics[metric_name].get('mean', 0)
                        moe_val = metrics[metric_name].get('moe', 0)
                        lower_bound = mean_val - moe_val
                        upper_bound = mean_val + moe_val
                        report_str += f"{mean_val:>8.4f} ± {moe_val:<6.4f} [{lower_bound:>6.4f},{upper_bound:>6.4f}]"
                    else:
                        report_str += f"{metrics.get(metric_name, 0):>8.4f} ± {'N/A':<6} [{'N/A':>6},{'N/A':>6}]" # Fallback
            else:
                report_str += f"{metrics.get('precision', 0):>10.4f}"
                report_str += f"{metrics.get('recall', 0):>10.4f}"
                report_str += f"{metrics.get('f1-score', 0):>10.4f}"

            report_str += f"{metrics.get('support', {}).get('mean', 0):>15.2f}\n"
    return report_str

def load_and_preprocess_data(dataset_config, use_subset, subset_percentage, random_state=42):
    """Loads dataset, extracts text/labels, and optionally creates a subset."""
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

    texts_full = dataset_split['text']
    labels_full = np.array(dataset_split['label'])
    print(f"Full dataset extracted in {time.time() - start_time:.2f}s. Samples: {len(texts_full)}")

    if len(texts_full) == 0: print("No data to process."); return None, None, None, None
        
    unique_labels_full = sorted(list(set(labels_full)))
    print(f"Unique labels in full dataset ({dataset_config} config): {len(unique_labels_full)}")

    texts_to_process, labels_to_process = texts_full, labels_full
    if use_subset:
        print(f"\nSelecting a {subset_percentage*100:.0f}% subset...")
        try:
            texts_to_process, _, labels_to_process, _ = sk_train_test_split(
                texts_full, labels_full, train_size=subset_percentage, random_state=random_state, 
                stratify=labels_full if len(unique_labels_full) > 1 else None)
        except ValueError as e:
            print(f"Error during stratified subset split: {e}")
            print("Attempting subset split without stratification...")
            texts_to_process, _, labels_to_process, _ = sk_train_test_split(
                texts_full, labels_full, train_size=subset_percentage, random_state=random_state, stratify=None)
        print(f"Subset size: {len(texts_to_process)}")
        if len(texts_to_process) == 0: print("Error: Subset empty."); return None, None, None, None
    
    if not isinstance(labels_to_process, np.ndarray): labels_to_process = np.array(labels_to_process)
    
    unique_labels_to_process = sorted(list(set(labels_to_process)))
    unique_labels_to_process_str = [f"Class {l}" for l in unique_labels_to_process]
    print(f"Unique labels in data for processing: {len(unique_labels_to_process)}")
    
    return texts_to_process, labels_to_process, unique_labels_to_process, unique_labels_to_process_str

def generate_word2vec_doc_features(texts, w2v_model, vector_size):
    """Generates document features by averaging Word2Vec vectors of words in a document."""
    features = []
    for text in texts:
        # Simple tokenization, consider NLTK's word_tokenize for more complex cases
        tokens = text.lower().split() 
        word_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if not word_vectors:
            features.append(np.zeros(vector_size)) # Zero vector for documents with no known words
        else:
            features.append(np.mean(word_vectors, axis=0))
    return np.array(features)

def perform_k_fold_cv_and_report(
    texts_to_process, labels_to_process, 
    unique_labels_to_process, unique_labels_to_process_str,
    n_splits_cv, 
    feature_config, # Dictionary: {'method': 'sklearn_vectorizer'/'word2vec', 'class': ..., 'params': ...}
    model_class, model_init_params, # Classifier class and its initialization parameters
    run_identifier_string, # Descriptive string for this run, e.g., "LogRegr_Word2Vec (volume)"
    cv_output_dir, random_state=42
):
    """Performs K-Fold CV, generates and saves reports for a given model and feature method."""
    print(f"\n--- Performing {n_splits_cv}-Fold Cross-Validation with {run_identifier_string} ---")
    
    if len(texts_to_process) < n_splits_cv:
        print(f"Error: Samples ({len(texts_to_process)}) < N_SPLITS_CV ({n_splits_cv}). Cannot perform K-Fold CV."); return False

    min_samples_per_class_for_kfold = n_splits_cv
    class_counts = np.unique(labels_to_process, return_counts=True)[1]
    if np.any(class_counts < min_samples_per_class_for_kfold):
        print(f"Warning: Some classes have fewer than {min_samples_per_class_for_kfold} samples. StratifiedKFold may be affected.")

    skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
    fold_accuracies = []
    all_fold_report_dicts = []
    texts_to_process_np = np.array(texts_to_process, dtype=object)

    degrees_freedom = n_splits_cv - 1; alpha_ci = 0.05; t_critical = 0
    if degrees_freedom > 0: t_critical = t.ppf(1 - alpha_ci / 2, degrees_freedom)
    else: print("Warning: N_SPLITS_CV <= 1, CI for metrics will not be calculated.")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(texts_to_process_np, labels_to_process)):
        fold_num = fold_idx + 1; print(f"\nProcessing Fold {fold_num}/{n_splits_cv}...")
        start_fold_time = time.time()
        
        X_train_fold_texts, X_val_fold_texts = texts_to_process_np[train_indices], texts_to_process_np[val_indices]
        y_train_fold, y_val_fold = labels_to_process[train_indices], labels_to_process[val_indices]
        print(f"Train: {len(X_train_fold_texts)}, Val: {len(X_val_fold_texts)}")

        X_train_fold_transformed, X_val_fold_transformed = None, None

        if feature_config['method'] == 'sklearn_vectorizer':
            vectorizer = feature_config['class'](**feature_config['params'])
            X_train_fold_transformed = vectorizer.fit_transform(X_train_fold_texts)
            X_val_fold_transformed = vectorizer.transform(X_val_fold_texts)
        elif feature_config['method'] == 'word2vec':
            tokenized_train_texts = [text.lower().split() for text in X_train_fold_texts]
            w2v_train_params = feature_config['params'].copy() # Use a copy
            w2v_train_params['seed'] = random_state # for reproducibility of Word2Vec training
            
            print(f"Training Word2Vec model for fold {fold_num}...")
            w2v_model = Word2Vec(sentences=tokenized_train_texts, **w2v_train_params)
            print("Word2Vec model trained.")

            vector_size = w2v_train_params.get('vector_size', 100) # Ensure consistency
            X_train_fold_transformed = generate_word2vec_doc_features(X_train_fold_texts, w2v_model, vector_size)
            X_val_fold_transformed = generate_word2vec_doc_features(X_val_fold_texts, w2v_model, vector_size)
        else:
            raise ValueError(f"Unsupported feature_config method: {feature_config['method']}")
        
        current_model_params = model_init_params.copy()
        if 'random_state' not in current_model_params: # Add random_state if model supports it and not already set
             try: # Check if model can accept random_state
                test_model = model_class(random_state=random_state, **current_model_params)
                current_model_params['random_state'] = random_state
             except TypeError: pass # Model doesn't accept random_state or it's already set

        fold_model = model_class(**current_model_params)
        fold_model.fit(X_train_fold_transformed, y_train_fold)
        y_pred_fold = fold_model.predict(X_val_fold_transformed)
        
        fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)
        fold_accuracies.append(fold_accuracy); print(f"Fold {fold_num} Accuracy: {fold_accuracy:.4f}")

        report_str_fold = classification_report(y_val_fold, y_pred_fold, labels=unique_labels_to_process, target_names=unique_labels_to_process_str, zero_division=0)
        report_dict_fold = classification_report(y_val_fold, y_pred_fold, labels=unique_labels_to_process, target_names=unique_labels_to_process_str, output_dict=True, zero_division=0)
        all_fold_report_dicts.append(report_dict_fold)

        os.makedirs(cv_output_dir, exist_ok=True)
        with open(os.path.join(cv_output_dir, f"fold_{fold_num}_report.txt"), "w", encoding='utf-8') as f:
            f.write(f"Classification Report for Fold {fold_num} ({run_identifier_string})\nAccuracy: {fold_accuracy:.4f}\n\n{report_str_fold}")
        print(f"Fold {fold_num} report saved. Processed in {time.time() - start_fold_time:.2f}s.")

    mean_report_metrics = {}
    report_keys_for_mean_calc = unique_labels_to_process_str + ['macro avg', 'weighted avg']
    for key in report_keys_for_mean_calc:
        if all_fold_report_dicts and key in all_fold_report_dicts[0]:
            mean_report_metrics[key] = {}
            for metric_name in ['precision', 'recall', 'f1-score']:
                if metric_name in all_fold_report_dicts[0][key]:
                    metric_values = [fr[key][metric_name] for fr in all_fold_report_dicts if key in fr and metric_name in fr[key]]
                    if not metric_values: continue
                    mean_val, std_val = np.mean(metric_values), np.std(metric_values)
                    sem_val, moe_val = 0, 0
                    if n_splits_cv > 1 and t_critical > 0: sem_val = std_val / np.sqrt(n_splits_cv); moe_val = t_critical * sem_val
                    mean_report_metrics[key][metric_name] = {'mean': mean_val, 'moe': moe_val}
            if 'support' in all_fold_report_dicts[0][key]:
                support_values = [fr[key]['support'] for fr in all_fold_report_dicts if key in fr and 'support' in fr[key]]
                if support_values: mean_report_metrics[key]['support'] = {'mean': np.mean(support_values)}
    
    formatted_mean_report_str = format_mean_report_to_string(mean_report_metrics, unique_labels_to_process_str, include_ci=(t_critical > 0))
    mean_report_str_path = os.path.join(cv_output_dir, "mean_classification_report.txt")
    with open(mean_report_str_path, "w", encoding='utf-8') as f:
        f.write(f"Mean Classification Report ({n_splits_cv}-Folds, {run_identifier_string})\n\n{formatted_mean_report_str}")
    print(f"Mean classification report (TXT) with CIs saved to {mean_report_str_path}") 
    return True

def run_classification_and_report( # Renamed from run_svm_classification
    X_train_features, X_test_features, y_train, y_test, 
    model_class, model_init_params, # Classifier class and its initialization parameters
    run_identifier_string, # Descriptive string for this run, e.g., "LogRegr_Word2Vec (Single Split - volume)"
    labels=None, target_names=None, output_dir=None, random_state=42
):
    """Trains a given model, predicts, evaluates, and optionally saves reports."""
    print(f"\n--- Running Classification with {run_identifier_string} ---")
    print(f"Training {model_class.__name__} classifier...")
    start_time = time.time()
    
    current_model_params = model_init_params.copy()
    if 'random_state' not in current_model_params: # Add random_state if model supports it
        try:
            test_model = model_class(random_state=random_state, **current_model_params) # Check if model can accept random_state
            current_model_params['random_state'] = random_state
        except TypeError:
            pass # Model doesn't accept random_state or it's already set in model_init_params

    classifier = model_class(**current_model_params)
    classifier.fit(X_train_features, y_train)
    print(f"{model_class.__name__} classifier trained in {time.time() - start_time:.2f} seconds.")

    print("Making predictions on the test set...")
    start_time = time.time()
    y_pred = classifier.predict(X_test_features)
    print(f"Predictions made in {time.time() - start_time:.2f} seconds.")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy:.4f}")
    
    report_str, report_dict = "", {}
    try:
        report_str = classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0)
        print(f"\nClassification Report for {run_identifier_string}:\n{report_str}")
        if output_dir: report_dict = classification_report(y_test, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    except ValueError as e: print(f"Could not generate classification report: {e}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_run_identifier = "".join(c if c.isalnum() else "_" for c in run_identifier_string)
        report_txt_path = os.path.join(output_dir, f"classification_report_{safe_run_identifier}.txt")
        with open(report_txt_path, "w", encoding='utf-8') as f:
            f.write(f"Classification Report for {run_identifier_string}\nAccuracy: {accuracy:.4f}\n\n{report_str if report_str else 'Report could not be generated.'}")
        print(f"Single run classification report (TXT) saved to {report_txt_path}")
        if report_dict: 
            report_json_path = os.path.join(output_dir, f"classification_report_{safe_run_identifier}.json")
            with open(report_json_path, "w", encoding='utf-8') as f: json.dump(report_dict, f, indent=4)
            print(f"Single run classification report (JSON) saved to {report_json_path}")
    return accuracy