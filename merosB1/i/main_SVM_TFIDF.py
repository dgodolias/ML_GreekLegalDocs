import sys
import os
import numpy as np 
import json 
import time
from scipy.stats import t 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer # Changed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from utils import run_svm_classification

# --- Configuration ---
DATASET_CONFIG = "chapter"  # Options: "volume", "chapter", "subject"
USE_SUBSET_FOR_CV_DATA = True
SUBSET_PERCENTAGE = 0.2
PERFORM_K_FOLD_CV = True
N_SPLITS_CV = 5
# --- End Configuration ---

def format_mean_report_to_string(mean_report_dict, report_class_labels_str, include_ci=False):
    """Formats the mean classification report dictionary into a string."""
    if include_ci:
        header = f"{'':<20}{'precision (±MoE)':>22}{'recall (±MoE)':>22}{'f1-score (±MoE)':>22}{'support (avg)':>15}\n\n"
    else: 
        header = f"{'':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n\n"
    
    report_str = header

    for label in report_class_labels_str: 
        if label in mean_report_dict:
            metrics = mean_report_dict[label]
            report_str += f"{label:<20}"
            if include_ci and 'precision' in metrics and isinstance(metrics['precision'], dict):
                report_str += f"{metrics['precision']['mean']:>8.4f} ± {metrics['precision']['moe']:<8.4f}"
                report_str += f"{metrics['recall']['mean']:>8.4f} ± {metrics['recall']['moe']:<8.4f}"
                report_str += f"{metrics['f1-score']['mean']:>8.4f} ± {metrics['f1-score']['moe']:<8.4f}"
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
                report_str += f"{metrics['precision']['mean']:>8.4f} ± {metrics['precision']['moe']:<8.4f}"
                report_str += f"{metrics['recall']['mean']:>8.4f} ± {metrics['recall']['moe']:<8.4f}"
                report_str += f"{metrics['f1-score']['mean']:>8.4f} ± {metrics['f1-score']['moe']:<8.4f}"
            else:
                report_str += f"{metrics.get('precision', 0):>10.4f}"
                report_str += f"{metrics.get('recall', 0):>10.4f}"
                report_str += f"{metrics.get('f1-score', 0):>10.4f}"

            report_str += f"{metrics.get('support', {}).get('mean', 0):>15.2f}\n"
    return report_str

def main_tfidf(): # Changed
    print("Starting TF-IDF SVM script...") # Changed
    feature_name = "TF-IDF" # Changed

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, "outputs")
    cv_output_dir = os.path.join(output_base_dir, f"tfidf_{DATASET_CONFIG}_cv_reports") # Changed
    
    if PERFORM_K_FOLD_CV:
        os.makedirs(cv_output_dir, exist_ok=True)
        print(f"INFO: Reports will be saved in '{cv_output_dir}'")
    else:
        single_run_output_dir = os.path.join(output_base_dir, f"tfidf_{DATASET_CONFIG}_single_run_reports")
        os.makedirs(single_run_output_dir, exist_ok=True)
        print(f"INFO: Single run reports will be saved in '{single_run_output_dir}'")


    print(f"INFO: Script configured to use a {SUBSET_PERCENTAGE*100:.0f}% subset of the data for processing." if USE_SUBSET_FOR_CV_DATA else "INFO: Script configured to use all available data for processing.")
    print(f"INFO: K-Fold Cross-Validation is ENABLED with {N_SPLITS_CV} splits." if PERFORM_K_FOLD_CV else "INFO: K-Fold Cross-Validation is DISABLED. Using single train/test split.")

    print(f"Loading dataset 'AI-team-UoA/greek_legal_code' with '{DATASET_CONFIG}' configuration...")
    start_time = time.time()
    try:
        ds = load_dataset("AI-team-UoA/greek_legal_code", DATASET_CONFIG, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset with config '{DATASET_CONFIG}': {e}"); return
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

    dataset_split = ds.get('train')
    if not dataset_split:
        print("Error: 'train' split not found in the dataset.")
        if len(ds.keys()) > 0:
            split_to_use = list(ds.keys())[0]; print(f"Attempting to use: '{split_to_use}'"); dataset_split = ds[split_to_use]
        else: print("No splits available."); return
    
    print("Extracting text and labels...")
    start_time = time.time()
    if 'text' not in dataset_split.column_names or 'label' not in dataset_split.column_names:
        print(f"Error: 'text' or 'label' column not found. Available: {dataset_split.column_names}"); return

    texts_full = dataset_split['text']
    labels_full = np.array(dataset_split['label'])
    print(f"Full dataset extracted in {time.time() - start_time:.2f}s. Samples: {len(texts_full)}")

    if len(texts_full) == 0: print("No data to process."); return
        
    unique_labels_full = sorted(list(set(labels_full)))
    print(f"Unique labels in full dataset ({DATASET_CONFIG} config): {len(unique_labels_full)}")

    texts_to_process, labels_to_process = texts_full, labels_full
    if USE_SUBSET_FOR_CV_DATA:
        print(f"\nSelecting a {SUBSET_PERCENTAGE*100:.0f}% subset...")
        try:
            texts_to_process, _, labels_to_process, _ = train_test_split(
                texts_full, labels_full, train_size=SUBSET_PERCENTAGE, random_state=42, 
                stratify=labels_full if len(unique_labels_full) > 1 else None)
        except ValueError as e:
            print(f"Error during stratified subset split: {e}")
            print("Attempting subset split without stratification...")
            texts_to_process, _, labels_to_process, _ = train_test_split(
                texts_full, labels_full, train_size=SUBSET_PERCENTAGE, random_state=42, 
                stratify=None)
        print(f"Subset size: {len(texts_to_process)}")
        if len(texts_to_process) == 0: print("Error: Subset empty."); return
    
    if not isinstance(labels_to_process, np.ndarray): labels_to_process = np.array(labels_to_process)
    
    unique_labels_to_process = sorted(list(set(labels_to_process)))
    unique_labels_to_process_str = [f"Class {l}" for l in unique_labels_to_process]
    print(f"Unique labels in data for processing: {len(unique_labels_to_process)}")

    if PERFORM_K_FOLD_CV and len(texts_to_process) < N_SPLITS_CV:
        print(f"Error: Samples ({len(texts_to_process)}) < N_SPLITS_CV ({N_SPLITS_CV}). Cannot perform K-Fold CV."); return

    min_samples_per_class_for_kfold = N_SPLITS_CV
    if PERFORM_K_FOLD_CV:
        class_counts = np.unique(labels_to_process, return_counts=True)[1]
        if np.any(class_counts < min_samples_per_class_for_kfold):
            print(f"Warning: Some classes in the data for processing have fewer than {min_samples_per_class_for_kfold} samples.")
            print("StratifiedKFold may fail or produce unreliable results for these classes.")


    if PERFORM_K_FOLD_CV:
        print(f"\n--- Performing {N_SPLITS_CV}-Fold Cross-Validation with {feature_name} on '{DATASET_CONFIG}' data ---")
        skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
        fold_accuracies = []
        all_fold_report_dicts = []
        texts_to_process_np = np.array(texts_to_process, dtype=object)

        degrees_freedom = N_SPLITS_CV - 1
        alpha_ci = 0.05 
        t_critical = 0
        if degrees_freedom > 0:
            t_critical = t.ppf(1 - alpha_ci / 2, degrees_freedom)
        else:
            print("Warning: Cannot calculate t_critical with N_SPLITS_CV <= 1.")

        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(texts_to_process_np, labels_to_process)):
            fold_num = fold_idx + 1
            print(f"\nProcessing Fold {fold_num}/{N_SPLITS_CV}...")
            start_fold_time = time.time()
            X_train_fold, X_val_fold = texts_to_process_np[train_indices], texts_to_process_np[val_indices]
            y_train_fold, y_val_fold = labels_to_process[train_indices], labels_to_process[val_indices]
            print(f"Train: {len(X_train_fold)}, Val: {len(X_val_fold)}")

            fold_vectorizer = TfidfVectorizer(max_features=5000) # Changed
            X_train_fold_transformed = fold_vectorizer.fit_transform(X_train_fold)
            X_val_fold_transformed = fold_vectorizer.transform(X_val_fold)
            
            fold_svm = SVC(kernel='linear', random_state=42)
            fold_svm.fit(X_train_fold_transformed, y_train_fold)
            y_pred_fold = fold_svm.predict(X_val_fold_transformed)
            fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)
            fold_accuracies.append(fold_accuracy)
            print(f"Fold {fold_num} Accuracy: {fold_accuracy:.4f}")

            report_str_fold = classification_report(
                y_val_fold, y_pred_fold, labels=unique_labels_to_process, 
                target_names=unique_labels_to_process_str, zero_division=0)
            report_dict_fold = classification_report(
                y_val_fold, y_pred_fold, labels=unique_labels_to_process, 
                target_names=unique_labels_to_process_str, output_dict=True, zero_division=0)
            all_fold_report_dicts.append(report_dict_fold)

            with open(os.path.join(cv_output_dir, f"fold_{fold_num}_report.txt"), "w", encoding='utf-8') as f:
                f.write(f"Classification Report for Fold {fold_num} ({feature_name})\nAccuracy: {fold_accuracy:.4f}\n\n{report_str_fold}")
            print(f"Fold {fold_num} report saved. Processed in {time.time() - start_fold_time:.2f}s.")

        mean_report_metrics = {}
        report_keys_for_mean_calc = unique_labels_to_process_str + ['macro avg', 'weighted avg']

        for key in report_keys_for_mean_calc:
            if all_fold_report_dicts and key in all_fold_report_dicts[0]: # Check if list is not empty
                mean_report_metrics[key] = {}
                for metric_name in ['precision', 'recall', 'f1-score']:
                    if metric_name in all_fold_report_dicts[0][key]:
                        metric_values = [fr[key][metric_name] for fr in all_fold_report_dicts if key in fr and metric_name in fr[key]]
                        if not metric_values: continue
                        mean_val = np.mean(metric_values)
                        std_val = np.std(metric_values)
                        sem_val = 0; moe_val = 0
                        if N_SPLITS_CV > 1 and t_critical > 0: # Ensure t_critical is valid
                            sem_val = std_val / np.sqrt(N_SPLITS_CV)
                            moe_val = t_critical * sem_val
                        mean_report_metrics[key][metric_name] = {'mean': mean_val, 'moe': moe_val}
                if 'support' in all_fold_report_dicts[0][key]:
                    support_values = [fr[key]['support'] for fr in all_fold_report_dicts if key in fr and 'support' in fr[key]]
                    if support_values:
                        mean_report_metrics[key]['support'] = {'mean': np.mean(support_values)}
        
        mean_report_json_path = os.path.join(cv_output_dir, "mean_classification_report.json")
        with open(mean_report_json_path, "w", encoding='utf-8') as f: json.dump(mean_report_metrics, f, indent=4)
        print(f"Mean classification report (JSON) saved to {mean_report_json_path}")

        formatted_mean_report_str = format_mean_report_to_string(mean_report_metrics, unique_labels_to_process_str, include_ci=(t_critical > 0))
        mean_report_str_path = os.path.join(cv_output_dir, "mean_classification_report.txt")
        with open(mean_report_str_path, "w", encoding='utf-8') as f:
            f.write(f"Mean Classification Report ({N_SPLITS_CV}-Folds, {feature_name})\n\n{formatted_mean_report_str}")
        print(f"Mean classification report (TXT) saved to {mean_report_str_path}")

        # --- Start of new/modified summary generation ---
        summary_str = f"\n--- K-Fold Cross-Validation Summary ({feature_name}) ---\n" # Changed title
        summary_str += f"Number of Folds: {N_SPLITS_CV}\n"

        # Calculate and add stats for Accuracy
        summary_str += f"\nAccuracy:\n"
        if fold_accuracies:
            mean_acc_val = np.mean(fold_accuracies)
            std_acc_val = np.std(fold_accuracies)
            summary_str += f"  Individual Fold Accuracies: {[round(val, 4) for val in fold_accuracies]}\n" # Changed label
            summary_str += f"  Mean Accuracy: {mean_acc_val:.4f}\n" # Changed label
            if N_SPLITS_CV > 1:
                summary_str += f"  Standard Deviation of Accuracy: {std_acc_val:.4f}\n" # Changed label
                if t_critical > 0:
                    sem_acc_val = std_acc_val / np.sqrt(N_SPLITS_CV)
                    moe_acc_val = t_critical * sem_acc_val
                    ci_acc_val = (mean_acc_val - moe_acc_val, mean_acc_val + moe_acc_val)
                    summary_str += f"  Standard Error of Mean (SEM) for Accuracy: {sem_acc_val:.4f}\n" # Changed label
                    summary_str += f"  95% Confidence Interval for Mean Accuracy: ({ci_acc_val[0]:.4f}, {ci_acc_val[1]:.4f})\n" # Changed label
                else:
                    summary_str += "  (SEM and CI for Accuracy not calculated as t_critical is not valid)\n"
            else:
                summary_str += "  (Std Dev, SEM, and CI for Accuracy not applicable for a single fold/split)\n"
        else:
            summary_str += "  Accuracy data not available (no fold accuracies).\n"
        
        summary_str += f"\nDetailed mean report with CIs for metrics saved to: {mean_report_str_path}\n"
        
        print(summary_str)
        with open(os.path.join(cv_output_dir, "cv_overall_summary.txt"), "w", encoding='utf-8') as f:
            f.write(summary_str)
        print(f"Overall CV summary saved to {os.path.join(cv_output_dir, 'cv_overall_summary.txt')}")
        # --- End of new/modified summary generation ---


    else: # Single train/test split
        print("\nSplitting data into a single training and testing set...")
        start_time_split = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            texts_to_process, labels_to_process, test_size=0.2, random_state=42,
            stratify=labels_to_process if len(set(labels_to_process)) > 1 else None)
        print(f"Data split in {time.time() - start_time_split:.2f}s. Train: {len(X_train)}, Test: {len(X_test)}")
        if not X_train or len(X_train) == 0: print("Error: Training set empty after split."); return

        print(f"\n--- Processing with {feature_name} for single split on '{DATASET_CONFIG}' data ---")
        vectorizer = TfidfVectorizer(max_features=5000) # Changed
        X_train_transformed = vectorizer.fit_transform(X_train)
        X_test_transformed = vectorizer.transform(X_test)
        
        run_svm_classification(X_train_transformed, X_test_transformed, y_train, y_test, 
                               feature_type=f"{feature_name} (Single Split - {DATASET_CONFIG})",
                               labels=unique_labels_to_process,
                               target_names=unique_labels_to_process_str,
                               output_dir=single_run_output_dir) # Pass output_dir


    print(f"\n{feature_name} SVM script finished for {DATASET_CONFIG} dataset.")

if __name__ == "__main__":
    main_tfidf() # Changed