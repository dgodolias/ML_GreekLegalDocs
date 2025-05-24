import sys
import os
import numpy as np
import math # Added for log2

# Adjust sys.path to include the parent directory (merosB1) to find utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # This should be ML_GreekLegalDocs

from utils import load_and_preprocess_data, script_execution_timer

DATASET_CONFIGS = ["volume", "chapter", "subject"]
SUBSET_PERCENTAGE = 1.0  # Use 1.0 for the full dataset
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 2 # As defined in load_and_preprocess_data and project description

@script_execution_timer
def main_check_balance():
    print(f"Checking class balance for dataset configurations: {DATASET_CONFIGS}\n")

    for config_name in DATASET_CONFIGS:
        print(f"--- Processing configuration: '{config_name}' ---")
        
        texts, labels, unique_labels_original_after_filtering, _ = load_and_preprocess_data(
            dataset_config=config_name,
            subset_percentage=SUBSET_PERCENTAGE,
            random_state=RANDOM_STATE,
            min_samples_per_class=MIN_SAMPLES_PER_CLASS 
        )

        if labels is None or len(labels) == 0:
            print(f"No labels found for configuration '{config_name}'. Skipping.\n")
            continue

        unique_classes, counts = np.unique(labels, return_counts=True)
        
        print(f"Target Variable: '{config_name}'")
        print(f"Total unique classes after initial filtering (min {MIN_SAMPLES_PER_CLASS} samples/class): {len(unique_classes)}")

        print("Class distribution:")

        
        sorted_indices = np.argsort(unique_classes)
        sorted_unique_classes_for_print = unique_classes[sorted_indices]
        sorted_counts_for_print = counts[sorted_indices]

        
        print(f"Total samples for '{config_name}' (after filtering): {len(labels)}")

        # Calculate imbalance metrics
        if len(counts) > 0:
            min_support = np.min(counts)
            max_support = np.max(counts)
            num_classes = len(counts)
            total_samples = np.sum(counts)

            print("\\nOverall Class Balance Metrics:")
            
            # Imbalance Ratio (IR)
            if min_support > 0:
                imbalance_ratio = max_support / min_support
                print(f"  Imbalance Ratio (Majority/Minority): {imbalance_ratio:.2f}")
                print(f"    - Min IR = 1.00: Perfect balance (all classes have equal support).")
                print(f"    - Max IR -> infinity: Extreme imbalance (majority class is vastly larger than minority).")
            else:
                print("  Imbalance Ratio: N/A (minority class has 0 samples)")

            # Coefficient of Variation (CV)
            if num_classes > 1: 
                mean_support = np.mean(counts)
                std_support = np.std(counts)
                if mean_support > 0:
                    cv_support = std_support / mean_support
                    print(f"  Coefficient of Variation of class sizes: {cv_support:.2f}")
                    print(f"    - Min CV = 0.00: Perfect balance (all classes have equal support, thus zero variance).")
                    print(f"    - Max CV (approx for N classes, one dominant): ~sqrt(N-1). Higher CV indicates greater disparity in class sizes.")
                else:
                    print("  Coefficient of Variation: N/A (mean support is 0)")
            else:
                print("  Coefficient of Variation: N/A (only one class, so no variation to measure)")

            # Entropy
            probabilities = counts / total_samples
            # Filter out zero probabilities to avoid log2(0)
            probabilities_nz = probabilities[probabilities > 0]
            if len(probabilities_nz) > 0:
                entropy = -np.sum(probabilities_nz * np.log2(probabilities_nz))
                print(f"  Entropy of class distribution: {entropy:.2f}")
                if num_classes > 0:
                    max_entropy = np.log2(num_classes) if num_classes > 1 else 0
                    print(f"    - Min Entropy = 0.00: Complete certainty/purity (all samples belong to a single class).")
                    print(f"    - Max Entropy = {max_entropy:.2f} (for {num_classes} classes): Complete uncertainty/uniformity (all classes are equally probable/represented, perfect balance).")
            else:
                print("  Entropy: N/A (no class probabilities > 0)")

        else:
            print("Imbalance Metrics: N/A (no classes with counts)")
            
        print("---\\n")

if __name__ == "__main__":
    main_check_balance()