import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
import warnings
import os # Import os module for path manipulation

# --- Configuration ---
CHOSEN_K = 21  # <--- !!! SET YOUR CHOSEN K HERE !!! (e.g., based on k_chooser.py results)
TEXT_COLUMN_TO_USE = 'summary'  # 'summary' or 'text'
VECTORIZATION_METHOD = 'tfidf' # TF-IDF
RANDOM_STATE = 42           # For reproducibility
# Construct the output path relative to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(SCRIPT_DIR, '..', 'documents_with_clusters.csv') # File to save results

def load_and_prepare_data():
    """Loads the dataset and prepares it for analysis."""
    print("Loading the 'DominusTea/GreekLegalSum' dataset...")
    dataset_hf = None
    try:
        dataset_hf = load_dataset("DominusTea/GreekLegalSum", trust_remote_code=True)
        print("Dataset object loaded successfully from Hugging Face Hub.")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face Hub: {e}")
        return None

    df = None
    if dataset_hf:
        dataset_split = dataset_hf.get('train')
        if dataset_split:
            print("Using 'train' split.")
            df = dataset_split.to_pandas()
        else:
            print("'train' split not found. Trying the first available split.")
            available_splits = list(dataset_hf.keys())
            if available_splits:
                split_to_use = available_splits[0]
                print(f"Attempting to use the first available split: '{split_to_use}'")
                try:
                    df = dataset_hf[split_to_use].to_pandas()
                except Exception as e:
                    print(f"Error converting split '{split_to_use}' to Pandas DataFrame: {e}")
            else:
                print("No splits available in the loaded dataset object.")
    
    if df is None:
        print("DataFrame could not be created.")
        return None

    print(f"\nDataFrame created. Shape: {df.shape}")
    df[TEXT_COLUMN_TO_USE] = df[TEXT_COLUMN_TO_USE].fillna('')
    return df

def vectorize_texts(texts_series, method='tfidf'):
    """Vectorizes texts using the specified method."""
    print(f"\nVectorizing texts using {method} on '{TEXT_COLUMN_TO_USE}' column...")
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_df=0.90,
            min_df=5,
            ngram_range=(1, 2),
            stop_words=None
        )
        X = vectorizer.fit_transform(texts_series)
        print(f"TF-IDF matrix shape: {X.shape}")
        return X, vectorizer
    else:
        raise ValueError(f"Unsupported vectorization method: {method}")

def run_final_kmeans(X, df_input, chosen_k_value):
    """Runs K-means for the chosen K, evaluates, and adds labels to DataFrame."""
    df = df_input.copy() # Work on a copy to avoid modifying the original df passed to function
    print(f"\nRunning K-means for the chosen K = {chosen_k_value}...")
    
    kmeans_model = KMeans(n_clusters=chosen_k_value, init='k-means++', n_init='auto', random_state=RANDOM_STATE)
    cluster_labels = kmeans_model.fit_predict(X)

    # Add cluster labels to the DataFrame
    df['cluster_id'] = cluster_labels
    print(f"\nCluster labels added to DataFrame in column 'cluster_id'.")

    
    print("\n--- Cluster Sizes ---")
    cluster_sizes = df['cluster_id'].value_counts().sort_index()
    for cluster_id_val, size in cluster_sizes.items():
        print(f"  Cluster {cluster_id_val}: {size} documents")
        
    return df, kmeans_model

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    print(f"--- K-means Clustering for a Specific K (K={CHOSEN_K}) ---")
    
    df_original = load_and_prepare_data()
    if df_original is None:
        print("Data loading failed. Exiting.")
        return

    X, vectorizer = vectorize_texts(df_original[TEXT_COLUMN_TO_USE], method=VECTORIZATION_METHOD)
    if X is None:
        print("Text vectorization failed. Exiting.")
        return

    df_with_clusters, kmeans_model = run_final_kmeans(X, df_original, CHOSEN_K)

    # Save the DataFrame with cluster assignments
    try:
        # Ensure the directory exists if it's not the current one
        # os.path.normpath will resolve ".." components if OUTPUT_CSV_FILE is absolute
        output_path_normalized = os.path.normpath(OUTPUT_CSV_FILE)
        output_dir = os.path.dirname(output_path_normalized)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        df_with_clusters.to_csv(output_path_normalized, index=False, encoding='utf-8-sig')
        print(f"\nDataFrame with cluster assignments saved to '{os.path.abspath(output_path_normalized)}'")
    except Exception as e:
        print(f"\nError saving DataFrame to CSV: {e}")

    print(f"\nProcess complete. The DataFrame in '{os.path.abspath(output_path_normalized)}' now includes a 'cluster_id' column.")
    print(f"The centroids of the clusters can be accessed via 'kmeans_model.cluster_centers_'.")
if __name__ == '__main__':
    main()

