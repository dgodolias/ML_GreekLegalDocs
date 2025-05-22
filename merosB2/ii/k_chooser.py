import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, normalized_mutual_info_score, adjusted_rand_score # Added adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
TEXT_COLUMN_TO_USE = 'summary'  # 'summary' or 'text'
# Using 'summary' as it's generally cleaner and less computationally intensive

VECTORIZATION_METHOD = 'tfidf' # TF-IDF is a common choice for K-means
MIN_K_TO_TEST = 2           # Minimum K value to test
MAX_K_TO_TEST = 202         # Range of K values to test (e.g., 2 to 100)
K_STEP = 5                  # Step for K values
RANDOM_STATE = 42           # For reproducibility

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

def run_kmeans_and_evaluate(X, df, min_k, max_k, k_step):
    """Runs K-means for a range of K and calculates evaluation metrics."""
    print(f"\nRunning K-means for K from {min_k} to {max_k} with a step of {k_step}...")
    
    k_values = list(range(min_k, max_k + 1, k_step))
    if not k_values: 
        if max_k >= min_k:
             k_values = [min_k] 
        else: 
            print(f"max_k ({max_k}) is less than min_k ({min_k}), K-means cannot be run.")
            return [], [], [], [], [], [], [], [] # Adjusted for ARI scores


    wcss = []
    silhouette_scores_micro = []
    silhouette_scores_macro = []
    nmi_category_scores = []
    nmi_tags_scores = []
    ari_category_scores = [] 
    ari_tags_scores = []   

    true_labels_category_full = df['case_category']
    true_labels_tags_full = df['case_tags'] 

    for k in k_values:
        print(f"  Processing K={k}...")
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=RANDOM_STATE)
        cluster_labels = kmeans.fit_predict(X)

        wcss.append(kmeans.inertia_)

        s_micro = silhouette_score(X, cluster_labels, metric='euclidean')
        silhouette_scores_micro.append(s_micro)

        sample_s_values = silhouette_samples(X, cluster_labels, metric='euclidean')
        cluster_avg_s_values = []
        for i in range(k): 
            ith_cluster_s_values = sample_s_values[cluster_labels == i]
            if len(ith_cluster_s_values) > 0:
                cluster_avg_s_values.append(ith_cluster_s_values.mean())
        
        if cluster_avg_s_values:
            s_macro = np.mean(cluster_avg_s_values)
        else:
            s_macro = np.nan 
        silhouette_scores_macro.append(s_macro)
        
        valid_indices_category = true_labels_category_full.notna()
        if valid_indices_category.sum() >= 2: 
            nmi_cat = normalized_mutual_info_score(
                true_labels_category_full[valid_indices_category],
                cluster_labels[valid_indices_category]
            )
            nmi_category_scores.append(nmi_cat)
            ari_cat = adjusted_rand_score( 
                true_labels_category_full[valid_indices_category],
                cluster_labels[valid_indices_category]
            )
            ari_category_scores.append(ari_cat)
        else:
            nmi_category_scores.append(np.nan)
            ari_category_scores.append(np.nan)

        valid_indices_tags = true_labels_tags_full.notna()
        if valid_indices_tags.sum() >= 2:
            nmi_tag = normalized_mutual_info_score(
                true_labels_tags_full[valid_indices_tags],
                cluster_labels[valid_indices_tags]
            )
            nmi_tags_scores.append(nmi_tag)
            ari_tag = adjusted_rand_score( 
                true_labels_tags_full[valid_indices_tags],
                cluster_labels[valid_indices_tags]
            )
            ari_tags_scores.append(ari_tag)
        else:
            nmi_tags_scores.append(np.nan)
            ari_tags_scores.append(np.nan)
            
    return k_values, wcss, silhouette_scores_micro, silhouette_scores_macro, nmi_category_scores, nmi_tags_scores, ari_category_scores, ari_tags_scores

def plot_evaluation_metrics(k_values, wcss, s_micro, s_macro, nmi_cat, nmi_tag, ari_cat, ari_tag):
    """Plots the evaluation metrics for choosing K."""
    if not k_values:
        print("No K values to plot. Skipping plotting.")
        return
        
    print("\nPlotting evaluation metrics...")
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('K-means Clustering Evaluation Metrics for Optimal K Selection', fontsize=16)

    # Elbow Method (WCSS)
    axs[0, 0].plot(k_values, wcss, marker='o', linestyle='-', color='dodgerblue')
    axs[0, 0].set_title('Elbow Method (WCSS vs. K)', fontsize=14)
    axs[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axs[0, 0].set_ylabel('WCSS (Inertia)', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Silhouette Scores
    axs[0, 1].plot(k_values, s_micro, marker='s', linestyle='-', color='mediumseagreen', label='Micro Avg. Silhouette')
    axs[0, 1].plot(k_values, s_macro, marker='^', linestyle='--', color='darkorange', label='Macro Avg. Silhouette (per cluster avg.)')
    axs[0, 1].set_title('Silhouette Scores vs. K', fontsize=14)
    axs[0, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axs[0, 1].set_ylabel('Silhouette Score', fontsize=12)
    axs[0, 1].legend(fontsize=10)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # NMI Scores (Combined on one plot)
    axs[1, 0].plot(k_values, nmi_cat, marker='D', linestyle='-', color='crimson', label='NMI vs. Case Category')
    axs[1, 0].plot(k_values, nmi_tag, marker='p', linestyle='--', color='purple', label='NMI vs. Case Tags')
    axs[1, 0].set_title('NMI Scores vs. K', fontsize=14)
    axs[1, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axs[1, 0].set_ylabel('NMI Score', fontsize=12)
    axs[1, 0].legend(fontsize=10)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # ARI Scores (Combined on one plot)
    axs[1, 1].plot(k_values, ari_cat, marker='o', linestyle='-', color='teal', label='ARI vs. Case Category')
    axs[1, 1].plot(k_values, ari_tag, marker='x', linestyle='--', color='sienna', label='ARI vs. Case Tags')
    axs[1, 1].set_title('ARI Scores vs. K', fontsize=14)
    axs[1, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axs[1, 1].set_ylabel('ARI Score', fontsize=12)
    axs[1, 1].legend(fontsize=10)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()


def main():
    """Main function to orchestrate the K-means analysis."""
    df = load_and_prepare_data()
    if df is None:
        print("Data loading failed. Exiting.")
        return

    X, vectorizer = vectorize_texts(df[TEXT_COLUMN_TO_USE], method=VECTORIZATION_METHOD)
    if X is None:
        print("Text vectorization failed. Exiting.")
        return

    k_values, wcss, s_micro, s_macro, nmi_cat, nmi_tag, ari_cat, ari_tag = run_kmeans_and_evaluate(X, df, MIN_K_TO_TEST, MAX_K_TO_TEST, K_STEP)
    
    if k_values: 
        plot_evaluation_metrics(k_values, wcss, s_micro, s_macro, nmi_cat, nmi_tag, ari_cat, ari_tag)
    else:
        print("\nAnalysis could not be completed as no valid K values were processed.")


if __name__ == '__main__':
    main()