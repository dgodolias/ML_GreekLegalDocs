import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
TEXT_COLUMN_TO_USE = 'summary'  # 'summary' or 'text'
# Using 'summary' as it's generally cleaner and less computationally intensive
# The assignment states: "κείμενα των αποφάσεων (ή των περιλήψεων τους)"

VECTORIZATION_METHOD = 'tfidf' # TF-IDF is a common choice for K-means
MAX_K_TO_TEST = 20          # Range of K values to test (e.g., 2 to 20)
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
    
    # Fill NaN in the text column to be used for vectorization
    df[TEXT_COLUMN_TO_USE] = df[TEXT_COLUMN_TO_USE].fillna('')
    
    # For NMI, case_category and case_tags are used. NaNs in these will be handled during NMI calculation.
    # Ensure case_tags are strings if they are to be used directly for NMI
    # If case_tags can be lists, they need to be converted to a hashable, comparable format (e.g., sorted tuple of strings)
    # Based on EDA, case_tags are initially objects (likely strings representing combined tags)
    # df['case_tags'] = df['case_tags'].apply(lambda x: tuple(sorted(x)) if isinstance(x, list) else x)
    # No, the EDA showed that the original string form of case_tags is what we need for NMI.
    # df.info() showed case_tags as object, and describe() showed unique string combinations.

    return df

def vectorize_texts(texts_series, method='tfidf'):
    """Vectorizes texts using the specified method."""
    print(f"\nVectorizing texts using {method} on '{TEXT_COLUMN_TO_USE}' column...")
    if method == 'tfidf':
        # Common TF-IDF parameters; can be tuned further
        vectorizer = TfidfVectorizer(
            max_df=0.90,  # Ignore terms that appear in more than 90% of documents
            min_df=5,     # Ignore terms that appear in less than 5 documents
            ngram_range=(1, 2),  # Consider unigrams and bigrams
            stop_words=None # No standard Greek stop words in sklearn; could add custom list
        )
        X = vectorizer.fit_transform(texts_series)
        print(f"TF-IDF matrix shape: {X.shape}")
        return X, vectorizer
    # Add other methods like Word2Vec, FastText, GloVe if needed, similar to B1
    # elif method == 'dense_embeddings':
    #     # Placeholder for dense embeddings logic
    #     print("Dense embeddings not implemented in this script version.")
    #     return None, None
    else:
        raise ValueError(f"Unsupported vectorization method: {method}")

def run_kmeans_and_evaluate(X, df, max_k):
    """Runs K-means for a range of K and calculates evaluation metrics."""
    print(f"\nRunning K-means for K from 2 to {max_k}...")
    
    k_values = range(2, max_k + 1)
    wcss = []  # Within-cluster sum of squares (for Elbow method)
    silhouette_scores_micro = []
    silhouette_scores_macro = []
    nmi_category_scores = []
    nmi_tags_scores = []

    # Prepare true labels for NMI
    true_labels_category_full = df['case_category']
    true_labels_tags_full = df['case_tags'] # Assumed to be strings representing tag combinations

    for k in k_values:
        print(f"  Processing K={k}...")
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=RANDOM_STATE)
        cluster_labels = kmeans.fit_predict(X)

        # 1. WCSS (Elbow Method)
        wcss.append(kmeans.inertia_)

        # 2. Silhouette Score
        # Micro Silhouette (Overall average silhouette score for each sample)
        s_micro = silhouette_score(X, cluster_labels, metric='euclidean')
        silhouette_scores_micro.append(s_micro)

        # Macro Silhouette (Average of the per-cluster average silhouette scores)
        sample_s_values = silhouette_samples(X, cluster_labels, metric='euclidean')
        cluster_avg_s_values = []
        for i in range(k): # For each cluster
            ith_cluster_s_values = sample_s_values[cluster_labels == i]
            if len(ith_cluster_s_values) > 0:
                cluster_avg_s_values.append(ith_cluster_s_values.mean())
        
        if cluster_avg_s_values:
            s_macro = np.mean(cluster_avg_s_values)
        else:
            s_macro = np.nan # Should not happen if k > 0 and X is not empty
        silhouette_scores_macro.append(s_macro)

        # 3. NMI Scores
        # Filter out NaNs from true labels and corresponding cluster_labels for NMI calculation
        
        # NMI with case_category
        valid_indices_category = true_labels_category_full.notna()
        if valid_indices_category.sum() >= 2: # NMI needs at least 2 samples with valid true labels
            nmi_cat = normalized_mutual_info_score(
                true_labels_category_full[valid_indices_category],
                cluster_labels[valid_indices_category]
            )
            nmi_category_scores.append(nmi_cat)
        else:
            nmi_category_scores.append(np.nan)

        # NMI with case_tags
        valid_indices_tags = true_labels_tags_full.notna()
        if valid_indices_tags.sum() >= 2:
            nmi_tag = normalized_mutual_info_score(
                true_labels_tags_full[valid_indices_tags],
                cluster_labels[valid_indices_tags]
            )
            nmi_tags_scores.append(nmi_tag)
        else:
            nmi_tags_scores.append(np.nan)
            
    return k_values, wcss, silhouette_scores_micro, silhouette_scores_macro, nmi_category_scores, nmi_tags_scores

def plot_evaluation_metrics(k_values, wcss, s_micro, s_macro, nmi_cat, nmi_tag):
    """Plots the evaluation metrics for choosing K."""
    print("\nPlotting evaluation metrics...")
    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
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

    # NMI with Case Category
    axs[1, 0].plot(k_values, nmi_cat, marker='D', linestyle='-', color='crimson')
    axs[1, 0].set_title('NMI (Clusters vs. Case Category) vs. K', fontsize=14)
    axs[1, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axs[1, 0].set_ylabel('NMI Score', fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # NMI with Case Tags
    axs[1, 1].plot(k_values, nmi_tag, marker='p', linestyle='-', color='purple')
    axs[1, 1].set_title('NMI (Clusters vs. Case Tags) vs. K', fontsize=14)
    axs[1, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axs[1, 1].set_ylabel('NMI Score', fontsize=12)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

    print("\n--- Comments on K Selection ---")
    print("Based on the plots, consider the following for choosing an optimal K:")
    print("1. Elbow Method (WCSS): Look for a 'knee' or 'elbow' point where the rate of WCSS decrease slows down significantly.")
    print("   A sharp bend suggests that adding more clusters beyond that point yields diminishing returns.")
    print("2. Silhouette Scores:")
    print("   - Micro Avg. Silhouette: Measures how similar an object is to its own cluster compared to other clusters (overall average). Higher is better.")
    print("   - Macro Avg. Silhouette: Average of the mean silhouette scores of samples within each cluster. Higher is better.")
    print("   Look for peaks in these scores. Values closer to 1 are ideal.")
    print("3. NMI Scores (Normalized Mutual Information):")
    print("   - NMI vs. Case Category: Measures the agreement between cluster assignments and predefined 'case_category' labels.")
    print("   - NMI vs. Case Tags: Measures the agreement between cluster assignments and predefined 'case_tags' labels.")
    print("   Higher NMI scores (closer to 1) indicate that the clusters align well with existing categorizations.")
    print("\nConsiderations:")
    print("- Often, there isn't a single 'perfect' K. You might need to balance insights from different metrics.")
    print("- If NMI scores are generally low, it might indicate that the natural groupings found by K-means (based on text similarity)")
    print("  do not perfectly align with the provided manual labels, which is common.")
    print("- The interpretability of the resulting clusters for a chosen K is also crucial for the subsequent step (B2.iii).")
    print("- A K that shows a good peak in Silhouette and reasonable NMI, while also being at or after an elbow in WCSS, is often a good candidate.")

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

    k_values, wcss, s_micro, s_macro, nmi_cat, nmi_tag = run_kmeans_and_evaluate(X, df, MAX_K_TO_TEST)
    
    plot_evaluation_metrics(k_values, wcss, s_micro, s_macro, nmi_cat, nmi_tag)

    print("\nAnalysis complete. Review the plots and comments to choose an optimal K for part B2.iii.")

if __name__ == '__main__':
    main()