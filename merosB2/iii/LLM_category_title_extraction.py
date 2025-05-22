import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import re
import random
from collections import Counter

# --- Configuration (should match kmeans_final.py and your choices) ---
CHOSEN_K = 20  
TEXT_COLUMN_TO_USE = 'summary'
RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_FILE = os.path.join(SCRIPT_DIR, '..', 'documents_with_clusters.csv') # File to save results

# TF-IDF Parameters (must be identical to kmeans_final.py)
TFIDF_MAX_DF = 0.90
TFIDF_MIN_DF = 5
TFIDF_NGRAM_RANGE = (1, 2)

# LLM Prompt (single document)
LLM_SINGLE_DOC_PROMPT_TEMPLATE = (
    "Σου δίνεται ένα κείμενο νομικής απόφασης. Ποιο είναι το κεντρικό θέμα της απόφασης; "
    "Απάντησε στην μορφή: ‘Θέμα:\\n…’. Το κείμενο είναι αυτό:\\n{LEGAL_TEXT}"
)

def load_data(csv_path):
    """Loads data with cluster assignments."""
    print(f"Loading data from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        if 'cluster_id' not in df.columns:
            raise ValueError("CSV file must contain 'cluster_id' column.")
        if TEXT_COLUMN_TO_USE not in df.columns:
            raise ValueError(f"CSV file must contain '{TEXT_COLUMN_TO_USE}' column.")
        df[TEXT_COLUMN_TO_USE] = df[TEXT_COLUMN_TO_USE].fillna('')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'. Please ensure kmeans_final.py has run and produced this file.")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def refit_vectorizer_and_kmeans(df_full):
    """
    Re-fits TF-IDF and K-means to get access to vectors and centroids.
    This is done because the models might not have been saved by kmeans_final.py.
    Ensures that parameters are identical to those used in kmeans_final.py.
    """
    print("\nRe-fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words=None
    )
    X_full = vectorizer.fit_transform(df_full[TEXT_COLUMN_TO_USE])
    print(f"TF-IDF matrix re-fitted. Shape: {X_full.shape}")

    print(f"\nRe-fitting K-means model for K={CHOSEN_K}...")
    kmeans_model = KMeans(n_clusters=CHOSEN_K, init='k-means++', n_init='auto', random_state=RANDOM_STATE)
    kmeans_model.fit(X_full) # Fit on all data to get comparable centroids
    print("K-means model re-fitted.")
    
    return X_full, vectorizer, kmeans_model

def get_docs_closest_to_centroid(df_cluster_docs, X_cluster_docs, centroid_vector, n_docs=3):
    """Finds n_docs closest to the centroid in the given cluster."""
    if X_cluster_docs.shape[0] == 0:
        return pd.DataFrame()
    
    distances = euclidean_distances(X_cluster_docs, centroid_vector.reshape(1, -1))
    closest_indices_in_cluster = np.argsort(distances.ravel())[:n_docs]
    
    # df_cluster_docs contains documents only from that cluster.
    # The indices in closest_indices_in_cluster are relative to X_cluster_docs / df_cluster_docs.
    return df_cluster_docs.iloc[closest_indices_in_cluster]

def get_random_docs_from_cluster(df_cluster_docs, n_docs=3, random_state=None):
    """Selects n_docs randomly from the given cluster."""
    if df_cluster_docs.shape[0] == 0:
        return pd.DataFrame()
    if df_cluster_docs.shape[0] <= n_docs:
        return df_cluster_docs
    return df_cluster_docs.sample(n=n_docs, random_state=random_state)

def call_mock_llm(prompt_text: str) -> str:
    """
    MOCK LMM Call. Replace this with your actual LLM API call.
    This mock attempts to generate a somewhat relevant theme.
    """
    # print(f"    Mock LLM received prompt: {prompt_text[:100]}...") # For debugging
    theme_content = "Προκαθορισμένο θέμα από mock LLM"
    
    # Try to extract some keywords to make the mock response slightly more dynamic
    text_to_search = prompt_text.split("Το κείμενο είναι αυτό:\\n")[-1]
    keywords = re.findall(r'\b[Α-Ωα-ωίϊΐόάέύϋΰώΊΪΌΆΈΎΫΏ]{5,}\b', text_to_search) # Greek words >= 5 chars
    
    if "απάτη" in text_to_search.lower() or any("απάτ" in k.lower() for k in keywords):
        theme_content = "Περί απάτης και σχετικών οικονομικών αδικημάτων"
    elif "αναίρεση" in text_to_search.lower() or any("αναίρεσ" in k.lower() for k in keywords):
        theme_content = "Ζητήματα αναιρέσεως και δικονομικού δικαίου"
    elif "σύμβαση" in text_to_search.lower() or any("συμβασ" in k.lower() for k in keywords):
        theme_content = "Ερμηνεία και εκτέλεση συμβάσεων"
    elif "εργατικ" in text_to_search.lower() or any("εργασ" in k.lower() for k in keywords):
        theme_content = "Εργατικές διαφορές και δικαιώματα"
    elif keywords:
        # Use first few keywords if specific common terms not found
        theme_content = "Σχετικά με " + ", ".join(keywords[:2])
        
    return f"Θέμα:\n{theme_content}"

def parse_llm_response(response_text: str) -> str | None:
    """Parses the LLM response to extract the theme."""
    match = re.search(r"Θέμα:\s*\n(.*?)(?:\n\n|$)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    print(f"    Warning: Could not parse LLM response: {response_text}")
    return None

def get_consolidated_title_from_llm(document_summaries: list[str]) -> str:
    """
    Gets themes for 3 document summaries from LLM and consolidates them.
    Consolidation: Majority wins, else random choice.
    """
    if not document_summaries:
        return "Δεν βρέθηκαν έγγραφα για εξαγωγή τίτλου"

    themes = []
    print(f"    Getting themes for {len(document_summaries)} summaries...")
    for i, summary_text in enumerate(document_summaries):
        if not summary_text or not isinstance(summary_text, str):
            print(f"    Skipping invalid summary (index {i}).")
            themes.append(f"Άκυρη περίληψη {i+1}") # Placeholder for invalid summary
            continue

        prompt = LLM_SINGLE_DOC_PROMPT_TEMPLATE.replace("{LEGAL_TEXT}", summary_text)
        llm_response = call_mock_llm(prompt) # Replace with actual LLM call
        parsed_theme = parse_llm_response(llm_response)
        if parsed_theme:
            themes.append(parsed_theme)
        else:
            themes.append(f"Αποτυχία εξαγωγής θέματος {i+1}") # Placeholder for parsing failure

    if not themes:
        return "Αποτυχία εξαγωγής θεμάτων από LLM"

    # Consolidation logic: majority wins, else random
    if len(themes) == 1: # handles cases where less than 3 docs were provided/processed
        return themes[0]
    
    theme_counts = Counter(themes)
    most_common = theme_counts.most_common(1)
    
    if most_common and most_common[0][1] >= 2: # Majority (at least 2 out of 3 are the same)
        final_title = most_common[0][0]
        print(f"    Consolidated title (majority): {final_title}")
    elif themes: # All different (or counts are 1 for all)
        final_title = random.choice(themes)
        print(f"    Consolidated title (random choice from: {themes}): {final_title}")
    else: # Should not happen if themes list was populated
        final_title = "Δεν ήταν δυνατή η δημιουργία τίτλου"
        print(f"    Could not determine consolidated title.")
        
    return final_title

def main():
    """Main function to generate titles for clusters."""
    df_full = load_data(INPUT_CSV_FILE)
    if df_full is None:
        return

    X_full, vectorizer, kmeans_model = refit_vectorizer_and_kmeans(df_full)
    
    # Get the actual cluster assignments from the loaded CSV, not from re-fitting K-means
    # as re-fitting might (rarely, due to initialization) produce slightly different cluster assignments
    # if the data order or exact float representations differ minutely.
    # The centroids from re-fitting are what we need for distance calculations.
    
    unique_cluster_ids = sorted(df_full['cluster_id'].unique())
    print(f"\nFound {len(unique_cluster_ids)} unique clusters: {unique_cluster_ids}")
    print("-" * 50)

    results = []

    for cluster_id in unique_cluster_ids:
        print(f"\nProcessing Cluster ID: {cluster_id}")
        
        # Get documents belonging to the current cluster
        df_cluster = df_full[df_full['cluster_id'] == cluster_id].copy()
        
        # Get TF-IDF vectors for documents in this cluster
        # We need original indices to map back from X_full
        original_indices_in_cluster = df_cluster.index 
        if not original_indices_in_cluster.empty:
            X_cluster_docs = X_full[original_indices_in_cluster]
        else:
            X_cluster_docs = np.array([]) # Empty array if cluster is empty
            
        if X_cluster_docs.ndim == 1 and X_cluster_docs.shape[0] > 0 : # if only one doc in cluster
             X_cluster_docs = X_cluster_docs.reshape(1, -1)


        current_centroid_vector = kmeans_model.cluster_centers_[cluster_id]

        # --- 1. Centroid-based 3-shot ---
        print("  Method 1: Using 3 documents closest to centroid...")
        if X_cluster_docs.shape[0] > 0:
            closest_docs_df = get_docs_closest_to_centroid(df_cluster, X_cluster_docs, current_centroid_vector, n_docs=3)
            centroid_summaries = closest_docs_df[TEXT_COLUMN_TO_USE].tolist()
            if len(centroid_summaries) < 3 and len(centroid_summaries) > 0: # if less than 3 docs, use what we have
                 print(f"    Warning: Cluster {cluster_id} has only {len(centroid_summaries)} docs. Using these for centroid method.")
            elif not centroid_summaries:
                 print(f"    Warning: No documents found for centroid method in cluster {cluster_id}.")
        else:
            centroid_summaries = []
            print(f"    Warning: Cluster {cluster_id} is empty or has no TF-IDF vectors. Skipping centroid method.")

        title_centroid = get_consolidated_title_from_llm(centroid_summaries)
        print(f"  Title (Centroid-based) for Cluster {cluster_id}: {title_centroid}")

        # --- 2. Random 3-shot ---
        print("\n  Method 2: Using 3 random documents from cluster...")
        random_docs_df = get_random_docs_from_cluster(df_cluster, n_docs=3, random_state=RANDOM_STATE)
        random_summaries = random_docs_df[TEXT_COLUMN_TO_USE].tolist()
        if len(random_summaries) < 3 and len(random_summaries) > 0:
             print(f"    Warning: Cluster {cluster_id} has only {len(random_summaries)} docs. Using these for random method.")
        elif not random_summaries:
             print(f"    Warning: No documents found for random method in cluster {cluster_id}.")


        title_random = get_consolidated_title_from_llm(random_summaries)
        print(f"  Title (Random-based) for Cluster {cluster_id}: {title_random}")
        
        results.append({
            "cluster_id": cluster_id,
            "title_centroid": title_centroid,
            "title_random": title_random,
            "cluster_size": len(df_cluster)
        })
        print("-" * 30)

    print("\n\n--- Summary of Generated Titles ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string())

    print("\n\nProcess complete.")
    print("Compare the 'title_centroid' and 'title_random' for each cluster.")
    print("Based on the assignment, you should now 'Επιλέξτε με επιχειρήματα τον καλύτερο τρόπο'.")
    print("Remember to replace 'call_mock_llm' with your actual LLM implementation.")

if __name__ == '__main__':
    main()
