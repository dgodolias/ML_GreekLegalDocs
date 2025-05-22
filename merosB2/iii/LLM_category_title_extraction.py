import csv
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import re
import random
# from collections import Counter # Counter is not used with the new single-call LLM approach
import sys
import time

# --- Dynamically add path to import llm_utils from one level up ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_UTILS_DIR = os.path.join(SCRIPT_DIR, '..') # One level up from merosB2/iii/ is merosB2/
sys.path.append(LLM_UTILS_DIR)

try:
    from llm_utils import send_message_to_llm
    print(f"Successfully imported send_message_to_llm from: {os.path.join(LLM_UTILS_DIR, 'llm_utils.py')}")
except ImportError as e:
    print(f"Error importing send_message_to_llm: {e}")
    print(f"Please ensure llm_utils.py is in the directory: {LLM_UTILS_DIR}")
    send_message_to_llm = None # Placeholder if import fails

# --- Configuration ---
CHOSEN_K = 21
TEXT_COLUMN_TO_USE = 'summary'
RANDOM_STATE = 42
INPUT_CSV_FILE = os.path.join(SCRIPT_DIR, '..', 'documents_with_clusters.csv')

# Lists of specific cluster_ids to process. If a list is empty, all clusters will be processed for that method.
# Example: CLUSTERS_TO_PROCESS_CENTROID = [0, 1, 5]
# Example: CLUSTERS_TO_PROCESS_RANDOM = [2, 3]
CLUSTERS_TO_PROCESS_CENTROID = [] 
CLUSTERS_TO_PROCESS_RANDOM = []

# TF-IDF Parameters
TFIDF_MAX_DF = 0.90
TFIDF_MIN_DF = 5
TFIDF_NGRAM_RANGE = (1, 2)

# LLM Prompt for multiple documents
LLM_MULTI_DOC_PROMPT_TEMPLATE = (
    "Σου δίνονται {NUM_SUMMARIES} περιλήψεις νομικών αποφάσεων που ανήκουν στην ίδια θεματική κατηγορία:\n\n"
    "{SUMMARIES_TEXT}\n\n"
    "Βάσει αυτών των περιλήψεων, ποιο είναι το κεντρικό, ενοποιημένο θέμα που τις καλύπτει όλες; "
    "Απάντησε με έναν σύντομο τίτλο (ιδανικά 3-7 λέξεις) στην μορφή: ‘Θέμα:\\n\"<ο τίτλος σου εδώ>\"’. "
    "Μην προσθέτεις εισαγωγικά σχόλια, εξηγήσεις ή περιττές φράσεις."
)

# Markers for LLM call outcomes
LLM_CALL_FAILURE_MARKER = "_LLM_CALL_FAILED_"

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
        print(f"Error: File not found at '{csv_path}'. Please ensure the CSV file exists at this location.")
        sys.exit(1) # Exit if data can't be loaded
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1) # Exit if data can't be loaded

def refit_vectorizer_and_kmeans(df_full, current_k_value):
    """Re-fits TF-IDF and K-means."""
    print("\nRe-fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_df=TFIDF_MAX_DF, min_df=TFIDF_MIN_DF, ngram_range=TFIDF_NGRAM_RANGE, stop_words=None
    )
    X_full = vectorizer.fit_transform(df_full[TEXT_COLUMN_TO_USE])
    print(f"TF-IDF matrix re-fitted. Shape: {X_full.shape}")

    print(f"\nRe-fitting K-means model for K={current_k_value}...")
    kmeans_model = KMeans(n_clusters=current_k_value, init='k-means++', n_init='auto', random_state=RANDOM_STATE)
    kmeans_model.fit(X_full)
    print("K-means model re-fitted.")
    return X_full, vectorizer, kmeans_model

def get_docs_closest_to_centroid(df_cluster_docs, X_cluster_docs, centroid_vector, n_docs=3):
    """Finds n_docs closest to the centroid."""
    if X_cluster_docs.shape[0] == 0: return pd.DataFrame()
    distances = euclidean_distances(X_cluster_docs, centroid_vector.reshape(1, -1))
    closest_indices_in_cluster = np.argsort(distances.ravel())[:n_docs]
    return df_cluster_docs.iloc[closest_indices_in_cluster]

def get_random_docs_from_cluster(df_cluster_docs, n_docs=3, random_state=None):
    """Selects n_docs randomly."""
    if df_cluster_docs.shape[0] == 0: return pd.DataFrame()
    if df_cluster_docs.shape[0] <= n_docs: return df_cluster_docs
    return df_cluster_docs.sample(n=n_docs, random_state=random_state)

def call_actual_llm(prompt_text: str) -> str:
    """
    Calls the actual LLM using send_message_to_llm from llm_utils, with retries.
    Returns the raw response string from LLM on success, or LLM_CALL_FAILURE_MARKER.
    """
    if send_message_to_llm is None:
        print("    LLM utility (send_message_to_llm) not available. Returning mock-like response.")
        keywords = re.findall(r'\b[Α-Ωα-ωίϊΐόάέύϋΰώΊΪΌΆΈΎΫΏ]{5,}\b', prompt_text.split("Περίληψη 1:")[-1] if "Περίληψη 1:" in prompt_text else prompt_text)
        theme_content = "Mock: Σχετικά με " + ", ".join(keywords[:2]) if keywords else "Mock: Γενικό θέμα"
        return f"Θέμα:\n\"{theme_content}\"" # Return raw mock response in expected LLM format, with quotes

    effective_system_prompt = (
        "Είσαι ένας βοηθός που εξάγει το κεντρικό θέμα από τα παρεχόμενα κείμενα. "
        "Απάντησε ΜΟΝΟ με το θέμα ακολουθώντας τη μορφή <ο τίτλος>, Για παραδειγμα: "
        "Input: <Κειμενο>, Output: <ο τίτλος> "
        "Μην προσθέτεις εισαγωγικά σχόλια, εξηγήσεις ή περιττές φράσεις."
    )

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 3

    for attempt in range(MAX_RETRIES):
        print(f"    LLM call attempt {attempt + 1}/{MAX_RETRIES}...")
        raw_response = send_message_to_llm(
            user_message=prompt_text,
            system_message=effective_system_prompt,
        )
        
        if raw_response: 
            if raw_response.strip().startswith("Θέμα:") or raw_response.strip().lower().startswith("θέμα:"): 
                print(f"    LLM attempt {attempt + 1} successful and response format seems OK.")
                return raw_response.strip() 
            else:
                print(f"    LLM attempt {attempt + 1} response malformed (did not start with 'Θέμα:'): '{raw_response[:150]}...'")
                if attempt == MAX_RETRIES - 1: 
                    print(f"    LLM returned malformed response after all retries. Passing as is for parsing.")
                    return raw_response.strip() 
        
        msg_reason = "returned empty" if not raw_response else "was malformed and not fixed"
        print(f"    LLM call attempt {attempt + 1} failed ({msg_reason}).")
        if attempt < MAX_RETRIES - 1:
            print(f"    Retrying in {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)
            
    print(f"    LLM call failed after {MAX_RETRIES} attempts.")
    return LLM_CALL_FAILURE_MARKER


def parse_llm_response(raw_llm_output: str) -> str:
    """
    Parses the raw LLM output. Returns a clean theme string, or an error/status string.
    Handles removal of literal "\\n" and ensures consistent quoting for valid themes.
    """
    if raw_llm_output == LLM_CALL_FAILURE_MARKER:
        return "Αποτυχία επικοινωνίας με LLM" # Unquoted error

    match = re.search(r"Θέμα:\s*\n?(.*?)(?:\n\n|$)", raw_llm_output, re.DOTALL | re.IGNORECASE)
    
    theme_text = ""

    if match:
        theme_text = match.group(1).strip() 
    else:
        lower_raw = raw_llm_output.lower()
        thema_keyword = "θέμα:"
        if thema_keyword in lower_raw:
            last_occurrence_index = lower_raw.rfind(thema_keyword)
            theme_text = raw_llm_output[last_occurrence_index + len(thema_keyword):].strip()
            if not theme_text:
                 print(f"    Warning: Fallback parsing found 'θέμα:' but no content after it. Raw: '{raw_llm_output[:100]}...'")
                 theme_text = raw_llm_output 
            else:
                 print(f"    Using fallback parsing (found 'θέμα:') for: '{raw_llm_output[:100]}...' -> Potential theme: '{theme_text[:50]}...'")
        else: 
            if not (raw_llm_output.startswith("Θέμα:") or raw_llm_output.lower().startswith("θέμα:")):
                 print(f"    Warning: 'Θέμα:' prefix not found by regex or keyword. Treating raw response as potential theme. Raw: '{raw_llm_output[:100]}...'")
            theme_text = raw_llm_output

    # --- Universal Cleaning for theme_text ---
    cleaned_theme = theme_text.strip()
    
    if '\\n' in cleaned_theme: 
        original_theme_for_log = cleaned_theme
        while cleaned_theme.startswith('\\n'):
            cleaned_theme = cleaned_theme[2:]
            cleaned_theme = cleaned_theme.strip() 
        if cleaned_theme != original_theme_for_log and (original_theme_for_log.startswith('\\n')):
            print(f"    Cleaned leading literal '\\n'. Original: '{original_theme_for_log[:60]}...', New: '{cleaned_theme[:60]}...'")

    # Strip surrounding quotes if present to get the "core" theme.
    # This handles cases where the LLM includes them (as prompted) or if the raw input was already quoted.
    if cleaned_theme.startswith('"') and cleaned_theme.endswith('"') and len(cleaned_theme) > 1:
        cleaned_theme = cleaned_theme[1:-1].strip() # Strip content inside quotes too
    
    cleaned_theme = cleaned_theme.strip() # Final strip after potential quote removal
    # --- End of Universal Cleaning ---

    if not cleaned_theme:
        return "Κενό θέμα από LLM (μετά από ανάλυση)" # Unquoted error
    
    if cleaned_theme.startswith("Θέμα:") or cleaned_theme.lower().startswith("θέμα:"):
        print(f"    Warning: Final cleaned theme still starts with 'Θέμα:'. Raw input was: '{raw_llm_output[:100]}...' Cleaned: '{cleaned_theme[:100]}...'")
        cleaned_theme = re.sub(r"^(Θέμα:|θέμα:)\s*\n?", "", cleaned_theme, flags=re.IGNORECASE).strip()
        while cleaned_theme.startswith('\\n'): 
            cleaned_theme = cleaned_theme[2:]
            cleaned_theme = cleaned_theme.strip()
        if not cleaned_theme:
            return "Κενό θέμα από LLM (μετά από επιθετική διόρθωση)" # Unquoted error
    
    if (not match and len(cleaned_theme) > 150) and cleaned_theme == raw_llm_output.strip(): 
         print(f"    Warning: Fallback parsed theme is very long and identical to raw input, might be unparsed response: '{cleaned_theme[:70]}...'")
         return f"Σφάλμα ανάλυσης LLM (μακροσκελές/αμετάβλητο): {cleaned_theme[:70]}"
    
    if cleaned_theme == raw_llm_output and (cleaned_theme.startswith("Θέμα:") or cleaned_theme.lower().startswith("θέμα:")):
        return f"Σφάλμα ανάλυσης LLM (αμετάβλητο): {cleaned_theme[:70]}"

    return cleaned_theme



def get_consolidated_title_from_llm(document_summaries: list[str]) -> str:
    """
    Sends multiple document summaries to the LLM (single call) to get a consolidated theme.
    Returns a clean theme (quoted) or an error/status string (unquoted).
    """
    if not document_summaries:
        return "Δεν βρέθηκαν έγγραφα για εξαγωγή τίτλου" # Unquoted status

    print(f"    Getting consolidated theme for {len(document_summaries)} summaries (single LLM call)...")

    summaries_text_parts = []
    for i, summary_text in enumerate(document_summaries):
        if not summary_text or not isinstance(summary_text, str):
            print(f"    Skipping invalid summary (index {i}).")
            continue 
        summaries_text_parts.append(f"Περίληψη {i+1}:\n{summary_text}")
    
    if not summaries_text_parts:
        return "Δεν υπάρχουν έγκυρες περιλήψεις για επεξεργασία" # Unquoted status

    summaries_block = "\n\n".join(summaries_text_parts)
    num_actual_summaries = len(summaries_text_parts)
    
    prompt = LLM_MULTI_DOC_PROMPT_TEMPLATE.replace("{NUM_SUMMARIES}", str(num_actual_summaries))
    prompt = prompt.replace("{SUMMARIES_TEXT}", summaries_block)

    llm_response_raw = call_actual_llm(prompt) 
    final_title = parse_llm_response(llm_response_raw) # Will be quoted if successful, or unquoted error
    
    print(f"    Consolidated title: {final_title}")
    return final_title

def main():
    """Main function to generate titles for clusters."""
    df_full = load_data(INPUT_CSV_FILE)

    if 'cluster_id' in df_full.columns and df_full['cluster_id'].nunique() > 0:
        k_for_refit = df_full['cluster_id'].nunique()
        print(f"K value for refitting K-means (determined from CSV): {k_for_refit}")
    else:
        print("Error: 'cluster_id' column is missing, empty, or has no unique values in the CSV.")
        print(f"Using CHOSEN_K = {CHOSEN_K} as a fallback, but this might be incorrect.")
        k_for_refit = CHOSEN_K 

    X_full, vectorizer, kmeans_model = refit_vectorizer_and_kmeans(df_full, k_for_refit)
    
    unique_cluster_ids_from_csv = sorted(df_full['cluster_id'].unique())
    print(f"\nFound {len(unique_cluster_ids_from_csv)} unique clusters in CSV: {unique_cluster_ids_from_csv}")
    
    if not CLUSTERS_TO_PROCESS_CENTROID and not CLUSTERS_TO_PROCESS_RANDOM:
        clusters_to_iterate_overall_set = set(unique_cluster_ids_from_csv)
        print("Processing all clusters found in CSV for both methods (as specific lists are empty).")
    else:
        clusters_to_iterate_overall_set = set()
        if CLUSTERS_TO_PROCESS_CENTROID:
            clusters_to_iterate_overall_set.update(CLUSTERS_TO_PROCESS_CENTROID)
            print(f"Method 1 (Centroid) will be attempted for specified clusters: {sorted(list(CLUSTERS_TO_PROCESS_CENTROID))}")
        else:
            clusters_to_iterate_overall_set.update(unique_cluster_ids_from_csv) 
            print("Method 1 (Centroid) will be attempted for all clusters (as its specific list is empty).")

        if CLUSTERS_TO_PROCESS_RANDOM:
            clusters_to_iterate_overall_set.update(CLUSTERS_TO_PROCESS_RANDOM)
            print(f"Method 2 (Random) will be attempted for specified clusters: {sorted(list(CLUSTERS_TO_PROCESS_RANDOM))}")
        else:
            clusters_to_iterate_overall_set.update(unique_cluster_ids_from_csv) 
            print("Method 2 (Random) will be attempted for all clusters (as its specific list is empty).")

    clusters_to_iterate_overall = sorted([cid for cid in clusters_to_iterate_overall_set if cid in unique_cluster_ids_from_csv])

    if not clusters_to_iterate_overall and unique_cluster_ids_from_csv:
        print(f"\nWarning: No valid clusters to process based on specified lists and available CSV data.")
        print(f"  Specified for Centroid: {CLUSTERS_TO_PROCESS_CENTROID if CLUSTERS_TO_PROCESS_CENTROID else 'Attempt All'}")
        print(f"  Specified for Random:   {CLUSTERS_TO_PROCESS_RANDOM if CLUSTERS_TO_PROCESS_RANDOM else 'Attempt All'}")
        print(f"  Available in CSV:     {unique_cluster_ids_from_csv}")
    elif not unique_cluster_ids_from_csv:
        print("\nNo clusters found in the CSV to process.")
        results_df = pd.DataFrame(columns=["cluster_id", "title_centroid", "title_random", "cluster_size"])
        output_csv_filename = "cluster_titles_summary.csv"
        output_csv_path = os.path.join(SCRIPT_DIR, output_csv_filename)
        try:
            results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
            print(f"\nNo clusters to process. Empty results saved to: {output_csv_path}")
        except Exception as e:
            print(f"\nError saving empty results to CSV: {e}")
        return 

    print(f"\nOverall, iterating through clusters: {clusters_to_iterate_overall}")
    print("-" * 50)

    results = []
    model_used_for_llm_calls = "google/gemma-3-4b-it:free" 

    for cluster_id in clusters_to_iterate_overall:
        print(f"\nProcessing Cluster ID: {cluster_id}")
        
        df_cluster = df_full[df_full['cluster_id'] == cluster_id].copy()
        original_indices_in_cluster = df_cluster.index 
        
        X_cluster_docs = np.array([]) 
        if not original_indices_in_cluster.empty:
            try:
                X_cluster_docs = X_full[original_indices_in_cluster]
                if X_cluster_docs.ndim == 1 and X_cluster_docs.shape[0] > 0 : 
                     X_cluster_docs = X_cluster_docs.reshape(1, -1) 
            except IndexError:
                 print(f"    Error: Index out of bounds when accessing TF-IDF vectors for cluster {cluster_id}. Skipping TF-IDF dependent methods for this cluster.")
                 X_cluster_docs = np.array([]) 
            
        if cluster_id >= k_for_refit: 
            print(f"  Skipping cluster_id {cluster_id} as it's out of bounds for re-fitted K-means model (K={k_for_refit}).")
            results.append({ # Unquoted error messages
                "cluster_id": cluster_id,
                "title_centroid": "Σφάλμα: ID Συστάδας εκτός ορίων μοντέλου",
                "title_random": "Σφάλμα: ID Συστάδας εκτός ορίων μοντέλου",
                "cluster_size": len(df_cluster)
            })
            continue

        current_centroid_vector = kmeans_model.cluster_centers_[cluster_id]

        title_centroid = "N/A - Δεν επεξεργάστηκε (εκτός λίστας CENTROID)" # Unquoted status
        run_method_1 = (not CLUSTERS_TO_PROCESS_CENTROID) or (cluster_id in CLUSTERS_TO_PROCESS_CENTROID)

        if run_method_1:
            print("  Method 1: Using 3 documents closest to centroid...")
            if X_cluster_docs.shape[0] > 0:
                closest_docs_df = get_docs_closest_to_centroid(df_cluster, X_cluster_docs, current_centroid_vector, n_docs=3)
                centroid_summaries = closest_docs_df[TEXT_COLUMN_TO_USE].tolist()
                if not centroid_summaries:
                     print(f"    Warning: No documents found for centroid method in cluster {cluster_id}, though X_cluster_docs was not empty.")
                     title_centroid = "N/A - Δεν βρέθηκαν έγγραφα για μέθοδο κεντροειδούς" # Unquoted status
                elif len(centroid_summaries) < 3:
                     print(f"    Warning: Cluster {cluster_id} has only {len(centroid_summaries)} docs for centroid method. Using these.")
                
                if centroid_summaries: # Check again in case it became empty after warning
                    title_centroid = get_consolidated_title_from_llm(centroid_summaries) # Will be quoted if title, unquoted if error
            else: 
                print(f"    Warning: Cluster {cluster_id} is empty or has no TF-IDF vectors. Skipping centroid method.")
                title_centroid = "N/A - Μη διαθέσιμα TF-IDF vectors για μέθοδο κεντροειδούς" # Unquoted status
        else:
            print(f"  Method 1: Skipped for cluster {cluster_id} as per CLUSTERS_TO_PROCESS_CENTROID.")
        print(f"  Τίτλος (Βάσει Κεντροειδούς) για Συστάδα {cluster_id}: {title_centroid}")

        title_random = "N/A - Δεν επεξεργάστηκε (εκτός λίστας RANDOM)" # Unquoted status
        run_method_2 = (not CLUSTERS_TO_PROCESS_RANDOM) or (cluster_id in CLUSTERS_TO_PROCESS_RANDOM)

        if run_method_2:
            print("\n  Method 2: Using 3 random documents from cluster...")
            random_docs_df = get_random_docs_from_cluster(df_cluster, n_docs=3, random_state=RANDOM_STATE)
            random_summaries = random_docs_df[TEXT_COLUMN_TO_USE].tolist()

            if not random_summaries:
                print(f"    Warning: No documents found for random method in cluster {cluster_id} (cluster might be smaller than n_docs or empty).")
                title_random = "N/A - Δεν βρέθηκαν έγγραφα για τυχαία μέθοδο" # Unquoted status
            elif len(random_summaries) < 3 : 
                 print(f"    Warning: Cluster {cluster_id} provided {len(random_summaries)} docs for random method (less than 3 requested). Using these.")
            
            if random_summaries: # Check again
                title_random = get_consolidated_title_from_llm(random_summaries) # Will be quoted if title, unquoted if error
        else:
            print(f"  Method 2: Skipped for cluster {cluster_id} as per CLUSTERS_TO_PROCESS_RANDOM.")
        print(f"  Τίτλος (Βάσει Τυχαίων) για Συστάδα {cluster_id}: {title_random}")
        
        results.append({
            "cluster_id": cluster_id,
            "title_centroid": title_centroid,
            "title_random": title_random,
            "cluster_size": len(df_cluster)
        })
        print("-" * 30)

    print("\n\n--- Περίληψη Δημιουργημένων Τίτλων ---")
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results to display or save.")
    else:
        pd.set_option('display.max_colwidth', None)
        print(results_df.to_string())

        output_csv_filename = "cluster_titles_summary.csv"
        output_csv_path = os.path.join(SCRIPT_DIR, output_csv_filename)
        try:
            results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"\nΤα αποτελέσματα αποθηκεύτηκαν στο: {output_csv_path}")
        except Exception as e:
            print(f"\nΣφάλμα κατά την αποθήκευση των αποτελεσμάτων σε CSV: {e}")

    print("\n\nΗ διαδικασία ολοκληρώθηκε.")
    print("Συγκρίνετε τους τίτλους 'title_centroid' και 'title_random' για κάθε συστάδα.")
    print("Βεβαιωθείτε ότι το αρχείο .env με το OPENROUTER_API_KEY βρίσκεται στον ριζικό κατάλογο του project (ML_GreekLegalDocs/.env).")
    print(f"Οι κλήσεις LLM θα χρησιμοποιήσουν το μοντέλο που ορίζεται στην call_actual_llm (π.χ., {model_used_for_llm_calls}).")

if __name__ == '__main__':
    main()
