import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from collections import Counter
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MatrixProcessing:
    def __init__(self, level='volume', top_words=10000, skip_top_words=20, skip_least_frequent=0, representation='bow'):
        """Initialize the MatrixProcessing class for GLC dataset.

        Args:
            level (str): The classification level ('volume', 'chapter', 'subject').
            top_words (int): Total number of words to consider for vocabulary.
            skip_top_words (int): Number of most frequent words to skip.
            skip_least_frequent (int): Number of least frequent words to skip.
            representation (str): Type of text representation ('bow' or 'tfidf').
        """
        self.level = level
        self.skip_top_words = skip_top_words
        self.vocab_size = top_words - skip_least_frequent
        self.representation = representation.lower()
        self.vectorizer = None
        self.label_to_index = None
        print(f"Vocab size for {self.level} with {self.representation.upper()}: {self.vocab_size}")

    def load_data(self):
        """Load the GLC dataset from Hugging Face for the specified level and preprocess it."""
        splits = {
            'train': f'{self.level}/train-00000-of-00001.parquet',
            'test': f'{self.level}/test-00000-of-00001.parquet',
            'validation': f'{self.level}/validation-00000-of-00001.parquet'
        }
        base_url = "hf://datasets/AI-team-UoA/greek_legal_code/"
        df_train = pd.read_parquet(base_url + splits['train'])
        df_test = pd.read_parquet(base_url + splits['test'])
        df_val = pd.read_parquet(base_url + splits['validation'])

        # Initialize vectorizer based on representation
        if self.representation == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.vocab_size + self.skip_top_words,
                stop_words=None,
                token_pattern=r'(?u)\b\w+\b',
                lowercase=True
            )
        elif self.representation == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.vocab_size + self.skip_top_words,
                stop_words=None,
                token_pattern=r'(?u)\b\w+\b',
                lowercase=True
            )
        else:
            raise ValueError("Representation must be 'bow' or 'tfidf'")

        # Fit vectorizer on training data and transform texts
        x_train = self.vectorizer.fit_transform(df_train['text']).toarray()
        x_val = self.vectorizer.transform(df_val['text']).toarray()
        x_test = self.vectorizer.transform(df_test['text']).toarray()

        # Skip top and least frequent words
        if self.skip_top_words > 0 or self.vocab_size < len(self.vectorizer.vocabulary_):
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            word_counts = np.sum(x_train, axis=0)
            sorted_indices = np.argsort(word_counts)[::-1]
            selected_indices = sorted_indices[self.skip_top_words:self.skip_top_words + self.vocab_size]
            x_train = x_train[:, selected_indices]
            x_val = x_val[:, selected_indices]
            x_test = x_test[:, selected_indices]

        # Map labels to indices
        unique_labels = df_train['label'].unique()
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}

        y_train = np.array([self.label_to_index[label] for label in df_train['label']])
        y_val = np.array([self.label_to_index.get(label, -1) for label in df_val['label']])
        y_test = np.array([self.label_to_index.get(label, -1) for label in df_test['label']])

        # Handle unseen labels in val/test by filtering them out
        val_mask = y_val != -1
        test_mask = y_test != -1
        x_val, y_val = x_val[val_mask], y_val[val_mask]
        x_test, y_test = x_test[test_mask], y_test[test_mask]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def tokenize(self, text):
        """Simple tokenization by splitting on whitespace and converting to lowercase."""
        return text.lower().split()

    def calculate_information_gain(self, vectors, labels, top_k=1000):
        """Calculate mutual information and return indices of top_k features."""
        mutual_info = mutual_info_classif(vectors, labels, discrete_features=True)
        top_indices = np.argsort(mutual_info)[-top_k:][::-1]
        return top_indices

    def vocabulary(self):
        """Return the vocabulary dictionary."""
        return self.vectorizer.vocabulary_ if self.vectorizer else {}