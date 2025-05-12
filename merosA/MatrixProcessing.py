import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from collections import Counter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MatrixProcessing:
    def __init__(self, level='volume', top_words=10000, skip_top_words=20, skip_least_frequent=0):
        """Initialize the MatrixProcessing class for GLC dataset.

        Args:
            level (str): The classification level ('volume', 'chapter', 'subject').
            top_words (int): Total number of words to consider for vocabulary.
            skip_top_words (int): Number of most frequent words to skip.
            skip_least_frequent (int): Number of least frequent words to skip.
        """
        self.level = level  # 'volume', 'chapter', or 'subject'
        self.skip_top_words = skip_top_words
        self.vocab_size = top_words - skip_least_frequent
        self.word_to_index = None
        self.label_to_index = None
        print(f"Vocab size for {self.level}: {self.vocab_size}")

    def load_data(self):
        """Load the GLC dataset from Hugging Face for the specified level."""
        splits = {
            'train': f'{self.level}/train-00000-of-00001.parquet',
            'test': f'{self.level}/test-00000-of-00001.parquet',
            'validation': f'{self.level}/validation-00000-of-00001.parquet'
        }
        base_url = "hf://datasets/AI-team-UoA/greek_legal_code/"
        df_train = pd.read_parquet(base_url + splits['train'])
        df_test = pd.read_parquet(base_url + splits['test'])
        df_val = pd.read_parquet(base_url + splits['validation'])

        # Create vocabulary from training data
        all_words = []
        for text in df_train['text']:
            all_words.extend(self.tokenize(text))
        word_counts = Counter(all_words)
        sorted_words = [word for word, count in word_counts.most_common()]

        # Limit vocabulary
        if len(sorted_words) > self.skip_top_words:
            vocabulary = sorted_words[self.skip_top_words:self.skip_top_words + self.vocab_size]
        else:
            vocabulary = sorted_words

        self.word_to_index = {word: i for i, word in enumerate(vocabulary)}

        # Map labels to indices
        unique_labels = df_train['label'].unique()
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}

        # Convert texts to binary vectors and labels to integers
        x_train = self.map_texts_to_binary_vectors(df_train['text'])
        x_val = self.map_texts_to_binary_vectors(df_val['text'])
        x_test = self.map_texts_to_binary_vectors(df_test['text'])

        y_train = np.array([self.label_to_index[label] for label in df_train['label']])
        y_val = np.array([self.label_to_index.get(label, -1) for label in df_val['label']])
        y_test = np.array([self.label_to_index.get(label, -1) for label in df_test['label']])

        # Handle unseen labels in val/test by filtering them out (optional)
        val_mask = y_val != -1
        test_mask = y_test != -1
        x_val, y_val = x_val[val_mask], y_val[val_mask]
        x_test, y_test = x_test[test_mask], y_test[test_mask]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def tokenize(self, text):
        """Simple tokenization by splitting on whitespace and converting to lowercase."""
        return text.lower().split()

    def map_texts_to_binary_vectors(self, texts):
        """Convert a list of texts to binary vectors based on vocabulary."""
        binary_vectors = []
        for text in texts:
            vector = np.zeros(self.vocab_size, dtype=int)
            tokens = self.tokenize(text)
            for token in tokens:
                if token in self.word_to_index:
                    vector[self.word_to_index[token]] = 1
            binary_vectors.append(vector)
        return np.array(binary_vectors)

    def calculate_information_gain(self, binary_vectors, labels, top_k=1000):
        """Calculate mutual information and return indices of top_k features."""
        mutual_info = mutual_info_classif(binary_vectors, labels, discrete_features=True)
        top_indices = np.argsort(mutual_info)[-top_k:][::-1]
        return top_indices

    def vocabulary(self):
        """Return the vocabulary dictionary."""
        return self.word_to_index