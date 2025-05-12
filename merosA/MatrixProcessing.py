import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.feature_selection import mutual_info_classif
from keras.api.datasets import imdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MatrixProcessing:
    def __init__(self, top_words=10000, skip_top_words=20, skip_least_frequent=0, start_char=1,
                 oov_char=2, index_from=3):
        self.skip_top_words = skip_top_words
        self.index_from = index_from
        self.vocab_size = top_words - skip_least_frequent
        self.start_char = start_char
        self.oov_char = oov_char
        print("Vocab size:", self.vocab_size)

    def load_data(self):
        # Load the predefined train and test splits from the IMDb dataset.
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            num_words=self.vocab_size,
            skip_top=self.skip_top_words,
            index_from=self.index_from,
            start_char=self.start_char,
            oov_char=self.oov_char
        )

        #Half of the training data is used for validation.
        split_index = len(x_train) // 2
        x_val = x_train[split_index:]
        y_val = y_train[split_index:]
        x_train = x_train[:split_index]
        y_train = y_train[:split_index]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def map_reviews_to_binary_vectors(self, dataset):
        binary_vectors = []
        for review in dataset:
            vector = np.zeros(self.vocab_size, dtype=int)
            for word_index in review:
                if word_index >= self.index_from:
                    vector[word_index - self.index_from] = 1
            binary_vectors.append(vector)
        return binary_vectors

    def calculate_information_gain(self, binary_vectors, labels, top_k=1000):
        # Calculate mutual information for each feature.
        mutual_info = mutual_info_classif(binary_vectors, labels, discrete_features=True)
        # Get the indices of the top_k features with the highest mutual information.
        top_indices = np.argsort(mutual_info)[-top_k:]
        # Reverse to sort in descending order.
        top_indices = top_indices[::-1]
        return top_indices

    def vocabulary(self):
        word_index = imdb.get_word_index()
        # Adjust the word indices based on index_from.
        word_index = {k: (v + self.index_from) for k, v in word_index.items()}
        
        # Sort the word index by frequency.
        sorted_word_index = sorted(word_index.items(), key=lambda item: item[1])
        
        # Skip the top most frequent words and limit the vocabulary size.
        limited_word_index = sorted_word_index[self.skip_top_words:self.vocab_size + self.skip_top_words]
        
        # Create a new dictionary with the limited vocabulary.
        limited_word_index = {k: v for k, v in limited_word_index}
        
        return limited_word_index
