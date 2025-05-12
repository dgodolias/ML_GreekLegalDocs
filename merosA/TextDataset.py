import torch
from torch.utils.data import Dataset
import numpy as np

class TextDataset(Dataset):
    def __init__(self, x, y, padding_length="average"):
        self.x = x
        self.y = y
        self.padding_length = padding_length
        self.max_length = self.calculate_max_length()

    def calculate_max_length(self):
        if self.padding_length == "average":
            return int(np.mean([len(review) for review in self.x]))
        else:
            return max([len(review) for review in self.x])

    def pad_sequences(self, sequence):
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            return sequence + [0] * (self.max_length - len(sequence))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        padded_sequence = self.pad_sequences(self.x[idx])
        return torch.tensor(padded_sequence, dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

def create_dataset(x, y, padding_length="average"):
    return TextDataset(x, y, padding_length)