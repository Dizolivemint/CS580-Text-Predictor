import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TextPredictionDataset(Dataset):
    def __init__(self, csv_path, tokenizer, seq_len):
        df = pd.read_csv(csv_path)
        self.contexts = df['context'].tolist()
        self.targets = df['target'].tolist()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        target = self.targets[idx]
        context_ids = self.tokenizer(context)[:self.seq_len]
        # Pad or truncate as needed:
        context_ids = context_ids + [0]*(self.seq_len - len(context_ids))
        target_id = self.tokenizer(target)[0]  # Assuming target is 1 token
        return torch.tensor(context_ids), torch.tensor(target_id)

def get_data_loaders(csv_path, tokenizer, seq_len=10, batch_size=32, val_split=0.1):
    dataset = TextPredictionDataset(csv_path, tokenizer, seq_len)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
