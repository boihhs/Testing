import os
import urllib.request
import numpy as np
import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tiny.txt"

def download_dataset(path: str = DATA_FILE) -> str:
    """Download the Tiny Shakespeare corpus if needed."""
    if not os.path.exists(path):
        print("Downloading dataset ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return path

def load_data(seq_len: int = 32):
    path = download_dataset()
    text = open(path, "r", encoding="utf-8").read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    tokens = np.array([stoi[ch] for ch in text], dtype=np.int32)
    idx = np.arange(0, len(tokens) - seq_len - 1)
    return tokens, idx, stoi, itos

def sample_sequence(tokens, idx, seq_len, g):
    start = idx[torch.randint(len(idx), (1,), generator=g)].item()
    return tokens[start: start + seq_len]
