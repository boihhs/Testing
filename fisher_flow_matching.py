# Minimal Fisher-Flow-Matching training example on Tiny Shakespeare.
# This script downloads the dataset if necessary and trains a toy
# neural field with PyTorch. It is intended only as a small educational
# demonstration of the algorithm and is not a full LLM.

import os
import urllib.request
import math

import numpy as np
import torch

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_FILE = "tiny.txt"

# ----------------------------------------------------------------------
# Utilities for dataset handling
# ----------------------------------------------------------------------

def download_dataset(path: str = DATA_FILE) -> str:
    """Download the tiny Shakespeare corpus if not present."""
    if not os.path.exists(path):
        print("Downloading dataset ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_data(seq_len: int = 32):
    """Return tokenized data and vocabulary."""
    path = download_dataset()
    text = open(path, "r", encoding="utf-8").read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    tokens = np.array([stoi[ch] for ch in text], dtype=np.int32)
    idx = np.arange(0, len(tokens) - seq_len - 1)
    return tokens, idx, stoi, itos

# ----------------------------------------------------------------------
# Neural field implemented with PyTorch
# ----------------------------------------------------------------------

class NeuralField(torch.nn.Module):
    """Small MLP neural field to output a scalar alpha."""

    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(2 * embed_dim + 3, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, t, x_s0, x_s1, s0, s1):
        """Forward pass returning alpha scalar."""
        e0 = self.embed(s0).mean(dim=0)
        e1 = self.embed(s1).mean(dim=0)
        feats = torch.cat([e0, e1, torch.tensor([t, x_s0, x_s1])])
        h = torch.tanh(self.fc1(feats))
        alpha = torch.tanh(self.fc2(h))
        return alpha.squeeze()

# ----------------------------------------------------------------------
# Fisher-Rao helper functions
# ----------------------------------------------------------------------

def geodesic_weights(eq: bool, t: torch.Tensor):
    """Return geodesic weights between two one-hot sequences."""
    phi = 0.5 * math.pi * t
    if eq:
        x0, x1 = torch.tensor(1.0), torch.tensor(0.0)
    else:
        x0 = torch.cos(phi) ** 2
        x1 = torch.sin(phi) ** 2
    return x0, x1

def log_map(x0: torch.Tensor, x1: torch.Tensor):
    c = torch.sqrt(x1)
    denom = torch.clamp(torch.sqrt(1.0 - x1), min=1e-6)
    temp = (2.0 * torch.arccos(torch.clamp(c, 0.0, 1.0))) / denom
    lm1 = temp * (c * x0)
    lm0 = -temp * (c * x0)
    return lm0, lm1

def true_velocity(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    lm0, lm1 = log_map(x0, x1)
    inv = 1.0 / torch.clamp(1.0 - t, min=1e-6)
    return lm0 * inv, lm1 * inv

# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------

def compute_loss(model, t: torch.Tensor, s0: torch.Tensor, s1: torch.Tensor):
    eq = torch.all(s0 == s1).item()
    x0, x1 = geodesic_weights(eq, t)
    u0, u1 = true_velocity(x0, x1, t)
    alpha = model(float(t), float(x0), float(x1), s0, s1)
    v1 = alpha * torch.sqrt(torch.clamp(x0 / torch.clamp(x1, min=1e-6), min=0.0))
    v0 = -alpha * torch.sqrt(torch.clamp(x1 / torch.clamp(x0, min=1e-6), min=0.0))
    loss = ((v0 - u0) ** 2) / torch.clamp(x0, min=1e-6)
    loss += ((v1 - u1) ** 2) / torch.clamp(x1, min=1e-6)
    return loss


def sample_sequence(tokens, idx, seq_len, g):
    start = idx[torch.randint(len(idx), (1,), generator=g)].item()
    return tokens[start : start + seq_len]


def train(steps=200, seq_len=32, lr=1e-3, seed=0):
    tokens, idx, stoi, _ = load_data(seq_len)
    tokens = torch.tensor(tokens, dtype=torch.long)
    idx = torch.tensor(idx, dtype=torch.long)
    vocab_size = len(stoi)
    model = NeuralField(vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    for step in range(steps):
        s0 = sample_sequence(tokens, idx, seq_len, g)
        s1 = torch.randint(vocab_size, (seq_len,), generator=g)
        t = torch.rand(1, generator=g) * 0.999 + 0.0005
        loss = compute_loss(model, t, s0, s1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"step {step+1}: loss={float(loss):.4f}")

    return model


if __name__ == "__main__":
    train()
