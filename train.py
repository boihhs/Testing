import torch
from dataset import load_data, sample_sequence
from model import NeuralField
from loss import compute_loss


def train(steps: int = 200, seq_len: int = 32, lr: float = 1e-3, seed: int = 0):
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
