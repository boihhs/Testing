import torch
from dataset import load_data, sample_sequence
from model import NeuralField
from train import train


def generate(model: NeuralField, seq_len: int, vocab_size: int, steps: int = 20):
    g = torch.Generator().manual_seed(0)
    tokens, idx, _, itos = load_data(seq_len)
    tokens = torch.tensor(tokens, dtype=torch.long)
    idx = torch.tensor(idx, dtype=torch.long)
    s0 = sample_sequence(tokens, idx, seq_len, g)
    s1 = torch.randint(vocab_size, (seq_len,), generator=g)
    t = 1.0
    x0 = torch.tensor(0.0)
    x1 = torch.tensor(1.0)
    dt = -1.0 / steps
    for _ in range(steps):
        alpha = model(float(t), float(x0), float(x1), s0, s1)
        v1 = alpha * torch.sqrt(torch.clamp(x0 / torch.clamp(x1, min=1e-6), min=0.0))
        v0 = -alpha * torch.sqrt(torch.clamp(x1 / torch.clamp(x0, min=1e-6), min=0.0))
        x0 = x0 + v0 * dt
        x1 = x1 + v1 * dt
        total = x0 + x1
        x0 = torch.clamp(x0 / total, 0.0, 1.0)
        x1 = torch.clamp(x1 / total, 0.0, 1.0)
        t += dt
    chosen = s0 if x0 >= x1 else s1
    return ''.join(itos[i.item()] for i in chosen)


if __name__ == "__main__":
    model = train(steps=50)
    tokens, _, stoi, _ = load_data(32)
    text = generate(model, 32, len(stoi))
    print("generated:\n", text)
