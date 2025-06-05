import torch

class NeuralField(torch.nn.Module):
    """Small MLP neural field returning a scalar alpha."""
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(2 * embed_dim + 3, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, t: float, x_s0: float, x_s1: float, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        e0 = self.embed(s0).mean(dim=0)
        e1 = self.embed(s1).mean(dim=0)
        feats = torch.cat([e0, e1, torch.tensor([t, x_s0, x_s1])])
        h = torch.tanh(self.fc1(feats))
        alpha = torch.tanh(self.fc2(h))
        return alpha.squeeze()
