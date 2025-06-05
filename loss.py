import math
import torch


def geodesic_weights(eq: bool, t: torch.Tensor):
    """Return geodesic weights between two one-hot sequences."""
    phi = 0.5 * math.pi * t
    if eq:
        x0 = torch.tensor(1.0)
        x1 = torch.tensor(0.0)
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
