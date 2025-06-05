# Testing

This repository provides a small demonstration of Fisher--Flow--Matching on discrete sequences. The code trains a toy neural field on the Tiny Shakespeare corpus using PyTorch.

## Usage

Install dependencies (PyTorch is required) and run training:

```bash
python3 train.py
```

After training you can generate a short sample with:

```bash
python3 inference.py
```

The implementation is intentionally minimal and mirrors the algorithm from [arXiv:2405.14664](https://arxiv.org/abs/2405.14664).
