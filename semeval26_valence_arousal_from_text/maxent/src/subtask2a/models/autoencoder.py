import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        return grad_output

def binary_ste(x):
    return BinarySTE.apply(x)


class BinaryAutoencoder(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=32, latent_dim=10):
        super().__init__()

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        z_logits = torch.sigmoid(self.enc_fc2(h))   # probabilities
        z_bin = binary_ste(z_logits)
        return z_bin, z_logits

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        x_hat = torch.sigmoid(self.dec_fc2(h))
        return x_hat

    def forward(self, x):
        z_bin, z_logits = self.encode(x)
        x_hat = self.decode(z_bin)
        return x_hat, z_logits