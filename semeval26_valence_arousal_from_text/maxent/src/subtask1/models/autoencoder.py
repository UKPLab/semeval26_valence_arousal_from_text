import torch
import torch.nn as nn

class BinaryAutoencoder(nn.Module):
    def __init__(self, input_dim=60, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def get_z_prob(self, X):
        """
        Returns latent probabilities z_prob in (0,1).
        """
        self.eval()
        logits = self.encoder(X)
        return torch.sigmoid(logits)

    def forward(self, x):
        logits = self.encoder(x)
        z_prob = torch.sigmoid(logits)
        z = (z_prob > 0.5).float() + z_prob - z_prob.detach()  # STE
        x_hat = self.decoder(z)
        return x_hat, z
