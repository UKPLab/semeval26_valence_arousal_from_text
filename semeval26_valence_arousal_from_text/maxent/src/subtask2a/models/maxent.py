import torch
import torch.nn as nn
import numpy as np

class MaxEnt(nn.Module):
    """
    Maximum Entropy model with RESTRICTED STATE SPACE.
    Exact probabilities over a predefined set of valid binary states.
    """

    def __init__(self, states, device='cuda'):
        super().__init__()

        self.device = device
        self.states = states.to(device)          # (N_states, n)
        self.n = self.states.shape[1]
        self.Ns = self.states.shape[0]

        self.h = nn.Parameter(0.1 * torch.randn(self.n, device=device))
        self.J = nn.Parameter(0.1 * torch.randn(self.n, self.n, device=device))

    def _symmetrize_J(self):
        J_sym = (self.J + self.J.T) / 2
        J_sym = J_sym - torch.diag(torch.diag(J_sym))
        return J_sym

    def _energy(self, states=None):
        if states is None:
            states = self.states
        J = self._symmetrize_J()
        linear = states @ self.h
        quadratic = torch.einsum('bi,ij,bj->b', states, J, states)
        return -linear - 0.5 * quadratic

    def _compute_probabilities(self):
        E = self._energy()
        log_Z = torch.logsumexp(-E, dim=0)
        probs = torch.exp(-E - log_Z)
        return probs, torch.exp(log_Z)

    def log_prob(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device) \
            if not isinstance(x, torch.Tensor) else x
        probs, Z = self._compute_probabilities()
        E = self._energy(x)
        return -E - torch.log(Z)

    def loss(self, x, lambda_=0.0):
        log_probs = self.log_prob(x).mean()
        l1_penalty = lambda_ * (self.h.abs().sum() +
                                0.5 * self._symmetrize_J().abs().sum())
        return -log_probs + l1_penalty

    def fit(self, data_np, lr=1e-2, steps=10000, patience=100, lambda_=0.0, verbose=True):
        data = torch.tensor(data_np, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_loss = None
        patience_counter = 0
        best_h = self.h.clone()
        best_J = self._symmetrize_J().clone()

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.loss(data, lambda_)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if verbose:
                print(f"Step {step:5d} | NLL = {current_loss:.6f}", end="\r")

            if best_loss is None or current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                best_h = self.h.clone()
                best_J = self._symmetrize_J().clone()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at step {step}")
                    break

        self.h.data = best_h
        self.J.data = best_J
        self._compute_probabilities()
