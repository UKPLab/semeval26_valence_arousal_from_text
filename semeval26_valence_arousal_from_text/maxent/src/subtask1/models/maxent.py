import torch
import torch.nn as nn
import numpy as np
from itertools import product

class MaxEnt(nn.Module):
    """
    Maximum Entropy (Ising-like) model for binary systems.
    Supports exact computation of probabilities and moments
    for small systems (n < ~20) via full state enumeration.
    """

    def __init__(self, n, device='cuda'):
        super().__init__()
        self.n = n
        self.device = device
        self.h = nn.Parameter(0.1 * torch.randn(n, device=device))
        self.J = nn.Parameter(0.1 * torch.randn(n, n, device=device))
        self.states = self._get_all_states()

    def _symmetrize_J(self):
        """Enforce symmetry and zero out diagonal of interaction matrix."""
        J_sym = (self.J + self.J.T) / 2
        J_sym = J_sym - torch.diag(torch.diag(J_sym))
        return J_sym

    def _get_all_states(self):
        """Enumerate all 2^n binary configurations as a (2^n x n) tensor."""
        state_list = list(product([0, 1], repeat=self.n))
        return torch.tensor(state_list, dtype=torch.float32, device=self.device)

    def _energy(self, states=None):
        """Compute Ising energy: lower energy ⇒ higher probability."""
        if states is None:
            states = self.states
        J = self._symmetrize_J()
        linear = states @ self.h
        quadratic = torch.einsum('bi,ij,bj->b', states, J, states)
        return -linear - 0.5 * quadratic

    def _compute_probabilities(self):
        """Compute normalized probabilities for all binary states."""
        E = self._energy()
        log_Z = torch.logsumexp(-E, dim=0)
        probs = torch.exp(-E - log_Z)
        assert torch.isclose(probs.sum(), torch.tensor(1.0, device=self.device))
        return probs, torch.exp(log_Z)

    def _model_pred(self):
        """Compute model expectations for means and pairwise correlations."""
        probs, _ = self._compute_probabilities()
        first_moment = torch.einsum('b,bi->i', probs, self.states)
        second_moment = torch.einsum('b,bi,bj->ij', probs, self.states, self.states)
        return first_moment, torch.triu(second_moment).flatten()

    def interaction_matrix(self):
        """Return the full effective interaction matrix (J + diag(h))."""
        J = self._symmetrize_J().detach().cpu().numpy()
        return J + np.diag(self.h.detach().cpu().numpy())

    def fit(self, data_np, lr=1e-2, steps=10000, verbose=True, patience=100, lambda_=0.):
        """
        Fit the MaxEnt model by minimizing negative log-likelihood (NLL)
        with optional L1 regularization and early stopping.
        """
        data = torch.tensor(data_np, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss, patience_counter = None, 0
        best_model = {'h': self.h.clone(), 'J': self._symmetrize_J().clone()}

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.loss(data, lambda_=lambda_)
            loss.backward()
            optimizer.step()

            if verbose:
                print(f"Step {step:5d}: Loss = {loss.item():.10f}", end='\r')

            current_loss = loss.item()
            if best_loss is None or current_loss < best_loss:
                best_loss, patience_counter = current_loss, 0
                best_model['h'] = self.h.clone()
                best_model['J'] = self._symmetrize_J().clone()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at step {step}. Best NLL: {best_loss:.5f}")
                    break

        self.h.data = best_model['h']
        self.J.data = best_model['J']
        self._compute_probabilities()

    def get_model_marginals(self):
        """Compute model expectations for means and pairwise correlations."""
        probs, _ = self._compute_probabilities()

        first_moment = torch.einsum('b,bi->i', probs, self.states)
        second_moment = torch.einsum('b,bi,bj->ij', probs, self.states, self.states)

        mask = torch.triu(torch.ones(self.n, self.n, device=self.device), diagonal=1) == 1
        model_cov_flat = second_moment[mask]

        return first_moment, model_cov_flat


    def get_empirical_marginals(self, data_np):
        """Compute empirical means and correlations from data."""
        data = torch.tensor(data_np, dtype=torch.float32, device=self.device) \
            if not isinstance(data_np, torch.Tensor) else data_np
        first_moment = data.mean(dim=0)
        second_moment = (data.T @ data) / data.size(0)
        mask = (torch.triu(torch.ones(self.n, self.n), diagonal=1) == 1).reshape(self.n, self.n)
        return first_moment, torch.triu(second_moment)[mask]

    def r_squared(self, data_np):
        """
        Compute R² for model vs empirical means and correlations.
        Indicates goodness of fit for marginals.
        """
        with torch.no_grad():
            model_mean, model_cov = self.get_model_marginals()
            emp_mean, emp_cov = self.get_empirical_marginals(data_np)

            sse_mean = torch.sum((emp_mean - model_mean) ** 2)
            sst_mean = torch.sum((emp_mean - emp_mean.mean()) ** 2)
            r2_mean = 1 - sse_mean / sst_mean if sst_mean > 0 else 0.0

            #triu_indices = torch.triu_indices(self.n, self.n, offset=1, device=self.device)
            #emp_cov_flat = emp_cov[triu_indices[0], triu_indices[1]]
            #model_cov_flat = model_cov[triu_indices[0], triu_indices[1]]
            emp_cov_flat = emp_cov
            model_cov_flat = model_cov
            # emp_cov and model_cov are already flattened upper-triangle vectors
            sse_cov = torch.sum((emp_cov_flat - model_cov_flat) ** 2)
            sst_cov = torch.sum((emp_cov_flat - emp_cov_flat.mean()) ** 2)
            r2_cov = 1 - sse_cov / sst_cov if sst_cov > 0 else 0.0

            return r2_mean.item(), r2_cov.item()

    def log_prob(self, x):
        """Compute log probability of given binary states."""
        x = torch.tensor(x, dtype=torch.float32, device=self.device) if not isinstance(x, torch.Tensor) else x
        _, Z = self._compute_probabilities()
        E = self._energy(x)
        return -E - torch.log(Z)

    def loss(self, x, lambda_=0.0):
        """Negative log-likelihood loss with L1 regularization on h and J."""
        x = torch.tensor(x, dtype=torch.float32, device=self.device) if not isinstance(x, torch.Tensor) else x
        log_probs = self.log_prob(x).mean()
        l1_penalty = lambda_ * (self.h.abs().sum() + 0.5 * self._symmetrize_J().abs().sum())
        return -log_probs + l1_penalty

    def save(self, path):
        """Save model parameters to disk."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
        return self

    def load(self, path):
        """Load model parameters from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self._compute_probabilities()
        print(f"Model loaded from {path}")
        return self

    def find_minima(self, sample_size=None):
        """
        Identify local energy minima and their basins of attraction.
        Performs steepest descent from each binary configuration.
        """
        with torch.no_grad():
            n, device = self.n, self.device
            all_states = self._get_all_states()
            E_all = self._energy(all_states)
            powers = 2 ** torch.arange(n, device=device)
            state_indices = (all_states * powers).sum(dim=1).long()

            attractor = state_indices.clone()
            E_attractor = E_all[attractor]
            converged = torch.zeros_like(state_indices, dtype=torch.bool)
            max_iter = 100

            for _ in range(max_iter):
                prev = attractor.clone()
                active = (~converged).nonzero(as_tuple=False).view(-1)
                if active.numel() == 0:
                    break

                current_E = E_all[attractor[active]]
                neighbors = attractor[active, None] ^ (1 << torch.arange(n, device=device))
                neighbor_E = E_all[neighbors]
                min_E, min_idx = neighbor_E.min(dim=1)
                best_neighbors = neighbors[torch.arange(neighbors.size(0), device=device), min_idx]
                improved = min_E < current_E

                attractor[active[improved]] = best_neighbors[improved]
                E_attractor[active[improved]] = min_E[improved]
                converged[active[~improved]] = True
                if torch.equal(attractor, prev):
                    break

            final_states = ((attractor.unsqueeze(1) & (1 << torch.arange(n, device=device))) > 0).float()
            unique_minima, inv = torch.unique(final_states, dim=0, return_inverse=True)
            minima_E = self._energy(unique_minima)
            cluster_sizes = torch.bincount(inv, minlength=unique_minima.size(0)).float()
            cluster_sizes /= cluster_sizes.sum()

            order = torch.argsort(cluster_sizes, descending=True)
            return unique_minima[order], minima_E[order], cluster_sizes[order]
