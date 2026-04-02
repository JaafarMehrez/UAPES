"""
UAPES: Uncertainty-Aware Path Exploration Sampling
Uncertainty Bias Module

This module computes the bias potential based on predictive uncertainty
from the Bayesian neural network committor model.

Author: Jaafar Mehrez (2026)
Email: jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
"""

import numpy as np
import torch
from typing import Optional, Callable


class UncertaintyBias:
    """
    Computes bias potential based on predictive uncertainty.

    Mathematical Formulation:
    -------------------------
    The bias potential is defined as:

        V(s) = alpha * sigma(s)

    where:
        - s is the collective variable (CV)
        - sigma(s) is the predictive uncertainty
        - alpha is a coupling constant (controls bias strength)

    The biasing force is:

        F_bias = -grad(V(s)) = -alpha * grad(sigma(s))

    This biases the system toward regions of high uncertainty,
    which correspond to transition states.

    Comparison with Traditional Methods:
    -------------------------------------------------------
    | Method       | Bias Target  | Goal                   |
    |--------------|--------------|------------------------|
    | Metadynamics | -F(s)        | Fill minima            |
    | OPES         | -log(p(s))   | Match target dist      |
    | UAPES        | alpha(s)     | Find transition states |

    Reference:
    - Zhu et al. (2025) - Enhanced Sampling in Age of ML
    - Trizio, Kang, Parrinello (2024) - Variational Committor
    """

    def __init__(
        self,
        committor_model,
        alpha: float = 1.0,
        scale_factor: float = 1.0,
        use_uncertainty_boost: bool = True,
    ):
        """
        Initialize uncertainty bias calculator.

        Args:
            committor_model: Trained BayesianCommittor model
            alpha: Coupling constant (bias strength)
            scale_factor: Scale factor for CV normalization
            use_uncertainty_boost: If True, bias increases prob of uncertain regions
        """
        self.committor_model = committor_model
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.use_uncertainty_boost = use_uncertainty_boost

    def compute_bias(self, cv_values: np.ndarray) -> np.ndarray:
        """
        Compute bias potential at given CV values.

        V(s) = alpha * sigma(s)

        Args:
            cv_values: Array of CV values, shape (n_points, cv_dim)

        Returns:
            Bias potential at each point, shape (n_points,)
        """
        cv_tensor = torch.tensor(cv_values, dtype=torch.float32)

        with torch.no_grad():
            _, uncertainty = self.committor_model.predict_with_uncertainty(cv_tensor)

        bias = self.alpha * uncertainty.numpy()

        return bias

    def compute_bias_force(
        self, cv_values: np.ndarray, epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        Compute biasing force (-dV/ds) using numerical differentiation.

        F(s) = -dV/ds ≈ -[V(s+e) - V(s-e)] / (2e)

        Args:
            cv_values: CV values at which to compute force
            e: Small step for numerical differentiation

        Returns:
            Force array, same shape as cv_values
        """
        cv_tensor = torch.tensor(cv_values, dtype=torch.float32, requires_grad=True)

        _, uncertainty = self.committor_model.predict_with_uncertainty(cv_tensor)

        # Force = -alpha * d(uncertainty)/ds
        force = (
            -self.alpha
            * torch.autograd.grad(
                uncertainty.sum(), cv_tensor, create_graph=True, retain_graph=True
            )[0]
        )

        return force.numpy()

    def compute_gradient_of_uncertainty(self, cv_values: np.ndarray) -> np.ndarray:
        """
        Compute gradient of uncertainty w.r.t. CVs.

        dsigma/ds = E[q * dq/ds] - E[q] * E[dq/ds]

        This is needed for applying the bias force in MD.

        Args:
            cv_values: CV values

        Returns:
            Gradient of uncertainty
        """
        cv_tensor = torch.tensor(cv_values, dtype=torch.float32, requires_grad=True)

        # Get multiple predictions for uncertainty
        self.committor_model.train()

        predictions = []
        gradients = []

        for _ in range(10):  # MC samples
            pred = self.committor_model(cv_tensor)
            predictions.append(pred)

            grad = torch.autograd.grad(pred.sum(), cv_tensor, retain_graph=True)[0]
            gradients.append(grad)

        predictions = torch.cat(predictions, dim=1)
        gradients = torch.cat(gradients, dim=1)

        # Compute expectation
        E_pred = predictions.mean(dim=1, keepdim=True)
        E_grad = gradients.mean(dim=1, keepdim=True)

        # Compute gradient of uncertainty
        # dsigma^2/ds = E[q^2 * dq/ds] - E[q^2] * E[dq/ds] for variance
        E_pred_grad = (predictions**2 * gradients).mean(dim=1, keepdim=True)
        E_pred_sq = (predictions**2).mean(dim=1, keepdim=True)

        grad_uncertainty = E_pred_grad - E_pred_sq * E_grad

        return grad_uncertainty.numpy()

    def apply_bias_to_energy(
        self, positions: np.ndarray, potential_fn: Callable, cv_fn: Callable
    ) -> float:
        """
        Compute total energy with uncertainty bias.

        E_total = E_original + V_bias

        Args:
            positions: Atomic positions
            potential_fn: Function that computes potential energy
            cv_fn: Function that computes CV from positions

        Returns:
            Total biased energy
        """
        # Get CV value
        cv = cv_fn(positions)

        # Get uncertainty bias
        bias = self.compute_bias(cv.reshape(1, -1))[0]

        # Total energy
        energy = potential_fn(positions) + bias

        return energy

    def get_bias_gradient_wrt_positions(
        self, positions: np.ndarray, cv_fn: Callable, cv_grad_fn: Callable
    ) -> np.ndarray:
        """
        Compute gradient of bias potential w.r.t. atomic positions.

        dV/dR = (dV/ds) * (ds/dR)

        Args:
            positions: Atomic positions
            cv_fn: Function to compute CV
            cv_grad_fn: Function to compute ds/dR

        Returns:
            Gradient w.r.t. positions
        """
        # Get CV and its gradient
        cv = cv_fn(positions)
        dCV_dR = cv_grad_fn(positions)

        # Get dV/ds
        dV_ds = self.compute_bias_force(cv.reshape(1, -1))[0]

        # Chain rule
        dV_dR = np.dot(dCV_dR, dV_ds)

        return dV_dR


class AdaptiveUncertaintyBias(UncertaintyBias):
    """
    Uncertainty bias with adaptive alpha scheduling.

    The coupling constant alpha can be increased during training
    as the model becomes more confident, allowing finer
    exploration of transition regions.
    """

    def __init__(
        self,
        committor_model,
        initial_alpha: float = 1.0,
        max_alpha: float = 10.0,
        schedule: str = "linear",
    ):
        """
        Initialize adaptive bias.

        Args:
            committor_model: Trained BayesianCommittor model
            initial_alpha: Starting alpha value
            max_alpha: Maximum alpha value
            schedule: Scheduling strategy ('linear', 'exp', 'sqrt')
        """
        super().__init__(committor_model, alpha=initial_alpha)

        self.initial_alpha = initial_alpha
        self.max_alpha = max_alpha
        self.schedule = schedule
        self.iteration = 0

    def update_alpha(self, iteration: int):
        """
        Update alpha based on iteration number.

        Args:
            iteration: Current iteration
        """
        self.iteration = iteration

        if self.schedule == "linear":
            self.alpha = min(self.initial_alpha + 0.1 * iteration, self.max_alpha)
        elif self.schedule == "exp":
            self.alpha = min(self.initial_alpha * (1.1**iteration), self.max_alpha)
        elif self.schedule == "sqrt":
            self.alpha = min(
                self.initial_alpha + self.max_alpha * np.sqrt(iteration / 100),
                self.max_alpha,
            )

    def get_current_alpha(self) -> float:
        """Get current alpha value."""
        return self.alpha


def create_uncertainty_bias(committor_model, alpha: float = 1.0) -> UncertaintyBias:
    """
    Factory function to create uncertainty bias calculator.

    Args:
        committor_model: Trained BayesianCommittor
        alpha: Bias coupling constant

    Returns:
        UncertaintyBias instance
    """
    return UncertaintyBias(committor_model=committor_model, alpha=alpha)


if __name__ == "__main__":
    print("Testing Uncertainty Bias Module...")

    # Create mock committor model for testing
    from uapes.bnn_committor import create_model

    model = create_model(input_dim=2)

    # Create bias calculator
    bias = create_uncertainty_bias(model, alpha=2.0)

    # Test bias computation
    cv_test = np.random.randn(100, 2)
    bias_values = bias.compute_bias(cv_test)

    print(f"CV shape: {cv_test.shape}")
    print(f"Bias values shape: {bias_values.shape}")
    print(f"Bias range: [{bias_values.min():.4f}, {bias_values.max():.4f}]")

    # Test force computation
    force = bias.compute_bias_force(cv_test[:5])
    print(f"Force shape: {force.shape}")

    print("\n Uncertainty bias module working!")
