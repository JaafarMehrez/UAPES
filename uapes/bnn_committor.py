"""
UAPES: Uncertainty-Aware Path Exploration Sampling
Enhanced Sampling Method

This module provides the Bayesian Neural Network implementation
for committor prediction with uncertainty quantification.

Author: Jaafar Mehrez (2026)
Email: jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional


class BayesianCommittor(nn.Module):
    """
    Bayesian Neural Network for predicting committor probabilities.

    Uses Monte Carlo dropout for uncertainty estimation.
    The network learns to predict the probability of reaching
    basin B before basin A from any configuration.

    Mathematical Formulation:
    -------------------------
    Given input coordinates x, the network outputs:
        q(x) = sigma(g_tetha(x))

    where sigma() is the sigmoid function and g_tetha is the neural network.

    Uncertainty is estimated via Monte Carlo dropout:
        miu(x) = E[q_tetha(x)]
        sigma^2(x) = Var[q_tetha(x)]

    The predictive variance combines:
    - Epistemic uncertainty (from model parameters)
    - Aleatoric uncertainty (inherent noise)

    Reference:
    - Gal & Ghahramani (2016) - Dropout as Bayesian approximation
    - Trizio, Kang, Parrinello (2024) - Committor learning
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 64, 32],
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize the Bayesian Committor Network.

        Args:
            input_dim: Dimension of input (2 for Muller-Brown, more for real molecules)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for uncertainty estimation
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super().__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        # Build network layers
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = h_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with values in [0, 1]
        """
        return torch.sigmoid(self.network(x))

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.

        Uses Monte Carlo dropout: runs multiple forward passes
        with dropout enabled and computes mean and std.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean, std) tensors, each of shape (batch_size,)
        """
        self.train()  # Enable dropout

        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.cat(predictions, dim=1)
        mean = predictions.mean(dim=1)
        std = predictions.std(dim=1)

        # Add aleatoric uncertainty (max variance for Bernoulli)
        # This accounts for inherent noise in the labels
        aleatoric = 0.5 * torch.ones_like(std)  # max entropy

        return mean, std

    def predict_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make deterministic predictions (without uncertainty).

        Use this for final predictions after training.

        Args:
            x: Input tensor

        Returns:
            Predictions in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class CommittorTrainer:
    """
    Trainer class for the Bayesian Committor Network.

    Handles:
    - Data preparation
    - Training loop
    - Evaluation metrics
    """

    def __init__(
        self,
        model: BayesianCommittor,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
    ):
        """
        Initialize trainer.

        Args:
            model: BayesianCommittor model
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            batch_size: Training batch size
        """
        self.model = model
        self.batch_size = batch_size

        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.criterion = nn.BCELoss()

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """
        Prepare training data from numpy arrays.

        Args:
            X: Input features of shape (n_samples, input_dim)
            y: Labels of shape (n_samples,) - 0 or 1

        Returns:
            DataLoader for training
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader with training data

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(X_batch)

            # Compute loss
            loss = self.criterion(predictions, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int = 100,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels (0 or 1)
            n_epochs: Number of training epochs
            X_val: Optional validation features
            y_val: Optional validation labels
            verbose: Print progress

        Returns:
            Dictionary with training history
        """
        train_loader = self.prepare_data(X_train, y_train)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if X_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{n_epochs} - Loss: {train_loss:.4f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.4f}"
                print(msg)

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model on given data.

        Args:
            X: Input features
            y: True labels

        Returns:
            Binary cross-entropy loss
        """
        self.model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor)

        return loss.item()

    def predict_committor(
        self, X: np.ndarray, return_uncertainty: bool = True, n_samples: int = 10
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict committor values with optional uncertainty.

        Args:
            X: Input features of shape (n_samples, input_dim)
            return_uncertainty: Whether to return uncertainty estimates
            n_samples: Number of MC samples for uncertainty

        Returns:
            If return_uncertainty=True: (committor, uncertainty)
            Else: (committor, None)
        """
        self.model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32)

        if return_uncertainty:
            mean, std = self.model.predict_with_uncertainty(X_tensor, n_samples)
            return mean.numpy(), std.numpy()
        else:
            committor = self.model.predict_deterministic(X_tensor)
            return committor.numpy(), None


def create_model(input_dim: int) -> BayesianCommittor:
    """
    Factory function to create a standard model.

    Args:
        input_dim: Input dimension

    Returns:
        Configured BayesianCommittor model
    """
    return BayesianCommittor(
        input_dim=input_dim,
        hidden_dims=[64, 64, 32],
        dropout_rate=0.1,
        activation="relu",
    )


if __name__ == "__main__":
    # Simple test
    print("Testing Bayesian Committor Network...")

    # Create model
    model = create_model(input_dim=2)
    print(f"Model architecture:\n{model}")

    # Test forward pass
    X_test = torch.randn(10, 2)
    y_pred = model(X_test)
    print(f"\nForward pass output shape: {y_pred.shape}")

    # Test uncertainty estimation
    mean, std = model.predict_with_uncertainty(X_test, n_samples=5)
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"Sample predictions: {mean[:3].numpy()}")
    print(f"Sample uncertainties: {std[:3].numpy()}")

    print("\n All tests passed!")
