"""
UAPES: Uncertainty-Aware Path Exploration Sampling
Main Sampling Module

This module implements the core UAPES algorithm that combines:
- Bayesian neural network for committor prediction
- Uncertainty-based biasing
- Active learning loop

Author: Jaafar Mehrez (2026)
Email: jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass, field

# Optional dependencies
try:
    from .bnn_committor import BayesianCommittor, CommittorTrainer

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .muller_brown import MullerBrownPotential, LangevinDynamics

    HAS_MB = True
except ImportError:
    HAS_MB = False

from .uncertainty_bias import UncertaintyBias


@dataclass
class UAPESConfig:
    """
    Configuration for UAPES algorithm.

    Attributes:
        alpha: Bias coupling constant
        n_iterations: Number of active learning iterations
        n_mc_samples: Monte Carlo samples for uncertainty
        cv_bounds: Bounds for CV space
        basin_radius: Radius to define basins
        max_md_steps: Maximum MD steps per iteration
        dt: MD timestep
        temperature: Temperature (kBT)
        friction: Friction coefficient
    """

    alpha: float = 1.0
    n_iterations: int = 10
    n_mc_samples: int = 10
    cv_bounds: Tuple[float, float] = (-2.0, 2.0)
    cv_grid_size: int = 20
    basin_radius: float = 0.5
    max_md_steps: int = 10000
    dt: float = 0.001
    temperature: float = 0.1
    friction: float = 2.0
    record_every: int = 100
    bounds: Optional[Tuple[float, float]] = None


class SimpleCommittorModel:
    """
    Simplified committor model for when PyTorch is not available.

    Uses a simple distance-based heuristic to estimate committor
    and uncertainty without deep learning.
    """

    def __init__(self):
        self.basin_A = np.array([-0.5, 1.5])  # Target basin
        self.basin_B = np.array([0.5, 0.0])  # Other basin
        self.trained = False

    def predict_with_uncertainty(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict committor and uncertainty.

        Uses distance-based heuristic:
        - Committor = sigmoid(distance to B / distance to A)
        - Uncertainty = higher when equidistant from both basins
        """
        x = np.atleast_2d(x)

        # Distance to each basin
        dist_A = np.linalg.norm(x - self.basin_A, axis=1)
        dist_B = np.linalg.norm(x - self.basin_B, axis=1)

        # Committor: probability of reaching B before A
        # Higher when closer to B
        total_dist = dist_A + dist_B + 1e-6
        committor = dist_A / total_dist

        # Uncertainty: maximum when equidistant
        uncertainty = 2 * np.minimum(dist_A, dist_B) / total_dist

        return committor, uncertainty

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model (no-op for simple model)."""
        self.trained = True


class UAPESSampler:
    """
    Main UAPES (Uncertainty-Aware Path Exploration Sampling) sampler.

    Algorithm Overview:
    -------------------
    1. Initialize with some basin states
    2. Train committor model
    3. Compute uncertainty map
    4. Apply uncertainty-based bias
    5. Run biased MD
    6. Update training data
    7. Repeat until convergence

    Mathematical Framework:
    -----------------------
    The bias potential: V(s) = α * σ(s)

    Where:
    - s = collective variable
    - σ(s) = predictive uncertainty of committor

    This biases the system toward transition regions
    where the model is uncertain (typically near saddle points).

    Reference:
    - This work (2026)
    """

    def __init__(
        self,
        config: UAPESConfig,
        potential_fn: Optional[Callable] = None,
        use_torch: bool = False,
    ):
        """
        Initialize UAPES sampler.

        Args:
            config: UAPES configuration
            potential_fn: Optional potential energy function
            use_torch: Use PyTorch BNN (requires torch installed)
        """
        self.config = config
        self.use_torch = use_torch and HAS_TORCH

        # Set up potential and dynamics
        if potential_fn is not None:
            self.potential = potential_fn
            self.dynamics = LangevinDynamics(
                potential=potential_fn,
                friction=config.friction,
                temperature=config.temperature,
                dt=config.dt,
                bounds=config.bounds,
            )
        elif HAS_MB:
            from .muller_brown import create_muller_brown

            self.potential, self.dynamics = create_muller_brown()
        else:
            raise ValueError("No potential function provided")

        # Initialize committor model
        if self.use_torch:
            self.committor_model = BayesianCommittor(input_dim=2)
        else:
            self.committor_model = SimpleCommittorModel()

        # Initialize uncertainty bias
        self.bias = UncertaintyBias(
            committor_model=self.committor_model, alpha=config.alpha
        )

        # Training data
        self.training_X = []
        self.training_y = []

        # Results storage
        self.trajectories = []
        self.uncertainty_maps = []

    def initialize_training_data(self, n_samples_per_basin: int = 50):
        """
        Initialize training data from basin states.

        Args:
            n_samples_per_basin: Number of samples per basin
        """
        print("Initializing training data...")

        # Basin A samples
        basin_A_center = np.array([-0.5, 1.5])
        for _ in range(n_samples_per_basin):
            x = basin_A_center + np.random.randn(2) * 0.2
            self.training_X.append(x)
            self.training_y.append(0.0)  # Label: reaches A first

        # Basin C samples (intermediate)
        basin_C_center = np.array([-1.0, 1.0])
        for _ in range(n_samples_per_basin):
            x = basin_C_center + np.random.randn(2) * 0.2
            self.training_X.append(x)
            self.training_y.append(0.5)  # Label: reaches C

        # Basin B samples
        basin_B_center = np.array([0.5, 0.0])
        for _ in range(n_samples_per_basin):
            x = basin_B_center + np.random.randn(2) * 0.2
            self.training_X.append(x)
            self.training_y.append(1.0)  # Label: reaches B first

        self.training_X = np.array(self.training_X)
        self.training_y = np.array(self.training_y)

        print(f"Created {len(self.training_X)} initial training points")

    def train_committor_model(self, n_epochs: int = 100):
        """
        Train the committor model.

        Args:
            n_epochs: Number of training epochs (if using PyTorch)
        """
        print("Training committor model...")

        if self.use_torch:
            trainer = CommittorTrainer(self.committor_model)
            history = trainer.train(
                self.training_X, self.training_y, n_epochs=n_epochs, verbose=False
            )
            print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        else:
            self.committor_model.train(self.training_X, self.training_y)
            print("Model trained (simple model)")

    def compute_uncertainty_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute uncertainty map over CV space.

        Returns:
            (CV_grid, uncertainty_values)
        """
        # Create CV grid
        bounds = self.config.cv_bounds
        grid_size = self.config.cv_grid_size

        x = np.linspace(bounds[0], bounds[1], grid_size)
        y = np.linspace(bounds[0], bounds[1], grid_size)
        X, Y = np.meshgrid(x, y)

        cv_grid = np.column_stack([X.ravel(), Y.ravel()])

        # Compute uncertainty
        _, uncertainty = self.committor_model.predict_with_uncertainty(cv_grid)

        uncertainty_map = uncertainty.reshape(grid_size, grid_size)

        self.uncertainty_maps.append(uncertainty_map)

        return cv_grid, uncertainty_map

    def find_high_uncertainty_regions(self, n_regions: int = 5) -> List[np.ndarray]:
        """
        Find regions of highest uncertainty that are also physically reasonable.

        Args:
            n_regions: Number of high-uncertainty regions to find

        Returns:
            List of starting points for MD
        """
        cv_grid, uncertainty_map = self.compute_uncertainty_map()

        # Compute energy at each grid point
        energies = self.potential(cv_grid)

        # Filter: only consider points with reasonable energy
        # Points with too high energy are not useful for sampling
        valid_mask = energies < 20.0  # Only below this energy

        if not np.any(valid_mask):
            valid_mask = np.ones(len(energies), dtype=bool)

        # Apply mask to uncertainty
        masked_uncertainty = np.where(valid_mask, uncertainty_map.ravel(), -np.inf)

        # Flatten and find top indices among valid points
        flat_idx = np.argsort(masked_uncertainty)[-n_regions:]

        # Get coordinates
        grid_size = self.config.cv_grid_size
        regions = []
        for idx in flat_idx:
            i = idx // grid_size
            j = idx % grid_size
            x = np.linspace(
                self.config.cv_bounds[0], self.config.cv_bounds[1], grid_size
            )[j]
            y = np.linspace(
                self.config.cv_bounds[0], self.config.cv_bounds[1], grid_size
            )[i]
            regions.append(np.array([x, y]))

        return regions

    def run_biased_md(self, x0: np.ndarray, bias_strength: float = 1.0) -> dict:
        """
        Run MD with uncertainty bias.

        Args:
            x0: Starting position
            bias_strength: Additional bias multiplier

        Returns:
            Trajectory results
        """
        x = x0.copy()
        v = np.random.randn(2) * np.sqrt(self.config.temperature)
        trajectory = [x.copy()]

        for step in range(self.config.max_md_steps):
            # Compute unbiased force
            force = self.potential.force(x)

            # Add uncertainty bias force
            try:
                cv = x.reshape(1, -1)
                bias_force = self.bias.compute_bias_force(cv).ravel()
                total_force = force + bias_strength * bias_force
            except:
                total_force = force

            # Velocity Verlet with bias
            v_half = v + (total_force - self.config.friction * v) * self.config.dt / 2
            x_new = x + v_half * self.config.dt

            # Clip position to stay within bounds
            if self.config.bounds is not None:
                x_new = np.clip(x_new, self.config.bounds[0], self.config.bounds[1])

            # Force at new position
            force_new = self.potential.force(x_new)
            v_new = (
                v_half
                + (force_new - self.config.friction * v_half) * self.config.dt / 2
            )

            # Noise
            noise = np.sqrt(
                2 * self.config.friction * self.config.temperature * self.config.dt
            )
            v_new += np.random.randn(2) * noise

            x = x_new
            v = v_new

            if step % self.config.record_every == 0:
                trajectory.append(x.copy())

            # Check if reached basin
            in_basin, basin_name = self.potential.define_basin(
                x, self.config.basin_radius
            )
            if in_basin:
                break
        else:
            basin_name = "unknown"

        return {
            "trajectory": np.array(trajectory),
            "final_basin": basin_name,
            "n_steps": step,
        }

    def update_training_data(self, trajectory: np.ndarray, final_basin: str):
        """
        Update training data with new trajectory.

        Args:
            trajectory: MD trajectory
            final_basin: Basin reached at end
        """
        # Label each point by which basin it reached
        # Using: A=0.0, C=0.5, B=1.0
        if final_basin == "A":
            label = 0.0
        elif final_basin == "B":
            label = 1.0
        elif final_basin == "C":
            label = 0.5
        else:
            return  # Unknown basin, skip

        # Add every nth point to training
        for x in trajectory[::10]:
            self.training_X = (
                np.vstack([self.training_X, x])
                if len(self.training_X) > 0
                else x.reshape(1, -1)
            )
            self.training_y = np.append(self.training_y, label)

    def run(self, n_iterations: Optional[int] = None) -> List[dict]:
        """
        Run the full UAPES algorithm.

        Args:
            n_iterations: Number of iterations (uses config if None)

        Returns:
            List of results from each iteration
        """
        n_iterations = n_iterations or self.config.n_iterations

        # Initialize
        self.initialize_training_data()

        all_results = []

        for iteration in range(n_iterations):
            print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")

            self.train_committor_model()
            start_points = self.find_high_uncertainty_regions(n_regions=3)
            print(f"Found {len(start_points)} high-uncertainty regions")
            for i, x0 in enumerate(start_points):
                print(f"  Running MD from region {i + 1}...")
                result = self.run_biased_md(x0)
                self.trajectories.append(result["trajectory"])
                self.update_training_data(result["trajectory"], result["final_basin"])
                print(f"    Reached basin: {result['final_basin']}")
            all_results.append(
                {
                    "iteration": iteration,
                    "n_training_points": len(self.training_X),
                    "start_points": start_points,
                    "trajectories": self.trajectories[-len(start_points) :],
                }
            )

        print(f"\n=== UAPES Complete ===")
        print(f"Total training points: {len(self.training_X)}")

        return all_results

    def get_results_summary(self) -> dict:
        """
        Get summary of UAPES results.

        Returns:
            Summary dictionary
        """
        return {
            "n_iterations": len(self.trajectories),
            "total_training_points": len(self.training_X),
            "uncertainty_maps": self.uncertainty_maps,
            "final_trajectories": self.trajectories,
        }


def create_uapes(
    alpha: float = 1.0, use_torch: bool = False, potential_fn: Optional[Callable] = None
) -> UAPESSampler:
    """
    Factory function to create UAPES sampler.

    Args:
        alpha: Bias coupling constant
        use_torch: Use PyTorch BNN
        potential_fn: Custom potential function

    Returns:
        Configured UAPES sampler
    """
    config = UAPESConfig(alpha=alpha)
    return UAPESSampler(config, potential_fn=potential_fn, use_torch=use_torch)


if __name__ == "__main__":
    print("Testing UAPES Sampler...")

    # Create sampler
    sampler = create_uapes(alpha=2.0)

    # Run short test
    sampler.initialize_training_data(n_samples_per_basin=20)

    # Train once
    sampler.train_committor_model()

    # Compute uncertainty map
    cv_grid, uncertainty = sampler.compute_uncertainty_map()
    print(f"Uncertainty map shape: {uncertainty.shape}")
    print(f"Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")

    # Find high uncertainty regions
    regions = sampler.find_high_uncertainty_regions(n_regions=3)
    print(f"High uncertainty regions: {regions}")

    print("\n UAPES sampler working!")
