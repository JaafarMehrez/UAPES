"""
UAPES: Uncertainty-Aware Path Exploration Sampling
Single-file Implementation for Müller-Brown Potential

This is a complete, standalone implementation that can run directly.

Author: Jaafar Mehrez (2026)
Email: jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
"""

import numpy as np
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

class MullerBrownPotential:
    """
    Müller-Brown potential energy surface.

    U(x, y) = sum_{i=1}^4 A_i * exp(a_i*(x-x0_i)^2 + b_i*(x-x0_i)*(y-y0_i) + c_i*(y-y0_i)^2)
    """

    def __init__(self):
        self.A = np.array([-200.0, -100.0, -170.0, 15.0])
        self.a = np.array([-1.0, -1.0, -6.5, 0.7])
        self.b = np.array([0.0, 0.0, 11.0, 0.6])
        self.c = np.array([-10.0, -10.0, -6.5, 0.7])
        self.x0 = np.array([1.0, 0.0, -0.5, -1.0])
        self.y0 = np.array([0.0, 0.5, 1.5, 1.0])

    def __call__(self, coords):
        # Potential energy
        coords_input = np.atleast_2d(coords.copy())
        x = coords_input[:, 0]
        y = coords_input[:, 1]

        energy = np.zeros(len(x))
        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            gaussian = np.exp(
                self.a[i] * dx**2 + self.b[i] * dx * dy + self.c[i] * dy**2
            )
            energy += self.A[i] * gaussian

        return energy if len(energy) > 1 else energy[0]

    def gradient(self, coords):
        # Gradient
        coords_input = np.atleast_2d(coords.copy())
        x = coords_input[:, 0]
        y = coords_input[:, 1]

        grad_x = np.zeros(len(x))
        grad_y = np.zeros(len(y))

        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            exp_term = np.exp(
                self.a[i] * dx**2 + self.b[i] * dx * dy + self.c[i] * dy**2
            )

            grad_x += self.A[i] * exp_term * (2 * self.a[i] * dx + self.b[i] * dy)
            grad_y += self.A[i] * exp_term * (self.b[i] * dx + 2 * self.c[i] * dy)

        grad = np.column_stack([grad_x, grad_y])
        # Ensure 1D output for single inputs
        if coords.ndim == 1:
            return grad[0]
        return grad

    def force(self, coords):
        # Force
        return -self.gradient(coords)

    def get_minima(self):
        # Basin locations
        return {
            "A": np.array([-0.5, 1.5]),
            "B": np.array([0.5, 0.0]),
            "C": np.array([-1.0, 1.0]),
        }

    def define_basin(self, point, radius=0.5):
        #Check which basin a point is in
        minima = self.get_minima()
        for name, center in minima.items():
            if np.linalg.norm(point - center) < radius:
                return True, name
        return False, "unknown"

class LangevinDynamics:
    # Langevin dynamics simulator

    def __init__(self, potential, friction=2.0, temperature=0.1, dt=0.001):
        self.potential = potential
        self.friction = friction
        self.temperature = temperature
        self.dt = dt
        self.noise_amp = np.sqrt(2 * friction * temperature * dt)

    def step(self, x, v=None):
        # 1 integration step
        if v is None:
            v = np.random.randn(*x.shape) * np.sqrt(self.temperature)

        force = self.potential.force(x)
        v_half = v + (force - self.friction * v) * self.dt / 2
        x_new = x + v_half * self.dt
        force_new = self.potential.force(x_new)
        v_new = v_half + (force_new - self.friction * v_half) * self.dt / 2
        v_new += np.random.randn(*x.shape) * self.noise_amp

        return x_new, v_new

    def run(self, x0, n_steps, record_every=1):
        # Run simulation
        x = x0.copy()
        v = None

        trajectory = [x.copy().tolist()]
        energies = [float(self.potential(x))]

        for step in range(n_steps):
            x, v = self.step(x, v)
            if step % record_every == 0:
                trajectory.append(x.copy().tolist())
                energies.append(float(self.potential(x)))

        return {"trajectory": np.array(trajectory), "energies": np.array(energies)}

class SimpleCommittorModel:
    """
    Simple committor model using distance-based heuristics.

    This replaces the Bayesian NN for the prototype.
    """

    def __init__(self):
        self.basin_A = np.array([-0.5, 1.5])  # Target basin
        self.basin_B = np.array([0.5, 0.0])  # Other basin
        self.trained = False

    def predict_with_uncertainty(self, x):
        """
        Predict committor and uncertainty.

        q = distance_to_A / (distance_to_A + distance_to_B)
        uncertainty = 2 * min(d_A, d_B) / (d_A + d_B)
        """
        x = np.atleast_2d(x)

        dist_A = np.linalg.norm(x - self.basin_A, axis=1)
        dist_B = np.linalg.norm(x - self.basin_B, axis=1)

        total_dist = dist_A + dist_B + 1e-6
        committor = dist_A / total_dist
        uncertainty = 2 * np.minimum(dist_A, dist_B) / total_dist

        return committor, uncertainty

    def train(self, X, y):
        """No training needed for simple model."""
        self.trained = True

class UncertaintyBias:
    """Bias based on predictive uncertainty."""

    def __init__(self, committor_model, alpha=1.0):
        self.committor_model = committor_model
        self.alpha = alpha

    def compute_bias(self, cv_values):
        """Compute bias potential."""
        _, uncertainty = self.committor_model.predict_with_uncertainty(cv_values)
        return self.alpha * uncertainty

    def compute_bias_force(self, cv_values):
        """Compute bias force (numerical gradient)."""
        epsilon = 1e-4
        cv_values = np.atleast_2d(cv_values)

        bias_plus = self.compute_bias(cv_values + epsilon)
        bias_minus = self.compute_bias(cv_values - epsilon)

        force = -(bias_plus - bias_minus) / (2 * epsilon)
        return force

class UAPESSampler:
    """Main UAPES algorithm."""

    def __init__(self, potential, config):
        self.potential = potential
        self.config = config
        self.dynamics = LangevinDynamics(
            potential,
            friction=config.get("friction", 2.0),
            temperature=config.get("temperature", 0.1),
            dt=config.get("dt", 0.001),
        )
        self.committor_model = SimpleCommittorModel()
        self.bias = UncertaintyBias(self.committor_model, config.get("alpha", 1.0))

        self.training_X = []
        self.training_y = []
        self.trajectories = []

    def initialize_training_data(self, n_per_basin=30):
        """Initialize from basin states."""
        print(f"Initializing training data ({3 * n_per_basin} points)...")

        basin_A = np.array([-0.5, 1.5])
        basin_B = np.array([0.5, 0.0])
        basin_C = np.array([-1.0, 1.0])

        for _ in range(n_per_basin):
            self.training_X.append(basin_A + np.random.randn(2) * 0.2)
            self.training_y.append(0.0)

        for _ in range(n_per_basin):
            self.training_X.append(basin_C + np.random.randn(2) * 0.2)
            self.training_y.append(0.5)

        for _ in range(n_per_basin):
            self.training_X.append(basin_B + np.random.randn(2) * 0.2)
            self.training_y.append(1.0)

        self.training_X = np.array(self.training_X)
        self.training_y = np.array(self.training_y)

    def train_model(self):
        """Train committor model."""
        self.committor_model.train(self.training_X, self.training_y)
        print("Model trained")

    def compute_uncertainty_map(self, grid_size=30):
        """Compute uncertainty over CV space."""
        bounds = self.config.get("cv_bounds", (-1.5, 1.5))

        x = np.linspace(bounds[0], bounds[1], grid_size)
        y = np.linspace(bounds[0], bounds[1], grid_size)
        X, Y = np.meshgrid(x, y)

        cv_grid = np.column_stack([X.ravel(), Y.ravel()])
        _, uncertainty = self.committor_model.predict_with_uncertainty(cv_grid)

        return cv_grid, uncertainty.reshape(grid_size, grid_size)

    def find_high_uncertainty_regions(self, n=3):
        """Find regions of highest uncertainty with reasonable energy."""
        cv_grid, uncertainty = self.compute_uncertainty_map()

        # Filter by energy to avoid high-energy regions
        energies = self.potential(cv_grid)
        valid_mask = energies < 20.0

        if not np.any(valid_mask):
            valid_mask = np.ones(len(energies), dtype=bool)

        masked_uncertainty = np.where(valid_mask, uncertainty.ravel(), -np.inf)

        flat_idx = np.argsort(masked_uncertainty)[-n:]
        grid_size = self.config.get("cv_grid_size", 30)
        bounds = self.config.get("cv_bounds", (-1.5, 1.5))

        regions = []
        for idx in flat_idx:
            i = idx // grid_size
            j = idx % grid_size
            x = np.linspace(bounds[0], bounds[1], grid_size)[j]
            y = np.linspace(bounds[0], bounds[1], grid_size)[i]
            regions.append(np.array([x, y]))

        return regions

    def run_biased_md(self, x0, max_steps=5000):
        """Run MD with uncertainty bias."""
        x = x0.copy()
        v = np.random.randn(2) * np.sqrt(self.config.get("temperature", 0.1))
        trajectory = [x.copy()]
        bounds = self.config.get("bounds")

        for step in range(max_steps):
            force = self.potential.force(x)

            try:
                cv = x.reshape(1, -1)
                bias_force = self.bias.compute_bias_force(cv).ravel()
                total_force = force + self.config.get("alpha", 1.0) * bias_force
            except:
                total_force = force

            v_half = (
                v
                + (total_force - self.config.get("friction", 2.0) * v)
                * self.config.get("dt", 0.001)
                / 2
            )
            x_new = x + v_half * self.config.get("dt", 0.001)

            # Clip to bounds
            if bounds is not None:
                x_new = np.clip(x_new, bounds[0], bounds[1])

            force_new = self.potential.force(x_new)
            v_new = (
                v_half
                + (force_new - self.config.get("friction", 2.0) * v_half)
                * self.config.get("dt", 0.001)
                / 2
            )

            x = x_new
            v = v_new

            if step % 100 == 0:
                trajectory.append(x.copy())

            in_basin, basin_name = self.potential.define_basin(x, 0.5)
            if in_basin:
                break
        else:
            basin_name = "unknown"

        return {
            "trajectory": np.array(trajectory),
            "final_basin": basin_name,
            "steps": step,
        }

    def update_training_data(self, trajectory, final_basin):
        """Add new data to training set."""
        # Using: A=0.0, C=0.5, B=1.0
        if final_basin == "A":
            label = 0.0
        elif final_basin == "B":
            label = 1.0
        elif final_basin == "C":
            label = 0.5
        else:
            return

        for x in trajectory[::10]:
            self.training_X = (
                np.vstack([self.training_X, x])
                if len(self.training_X) > 0
                else x.reshape(1, -1)
            )
            self.training_y = np.append(self.training_y, label)

    def run(self, n_iterations=5):
        """Run full UAPES algorithm."""
        self.initialize_training_data()

        print("\n" + "=" * 60)
        print("Tracking Uncertainty Evolution")
        print("=" * 60)
        print(
            f"{'Iter':<6} {'Region':<8} {'Start Point':<22} {'Energy':<12} {'Final Basin'}"
        )
        print("-" * 70)

        for iteration in range(n_iterations):
            print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")

            self.train_model()

            # Find uncertainty regions
            start_points = self.find_high_uncertainty_regions(n=3)

            # Run MD from each region and track results
            basin_counts = {"A": 0, "B": 0, "C": 0, "unknown": 0}

            for i, x0 in enumerate(start_points):
                result = self.run_biased_md(x0)
                self.trajectories.append(result["trajectory"])
                self.update_training_data(result["trajectory"], result["final_basin"])

                basin = result["final_basin"]
                basin_counts[basin] = basin_counts.get(basin, 0) + 1

                # Print uncertainty region with result
                E = float(self.potential(x0))
                print(
                    f"{iteration + 1:<6} {i + 1:<8} [{x0[0]:>7.3f}, {x0[1]:>7.3f}] {E:>10.2f}  -> {basin}"
                )

        print(f"\n{'=' * 70}")
        print(f"Final training points: {len(self.training_X)}")

        # Test committor at different regions
        print("\nFinal Model Predictions:")
        print(f"{'Point':<18} {'Committor':<12} {'Uncertainty':<12} {'Energy':<12}")
        print("-" * 55)

        test_points = [
            ("Basin A", [-0.5, 1.5]),
            ("Basin B", [0.5, 0.0]),
            ("Basin C", [-1.0, 1.0]),
            ("Saddle AB", [-0.1, 0.75]),
            ("Saddle AC", [-0.7, 1.25]),
            ("Transition", [-0.26, 0.57]),
        ]

        for name, pt in test_points:
            q, unc = self.committor_model.predict_with_uncertainty(np.array([pt]))
            E = float(self.potential(pt))
            print(f"{name:<18} {q[0]:<12.4f} {unc[0]:<12.4f} {E:<12.2f}")

        return self.trajectories

def main():
    """Run UAPES on Müller-Brown potential."""
    print("=" * 60)
    print("UAPES: Uncertainty-Aware Path Exploration Sampling")
    print("Testing on Müller-Brown Potential")
    print("=" * 60)

    # Create potential
    potential = MullerBrownPotential()

    # Test potential
    print("\nTesting potential...")
    print(f"  Basin A (-0.5, 1.5): E = {potential([-0.5, 1.5]):.4f}")
    print(f"  Basin B (0.5, 0.0): E = {potential([0.5, 0.0]):.4f}")
    print(f"  Saddle (-0.1, 0.75): E = {potential([-0.1, 0.75]):.4f}")

    # Test dynamics
    print("\nTesting dynamics...")
    dyn = LangevinDynamics(potential, friction=2.0, temperature=0.1)
    result = dyn.run(np.array([-0.5, 1.5]), n_steps=1000)
    print(f"  Trajectory length: {len(result['trajectory'])}")

    # Test committor model
    print("\nTesting committor model...")
    model = SimpleCommittorModel()
    test_points = [[-0.5, 1.5], [0.5, 0.0], [-0.1, 0.75], [0.0, 0.5]]
    x = np.array(test_points)
    q, unc = model.predict_with_uncertainty(x)
    for i, pt in enumerate(test_points):
        print(f"  {pt}: q={q[i]:.4f}, σ={unc[i]:.4f}")

    # Run UAPES
    print("\n" + "=" * 60)
    print("Running UAPES Algorithm")
    print("=" * 60)

    config = {
        "alpha": 2.0,
        "cv_bounds": (-1.5, 1.5),
        "cv_grid_size": 30,
        "temperature": 10.0,
        "friction": 1.0,
        "dt": 0.001,
        "bounds": (-1.5, 1.5),
        "max_md_steps": 10000,
    }

    sampler = UAPESSampler(potential, config)
    trajectories = sampler.run(n_iterations=5)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

    # Plot trajectories
    plot_trajectories(potential, sampler)


def plot_trajectories(potential, sampler):
    """Plot MD trajectories on the potential energy surface."""
    print("\nGenerating trajectory plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot potential energy contours
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-0.5, 2.0, 100)
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack([X.ravel(), Y.ravel()])
    Z = potential(coords).reshape(100, 100)

    # Contour plot
    levels = np.linspace(Z.min(), 100, 20)
    ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.8)
    ax.contour(X, Y, Z, levels=levels, colors="gray", alpha=0.3, linewidths=0.5)

    # Mark basins
    minima = potential.get_minima()
    colors = {"A": "red", "B": "blue", "C": "green"}
    for name, coords in minima.items():
        ax.plot(
            coords[0],
            coords[1],
            "o",
            color=colors[name],
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label=f"Basin {name}",
        )
        ax.annotate(
            name,
            (float(coords[0]) + 0.1, float(coords[1]) + 0.1),
            fontsize=12,
            fontweight="bold",
            color=colors[name],
        )

    # Plot trajectories with different colors for each iteration
    n_traj = len(sampler.trajectories)
    colors_traj = plt.cm.rainbow(np.linspace(0, 1, n_traj))

    for i, traj in enumerate(sampler.trajectories):
        # Subsample for clarity
        traj_plot = traj[:: max(1, len(traj) // 100)]
        ax.plot(
            traj_plot[:, 0],
            traj_plot[:, 1],
            "-",
            color=colors_traj[i],
            alpha=0.6,
            linewidth=1.5,
        )
        # Mark start and end
        ax.plot(
            traj[0, 0],
            traj[0, 1],
            "o",
            markersize=5,
            color=colors_traj[i],
            markeredgecolor="white",
        )
        ax.plot(
            traj[-1, 0],
            traj[-1, 1],
            "s",
            markersize=6,
            color=colors_traj[i],
            markeredgecolor="black",
        )

    # Mark saddle points
    saddles = {"AB": [-0.1, 0.75], "AC": [-0.7, 1.25]}
    for name, coords in saddles.items():
        ax.plot(
            coords[0],
            coords[1],
            "x",
            markersize=10,
            color="yellow",
            markeredgewidth=2,
            label=f"Saddle {name}",
        )
        ax.annotate(
            name + "*",
            (float(coords[0]) - 0.15, float(coords[1]) - 0.1),
            fontsize=10,
            color="yellow",
            fontweight="bold",
        )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        "UAPES: MD Trajectories on Müller-Brown Potential\n"
        "(o: basins, x: saddles, circles: start, squares: end)",
        fontsize=12,
    )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 2.0)

    # Legend
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("uapes_trajectories.png", dpi=150)
    print("Saved: uapes_trajectories.png")


if __name__ == "__main__":
    main()
