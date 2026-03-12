"""
UAPES: Uncertainty-Aware Path Exploration Sampling
Muller-Brown Potential Implementation

This module provides the Muller-Brown potential and its derivatives,
as well as Langevin dynamics for testing the UAPES algorithm.

The Muller-Brown potential is a classic 2D test case with three
metastable minima (A, B, C) and saddle points connecting them.

Author: Jaafar Mehrez (2026)
Email: jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class MullerBrownParams:
    """
    Parameters for the Muller-Brown potential.

    The potential is:
    U(x,y) = sum_i A_i exp(a_i(x-x0_i)^2 + b_i(x-x0_i)(y-y0_i) + c_i(y-y0_i)^2)

    Default values are the standard Muller-Brown parameters.
    """

    A: np.ndarray = None
    a: np.ndarray = None
    b: np.ndarray = None
    c: np.ndarray = None
    x0: np.ndarray = None
    y0: np.ndarray = None

    def __post_init__(self):
        if self.A is None:
            self.A = np.array([-200.0, -100.0, -170.0, 15.0])
            self.a = np.array([-1.0, -1.0, -6.5, 0.7])
            self.b = np.array([0.0, 0.0, 11.0, 0.6])
            self.c = np.array([-10.0, -10.0, -6.5, 0.7])
            self.x0 = np.array([1.0, 0.0, -0.5, -1.0])
            self.y0 = np.array([0.0, 0.5, 1.5, 1.0])


class MullerBrownPotential:
    """
    Muller-Brown potential energy surface.

    Mathematical Formulation:
    U(x, y) = sum_i=1^4 A_i exp[a_i(x-x0_i)^2 + b_i(x-x0_i)(y-y0_i) + c_i(y-y0_i)^2]

    This is a sum of four Gaussian functions. It has:
    - Three metastable minima: A (deep), B, C
    - Two saddle points
    - A global minimum at approximately (-0.5, 1.5)

    Basin Definitions:
    Basin A (deep minimum): centered at (x,y) = (-0.5, 1.5)
    Basin B: centered at (x,y) = (0.5, 0.0)
    Basin C: centered at (x,y) = (-1.0, 1.0)
    """

    def __init__(self, params: Optional[MullerBrownParams] = None):
        """
        Initialize the Muller-Brown potential.

        Args:
            params: Optional custom parameters
        """
        self.params = params or MullerBrownParams()

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute potential energy.

        Args:
            coords: Array of shape (n_points, 2) or (2,) with (x, y) coordinates

        Returns:
            Energy values, shape (n_points,) or scalar
        """
        coords = np.atleast_2d(coords)
        x = coords[:, 0]
        y = coords[:, 1]

        energy = np.zeros(len(x))

        for i in range(4):
            dx = x - self.params.x0[i]
            dy = y - self.params.y0[i]

            gaussian = np.exp(
                self.params.a[i] * dx**2
                + self.params.b[i] * dx * dy
                + self.params.c[i] * dy**2
            )

            energy += self.params.A[i] * gaussian

        return energy if len(energy) > 1 else energy[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute gradient of potential energy (dU/dx, dU/dy).

        Args:
            coords: Array of shape (n_points, 2) or (2,)

        Returns:
            Gradient array of shape (n_points, 2)
        """
        coords = np.atleast_2d(coords)
        x = coords[:, 0]
        y = coords[:, 1]

        grad_x = np.zeros(len(x))
        grad_y = np.zeros(len(y))

        for i in range(4):
            dx = x - self.params.x0[i]
            dy = y - self.params.y0[i]

            exp_term = np.exp(
                self.params.a[i] * dx**2
                + self.params.b[i] * dx * dy
                + self.params.c[i] * dy**2
            )

            grad_x += (
                self.params.A[i]
                * exp_term
                * (2 * self.params.a[i] * dx + self.params.b[i] * dy)
            )

            grad_y += (
                self.params.A[i]
                * exp_term
                * (self.params.b[i] * dx + 2 * self.params.c[i] * dy)
            )

        return np.column_stack([grad_x, grad_y])

    def force(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute force (negative gradient).

        F = -grad(U)

        Args:
            coords: Coordinates

        Returns:
            Force array
        """
        grad = self.gradient(coords)
        if grad.shape[0] == 1:
            return grad[0]
        return -grad

    def hessian(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix (second derivatives).

        Args:
            coords: Array of shape (n_points, 2)

        Returns:
            Hessian arrays of shape (n_points, 2, 2)
        """
        coords = np.atleast_2d(coords)
        x = coords[:, 0]
        y = coords[:, 1]

        hess = np.zeros((len(x), 2, 2))

        for i in range(4):
            dx = x - self.params.x0[i]
            dy = y - self.params.y0[i]

            exp_term = np.exp(
                self.params.a[i] * dx**2
                + self.params.b[i] * dx * dy
                + self.params.c[i] * dy**2
            )

            d2dx2 = (
                self.params.A[i]
                * exp_term
                * (
                    2 * self.params.a[i]
                    + (2 * self.params.a[i] * dx + self.params.b[i] * dy) ** 2
                )
            )

            d2dy2 = (
                self.params.A[i]
                * exp_term
                * (
                    2 * self.params.c[i]
                    + (self.params.b[i] * dx + 2 * self.params.c[i] * dy) ** 2
                )
            )

            d2dxdy = (
                self.params.A[i]
                * exp_term
                * (
                    self.params.b[i]
                    + (2 * self.params.a[i] * dx + self.params.b[i] * dy)
                    * (self.params.b[i] * dx + 2 * self.params.c[i] * dy)
                )
            )

            hess[:, 0, 0] += d2dx2
            hess[:, 1, 1] += d2dy2
            hess[:, 0, 1] += d2dxdy
            hess[:, 1, 0] += d2dxdy

        return hess

    def get_minima(self) -> dict:
        """
        Get approximate locations of the three minima.

        Returns:
            Dictionary with minima coordinates
        """
        return {
            "A": np.array([-0.5, 1.5]),
            "B": np.array([0.5, 0.0]),
            "C": np.array([-1.0, 1.0]),
        }

    def get_saddle_points(self) -> dict:
        """
        Get approximate saddle point locations.

        Returns:
            Dictionary with saddle point coordinates
        """
        return {"AB": np.array([-0.1, 0.75]), "AC": np.array([-0.7, 1.25])}

    def define_basin(self, point: np.ndarray, radius: float = 0.5) -> Tuple[bool, str]:
        """
        Determine which basin a point belongs to.

        Args:
            point: (x, y) coordinates
            radius: Distance threshold

        Returns:
            (in_basin, basin_name)
        """
        minima = self.get_minima()

        for name, center in minima.items():
            if np.linalg.norm(point - center) < radius:
                return True, name

        return False, "unknown"

    def is_transition_state(self, point: np.ndarray, threshold: float = 0.3) -> bool:
        """
        Check if a point is near a transition state.

        Transition states are approximately where the committor = 0.5,
        which is near the saddle points.

        Args:
            point: (x, y) coordinates
            threshold: Distance threshold from saddle

        Returns:
            True if near a transition state
        """
        saddles = self.get_saddle_points()

        for name, center in saddles.items():
            if np.linalg.norm(point - center) < threshold:
                return True

        return False


class LangevinDynamics:
    """
    Langevin dynamics simulator for the Muller-Brown potential.

    The Langevin equation:
    dR/dt = -grad(U)/m - gamma*v + sqrt(2*gamma*kBT) * R(t)

    Discretized (velocity Verlet):
    v(t+dt/2) = v(t) + (F - gamma*v)dt/2
    x(t+dt) = x(t) + v(t+dt/2)*dt
    v(t+dt) = v(t+dt/2) + (F - gamma*v(t+dt/2))*dt/2

    where:
    - F = -grad(U(x)) is the force
    - gamma is the friction coefficient
    - kB is Boltzmann constant
    - T is temperature
    """

    def __init__(
        self,
        potential: MullerBrownPotential,
        mass: float = 1.0,
        friction: float = 1.0,
        temperature: float = 1.0,
        dt: float = 0.001,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize Langevin dynamics.

        Args:
            potential: Potential energy function
            mass: Particle mass
            friction: Friction coefficient (gamma)
            temperature: Temperature (kBT units)
            dt: Integration timestep
            bounds: Optional (min, max) bounds for position clipping
        """
        self.potential = potential
        self.mass = mass
        self.friction = friction
        self.temperature = temperature
        self.dt = dt
        self.bounds = bounds or (-2.0, 2.0)

        self.noise_amp = np.sqrt(2 * friction * temperature * dt)

    def step(
        self, x: np.ndarray, v: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one integration step using velocity Verlet discretization.

        Args:
            x: Current position
            v: Current velocity (if None, initialize from Maxwell-Boltzmann)

        Returns:
            (new_position, new_velocity)
        """
        if v is None:
            v = np.random.randn(*x.shape) * np.sqrt(self.temperature / self.mass)
        force = self.potential.force(x)
        v_half = v + (force - self.friction * v) * self.dt / 2
        x_new = x + v_half * self.dt

        # Clip position to stay within bounds
        x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
        force_new = self.potential.force(x_new)
        v_new = v_half + (force_new - self.friction * v_half) * self.dt / 2
        noise = np.random.randn(*x.shape) * self.noise_amp
        v_new += noise
        return x_new, v_new

    def run(
        self, x0: np.ndarray, n_steps: int, record_every: int = 1, verbose: bool = False
    ) -> dict:
        """
        Run Langevin dynamics simulation.

        Args:
            x0: Initial position
            n_steps: Number of integration steps
            record_every: Record trajectory every N steps
            verbose: Print progress

        Returns:
            Dictionary with trajectory, energies, etc.
        """
        x = x0.copy()
        v = None

        trajectory = [x.copy()]
        energies = [self.potential(x)]
        velocities = []

        for step in range(n_steps):
            x, v = self.step(x, v)
            if step % record_every == 0:
                trajectory.append(x.copy())
                energies.append(self.potential(x))
                velocities.append(v.copy())
            if verbose and (step + 1) % 1000 == 0:
                print(f"Step {step + 1}/{n_steps}, E={energies[-1]:.4f}")

        return {
            "trajectory": np.array(trajectory),
            "energies": np.array(energies),
            "velocities": np.array(velocities) if velocities else None,
        }

    def run_until_basin(
        self,
        x0: np.ndarray,
        target_basin: str,
        max_steps: int = 100000,
        basin_radius: float = 0.5,
    ) -> Tuple[dict, str]:
        """
        Run until system reaches a target basin.

        Args:
            x0: Initial position
            target_basin: Name of target basin ('A', 'B', or 'C')
            max_steps: Maximum steps before giving up
            basin_radius: Radius to consider as "in basin"

        Returns:
            (trajectory, final_basin)
        """
        x = x0.copy()
        trajectory = [x.copy()]

        for step in range(max_steps):
            x, v = self.step(x)
            trajectory.append(x.copy())

            in_basin, basin_name = self.potential.define_basin(x, basin_radius)

            if in_basin:
                return {
                    "trajectory": np.array(trajectory),
                    "steps": step,
                    "final_energy": self.potential(x),
                }, basin_name

        return {"trajectory": np.array(trajectory), "steps": max_steps}, "unknown"


def create_muller_brown() -> Tuple[MullerBrownPotential, LangevinDynamics]:
    """
    Create standard Muller-Brown system.

    Returns:
        (potential, dynamics)
    """
    potential = MullerBrownPotential()
    dynamics = LangevinDynamics(
        potential=potential, mass=1.0, friction=2.0, temperature=0.1, dt=0.001
    )

    return potential, dynamics


if __name__ == "__main__":
    print("Testing Muller-Brown Implementation...")

    pot, dyn = create_muller_brown()

    x = np.array([0.0, 0.0])
    energy = pot(x)
    print(f"Potential at (0,0): {energy:.4f}")

    grad = pot.gradient(x)
    print(f"Gradient at (0,0): {grad}")

    minima = pot.get_minima()
    for name, coords in minima.items():
        E = pot(coords)
        print(f"Minimum {name}: {coords}, E={E:.4f}")

    print("\nRunning test simulation...")
    result = dyn.run(np.array([0.0, 0.0]), n_steps=1000, record_every=100)
    print(f"Trajectory shape: {result['trajectory'].shape}")
    print(
        f"Energy range: [{result['energies'].min():.2f}, {result['energies'].max():.2f}]"
    )

    print("\n Muller-Brown implementation working!")
