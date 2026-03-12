# UAPES: Uncertainty-Aware Path Exploration Sampling

Demo of enhanced sampling method for molecular dynamics with predictive uncertainty to guide exploration of transition pathways between stable states.

## Overview

UAPES combines:
- **Committor functions**: Probability of reaching one basin before another
- **Uncertainty quantification**: Identifies regions where predictions are uncertain
- **Active learning**: Iteratively explores and improves the model

## Installation

```bash
git clone git@github.com:JaafarMehrez/UAPES.git
cd uapes
pip install -e .
```

## Requirements

- numpy
- matplotlib
- torch (optional, for BNN committor)

## Quick Start

```python
from uapes import UAPESSampler, UAPESConfig, MullerBrownPotential

# Create configuration
config = UAPESConfig(
    alpha=2.0,
    n_iterations=10,
    temperature=10.0,
    friction=1.0,
)

# Create sampler
potential = MullerBrownPotential()
sampler = UAPESSampler(config, potential_fn=potential)

# Run UAPES
results = sampler.run()
```

## Algorithm

1. **Initialize**: Start with basin states (A, B, C)
2. **Train**: Learn committor model from training data
3. **Identify**: Find high-uncertainty regions in CV space
4. **Bias**: Apply uncertainty-based bias potential
5. **Sample**: Run biased MD trajectories
6. **Update**: Add new data to training set
7. **Repeat**: Iterate until convergence

The bias potential: **V(s) = α × σ(s)**

Where σ(s) is the predictive uncertainty of the committor at collective variable s.

## Example: Müller-Brown Potential

```bash
python run_muller_brown.py
```

This runs UAPES on the classic Müller-Brown potential with three basins (A, B, C).

## Modules

| Module | Description |
|--------|-------------|
| `sampling` | Core UAPES algorithm |
| `bnn_committor` | Bayesian NN for committor prediction |
| `uncertainty_bias` | Uncertainty-based bias computation |
| `muller_brown` | Müller-Brown potential and dynamics |


