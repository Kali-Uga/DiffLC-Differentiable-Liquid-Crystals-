# DiffLC: A Differentiable Landau-de Gennes Framework for Liquid Crystal Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-blue?logo=jax&logoColor=white)](https://github.com/google/jax)

**DiffLC** is a high-performance numerical engine designed for the simulation of nematic liquid crystal (LC) dynamics and the solution of complex inverse problems. Built on the **JAX** autograd engine, DiffLC treats the entire physical pipeline—from $Q$-tensor relaxation to Berreman 4×4 optics—as a differentiable operator.

This framework is specifically engineered to bridge the gap between classical continuum mechanics and modern machine learning workflows, enabling high-fidelity parameter recovery and Bayesian Optimal Experimental Design (OED).

## Core Capabilities

- **Differentiable Physics:** Full support for exact first-order parametric sensitivities using JAX's `jacfwd` auto-differentiation, bypassing the inaccuracies of finite-difference methods.
- **Landau-de Gennes (LdG) Dynamics:** Fast tridiagonal Thomas algorithm paired with local implicit Euler (Newton) updates for thermotropic bulk energy. The scalar order parameter $S$ evolves freely (no fixed-S projections).
- **Advanced Optical Modeling:** Features a full **Berreman 4×4 matrix engine** supporting oblique incidence, multiple wavelengths, and boundary matching for reflected/transmitted amplitudes. Returns complete Stokes vectors.
- **Parametric Identification:** Multi-cell TRF inverse solvers recovering physical constants ($K_{11}$, $K_{22}$, $K_{33}$, $\gamma_1$, $W$) and information-theoretic utilities (Fisher Information Matrix).

## Technical Architecture

The codebase is modularized to support both direct physical simulations and integration into larger AI-driven research pipelines:

- `src/difflc/solver.py`: Semi-implicit time-stepping LdG engine via JAX `lax.scan` and `jax.jit`.
- `src/difflc/qtensor.py`: Gauge-invariant $Q$-tensor operations and eigenvalue diagnostics (scalar order, biaxiality).
- `src/difflc/optics.py`: Differentiable optical response models (Berreman 4×4 and Jones fallback).
- `src/difflc/oed.py`: Bayesian utilities for Fisher Information and D-optimality.
- `src/difflc/inverse.py`: Non-linear parameter recovery via multi-start Trust Region methods.

## Installation

```bash
git clone https://github.com/Kali-Uga/DiffLC-Differentiable-Liquid-Crystals-.git
cd DiffLC-Differentiable-Liquid-Crystals-
pip install -e .
```

## Quick Start

```python
import numpy as np
from difflc import default_cfg, default_cells, build_protocols, make_model

# 1. Configuration (E7 material)
cfg = default_cfg()
cells = default_cells()
protocols = build_protocols(cells, cfg)

# 2. Build differentiable model for the first cell
p0 = protocols[0]
model = make_model(cfg, p0.cell)

# 3. True parameters [K11, K22, K33, gamma1, W]
p_true = np.array([cfg.K11, cfg.K22, cfg.K33, cfg.gamma1, cfg.W])

# 4. Run forward simulation
out = model.run_protocol_np(p_true, p0.V_abs)

print(f"Time steps: {out['time'].shape}")
print(f"Stokes vectors (rec, wl, theta, pol, 4): {out['stokes'].shape}")

# 5. Get JAX exact Jacobian of the normalized Stokes signals w.r.t log10-parameters
log10_params = np.log10(p_true)
J = model.jac_signal_logparams_np(log10_params, p0.V_abs)
print(f"Jacobian shape: {J.shape}")
```

## Citation

If you use this framework in your scientific work, please cite:

```bibtex
@software{difflc2026,
  author = {Krutoy, Nikita},
  title = {DiffLC: Differentiable Landau-de Gennes Simulation Suite},
  year = {2026},
  url = {https://github.com/Kali-Uga/DiffLC-Differentiable-Liquid-Crystals-}
}
```
