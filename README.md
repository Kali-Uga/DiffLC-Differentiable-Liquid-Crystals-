# DiffLC: A Differentiable Landau-de Gennes Framework for Liquid Crystal Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

**DiffLC** is a high-performance numerical engine designed for the simulation of nematic liquid crystal (LC) dynamics and the solution of complex inverse problems. Built on the **PyTorch** autograd engine, DiffLC treats the entire physical pipeline—from $Q$-tensor relaxation to differential Jones calculus—as a differentiable operator.

This framework is specifically engineered to bridge the gap between classical continuum mechanics and modern machine learning workflows, enabling high-fidelity parameter recovery and Bayesian Optimal Experimental Design (OED).

## Core Capabilities

- **Differentiable Physics:** Full support for first- and second-order parametric sensitivities using `torch.func` (Functional API), bypassing the inaccuracies of finite-difference methods.
- **Landau-de Gennes (LdG) Dynamics:** Implements a multi-constant $Q$-tensor expansion ($L_1, L_2, L_3$) to accurately model splay, twist, and bend elasticities, including weak surface anchoring effects.
- **Advanced Optical Modeling:** Features a Sliced Differential Jones Matrix engine for precise calculation of optical transmission (CPI) in TN-cells with non-trivial director gradients.
- **Parametric Identification:** Optimized for information-theoretic analysis, allowing researchers to quantify the "sloppiness" of physical constants and identify well-constrained parameter manifolds.

## Technical Architecture

The codebase is modularized to support both direct physical simulations and integration into larger AI-driven research pipelines:

- `src/difflc/solver.py`: Semi-implicit time-stepping LdG engine.
- `src/difflc/qtensor.py`: Gauge-invariant $Q$-tensor operations and projections.
- `src/difflc/optics.py`: Differentiable optical response models.
- `src/difflc/oed.py`: Bayesian utilities for maximizing Information Gain (D-optimality).
- `src/difflc/inverse.py`: Non-linear parameter recovery via Gradient-enhanced Trust Region methods.

## Installation

```bash
git clone https://github.com/Kali-Uga/difflc.git
cd difflc
pip install -e .
```

## AI Integration & Research Utility

DiffLC is designed to be "AI-native." By providing exact Jacobians of physical systems, it facilitates:
1. **Hybrid Physics-Neural Networks:** Using the LC solver as a differentiable layer within deep learning architectures.
2. **LLM-Guided Discovery:** Leveraging OpenAI's Codex/GPT models to automate experimental protocol synthesis and interpret Bayesian sensitivity analysis in natural language.

## Quick Start

```python
import torch
from difflc import default_params, run_dc_protocol_diff

# Initialize physical priors for 5CB
params = default_params()

# Compute a differentiable forward pass
output = run_dc_protocol_diff(
    params.L1, params.L2, params.L3, 
    V_factor=3.0, params=params
)

# Access the gradient path back to material viscosity
transmission = output["I_cross"][-1]
transmission.backward()
print(f"Sensitivity (dI/d_gamma): {params.gamma1.grad}")
```

## Citation

If you use this framework in your scientific work, please cite:

```bibtex
@software{difflc2026,
  author = {Krutoy, Nikita},
  title = {DiffLC: Differentiable Landau-de Gennes Simulation Suite},
  year = {2026},
  url = {[https://github.com/Kali-Uga/difflc](https://github.com/Kali-Uga/DiffLC-Differentiable-Liquid-Crystals-)}
}
