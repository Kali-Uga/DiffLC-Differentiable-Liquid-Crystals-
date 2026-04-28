# DiffLC

DiffLC is a compact, publication-oriented research codebase for twisted-nematic 5CB simulation, built around a differentiable Landau-de Gennes solver and sliced Jones optics.

The codebase is organized so that the reusable physics lives in `src/difflc/`, while the notebook layer remains a thin demonstration surface.

## What it does

- Converts between director fields and Q-tensor representations.
- Simulates a 1D twisted-nematic cell with weak anchoring.
- Computes optical transmission with a differential Jones model.
- Provides optional experiment-design and inverse-solver helpers.

## Project layout

- `src/difflc/utils.py`: physical parameters and helper functions.
- `src/difflc/qtensor.py`: Q-tensor conversion utilities.
- `src/difflc/solver.py`: differentiable solver and direct protocol runner.
- `src/difflc/optics.py`: Jones matrices and transmission helpers.
- `src/difflc/oed.py`: optional Bayesian OED helpers.
- `src/difflc/inverse.py`: optional TRF inverse recovery helpers.
- `tests/`: smoke tests for the core public API.
- `notebooks/5cb_sim_demo.ipynb`: demo notebook that imports the package.

## Install

For the core package:

```bash
pip install -e .
```

For development and research extras:

```bash
pip install -e '.[dev,research]'
```

## Run tests

```bash
pytest
```

## Quickstart

```python
from difflc import default_params, run_dc_protocol_diff

params = default_params()
run = run_dc_protocol_diff(
    params.L1,
    params.L2,
    params.L3,
    params.gamma1,
    params.W_surf,
    V_factor=3.0,
    T_on=0.8,
    T_off=0.0,
    dt_fw=1e-3,
    params=params,
)

print(run["I_cross"].shape)
```

## Reproducibility notes

- The default parameters reproduce the canonical 5CB setup used in the notebook.
- The package is pure Python, with PyTorch as the main numerical dependency.
- The test suite is intentionally lightweight so it can run in CI and on contributor machines.

## References

The implementation follows the same physical setup as the original notebook inspired by TN 5CB literature, with a focus on weak anchoring, 1D geometry, and a 90-degree twist cell.
