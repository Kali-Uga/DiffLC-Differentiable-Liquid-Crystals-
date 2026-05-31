"""DiffLC public API — v0.2 (JAX backend, E7, Berreman optics).

Quick start::

    from difflc import default_cfg, default_cells, build_protocols, make_model
    import numpy as np

    cfg = default_cfg()
    cells = default_cells()
    protocols = build_protocols(cells, cfg)

    # Build one model per protocol (cell)
    models = {p.name: make_model(cfg, p.cell) for p in protocols}

    p_true = np.array([cfg.K11, cfg.K22, cfg.K33, cfg.gamma1, cfg.W])
    out = models[protocols[0].name].run_protocol_np(p_true, protocols[0].V_abs)
    print(out["stokes"].shape)   # (n_rec, n_wl, n_theta, n_pol, 4)
"""

from .utils import (
    E7Config,
    CellSpec,
    Protocol,
    TimingConfig,
    K_to_L,
    threshold_voltage,
    build_protocols,
    default_cfg,
    default_cells,
    default_timing,
    WAVELENGTHS_NM,
    INCIDENCE_DEG,
    DEFAULT_INPUT_POLS,
    POL_LABELS,
    jones_linear,
    jones_circular,
)

from .qtensor import (
    director_from_angles,
    Q_from_director,
    ang2Q,
    Q2v,
    v2Q,
    proj_ST,
    extract_director,
    scalar_order_from_eigvals,
    biaxiality_from_eigvals,
    angles_from_Q,
    angles_from_v,
)

from .optics import (
    eps_from_Q,
    berreman_B_from_eps,
    berreman_layer_transfer,
    stokes_from_jones2,
    stokes_oblique,
    stokes_normal,
    all_stokes,
)

from .solver import make_model

from .oed import (
    compute_fim,
    fim_diagnostics,
    run_campaign,
)

from .inverse import (
    solve_inverse,
    print_recovery_table,
    PARAM_NAMES,
    PARAM_UNITS,
)

__version__ = "0.2.0"

__all__ = [
    # config
    "E7Config",
    "CellSpec",
    "Protocol",
    "TimingConfig",
    "K_to_L",
    "threshold_voltage",
    "build_protocols",
    "default_cfg",
    "default_cells",
    "default_timing",
    "WAVELENGTHS_NM",
    "INCIDENCE_DEG",
    "DEFAULT_INPUT_POLS",
    "POL_LABELS",
    "jones_linear",
    "jones_circular",
    # qtensor
    "director_from_angles",
    "Q_from_director",
    "ang2Q",
    "Q2v",
    "v2Q",
    "proj_ST",
    "extract_director",
    "scalar_order_from_eigvals",
    "biaxiality_from_eigvals",
    "angles_from_Q",
    "angles_from_v",
    # optics
    "eps_from_Q",
    "berreman_B_from_eps",
    "berreman_layer_transfer",
    "stokes_from_jones2",
    "stokes_oblique",
    "stokes_normal",
    "all_stokes",
    # solver
    "make_model",
    # oed
    "compute_fim",
    "fim_diagnostics",
    "run_campaign",
    # inverse
    "solve_inverse",
    "print_recovery_table",
    "PARAM_NAMES",
    "PARAM_UNITS",
    # meta
    "__version__",
]
