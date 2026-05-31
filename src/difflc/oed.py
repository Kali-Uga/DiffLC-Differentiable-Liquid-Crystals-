"""Fisher Information Matrix and Optimal Experimental Design utilities.

Uses JAX auto-differentiation (jacfwd) for exact Jacobians.
Supports multi-cell joint campaigns.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# FIM computation
# ---------------------------------------------------------------------------


def compute_fim(
    models,
    protocols,
    log10_p_true,
    noise_std,
    *,
    dt=5e-4,
    T_on=0.300,
    T_off=0.200,
    T_eq=0.100,
    record_every=8,
    use_jax_jac=True,
):
    """Compute joint Fisher Information Matrix (FIM) over all cells.

    Parameters
    ----------
    models       : dict mapping protocol.name → model SimpleNamespace
    protocols    : list of Protocol
    log10_p_true : (5,) array — log10 of true parameters [K11,K22,K33,gamma1,W]
    noise_std    : float — assumed measurement noise std
    use_jax_jac  : use JAX jacfwd (True) or finite differences (False)

    Returns
    -------
    FIM : (5, 5) Fisher Information Matrix
    J_dict : dict of Jacobians per protocol
    """
    n_params = len(log10_p_true)
    FIM = np.zeros((n_params, n_params))
    J_dict = {}

    for p in protocols:
        m = models[p.name]
        V_abs = p.V_abs

        if use_jax_jac:
            J = m.jac_signal_logparams_np(
                log10_p_true,
                V_abs,
                dt_=dt,
                T_on_=T_on,
                T_off_=T_off,
                T_eq_=T_eq,
                rec_=record_every,
            )
        else:
            J = m.signal_jac_fd_np(
                log10_p_true,
                V_abs,
                dt_=dt,
                T_on_=T_on,
                T_off_=T_off,
                T_eq_=T_eq,
                rec_=record_every,
            )

        Jw = J / noise_std
        FIM += Jw.T @ Jw
        J_dict[p.name] = J

    return FIM, J_dict


def fim_diagnostics(FIM, p_true, param_names=None):
    """Compute CRLB, correlation matrix, and D-optimality from FIM.

    Returns dict with keys: crlb_pct, corr, logdet, FIM.
    """
    n = FIM.shape[0]
    if param_names is None:
        param_names = [f"p{i}" for i in range(n)]

    try:
        FIM_reg = FIM + 1e-20 * np.eye(n)
        F_inv = np.linalg.inv(FIM_reg)
        crlb = 100.0 * np.sqrt(np.abs(np.diag(F_inv))) / np.abs(p_true)
    except np.linalg.LinAlgError:
        crlb = np.full(n, np.nan)
        F_inv = None

    d_ = np.sqrt(np.diag(FIM) + 1e-30)
    corr = FIM / np.outer(d_, d_)

    sign, logdet = np.linalg.slogdet(FIM)
    logdet = -np.inf if sign <= 0 else float(logdet)

    return dict(
        FIM=FIM,
        corr=corr,
        crlb_pct=crlb,
        logdet=logdet,
        F_inv=F_inv,
        param_names=param_names,
    )


# ---------------------------------------------------------------------------
# Campaign runner helpers
# ---------------------------------------------------------------------------


def run_campaign(
    models,
    protocols,
    params_K,
    *,
    noise_std=1e-2,
    seed=42,
    dt=5e-4,
    T_on=0.300,
    T_off=0.200,
    T_eq=0.100,
    record_every=8,
):
    """Run forward simulation over all protocols and add noise.

    Returns
    -------
    dict with keys:
      clean      : list of (n_rec, n_wl, n_theta, n_pol, 3) normalised Stokes
      noisy      : same + Gaussian noise
      runs       : list of raw run dicts
      target_flat: 1-D noisy concatenation (for inverse)
      clean_flat : 1-D clean concatenation
    """
    rng = np.random.default_rng(seed)
    clean = []
    noisy = []
    runs = []

    for p in protocols:
        m = models[p.name]
        out = m.run_protocol_np(
            params_K,
            p.V_abs,
            dt_=dt,
            T_on_=T_on,
            T_off_=T_off,
            T_eq_=T_eq,
            rec_=record_every,
        )
        st = out["stokes"]
        S0_comp = st[..., 0:1]
        sn = st[..., 1:4] / np.clip(S0_comp, 1e-30, None)
        clean.append(sn)
        noisy.append(sn + rng.normal(scale=noise_std, size=sn.shape))
        runs.append(out)

    return dict(
        clean=clean,
        noisy=noisy,
        runs=runs,
        target_flat=np.concatenate([x.ravel() for x in noisy]),
        clean_flat=np.concatenate([x.ravel() for x in clean]),
        noise_std=noise_std,
        time_ms=runs[0]["time"] * 1e3,
    )
