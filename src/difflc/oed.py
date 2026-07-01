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


def fim_diagnostics(FIM, p_true=None, param_names=None):
    """Compute CRLB, parameter correlation, and D-optimality from a FIM.

    IMPORTANT — units. The FIM is built from the Jacobian of the signal w.r.t.
    **log10 parameters** (see ``compute_fim`` → ``jac_signal_logparams_np``).
    Hence ``diag(F_inv)`` are variances in *decades²*, and ``sqrt`` gives the
    standard deviation of ``log10(p)``, i.e. σ_log. The relative (percent)
    uncertainty of the *linear* parameter is therefore

        rel% = 100 · (10**σ_log − 1)          (exact, multiplicative)
             ≈ 100 · ln(10) · σ_log            (small-σ limit)

    The previous implementation divided σ_log by the linear ``p_true`` — mixing
    decades with newtons — which is dimensionally meaningless. ``p_true`` is no
    longer required (relative error is intrinsic to the log parametrisation);
    it is kept only for backward-compatible call signatures.

    The parameter **correlation** matrix is the normalised **covariance**
    (F_inv), not the normalised FIM — these differ for ill-conditioned FIMs.

    Returns dict: crlb_pct, sigma_log, corr (from cov), corr_fim (legacy),
    logdet, FIM, F_inv, cond, param_names.
    """
    n = FIM.shape[0]
    if param_names is None:
        param_names = [f"p{i}" for i in range(n)]

    # Regularise relative to the FIM scale (trace/n), not by an absolute 1e-20.
    scale = float(np.trace(FIM)) / n if np.trace(FIM) > 0 else 1.0
    try:
        F_inv = np.linalg.inv(FIM + 1e-12 * scale * np.eye(n))
        sigma_log = np.sqrt(np.abs(np.diag(F_inv)))
        crlb_pct = 100.0 * (10.0**sigma_log - 1.0)  # relative error of linear param
        d_cov = np.sqrt(np.abs(np.diag(F_inv)))
        corr = F_inv / np.outer(d_cov, d_cov)        # correlation of estimates
    except np.linalg.LinAlgError:
        F_inv = None
        sigma_log = np.full(n, np.nan)
        crlb_pct = np.full(n, np.nan)
        corr = np.full((n, n), np.nan)

    d_fim = np.sqrt(np.abs(np.diag(FIM)) + 1e-30)
    corr_fim = FIM / np.outer(d_fim, d_fim)          # legacy (normalised FIM)

    sign, logdet = np.linalg.slogdet(FIM)
    logdet = -np.inf if sign <= 0 else float(logdet)
    cond = float(np.linalg.cond(FIM))

    return dict(
        FIM=FIM,
        F_inv=F_inv,
        sigma_log=sigma_log,
        crlb_pct=crlb_pct,
        corr=corr,
        corr_fim=corr_fim,
        logdet=logdet,
        cond=cond,
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
