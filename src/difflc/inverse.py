"""Multi-cell TRF inverse solver — JAX backend.

Recovers (K11, K22, K33, gamma1, W) from normalised Stokes measurements.
Parameters are optimised in log10 space with ±1 decade bounds.
Supports multiple random starts and JAX or finite-difference Jacobians.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares


PARAM_NAMES = ["K11", "K22", "K33", "gamma1", "W"]
PARAM_UNITS = ["N", "N", "N", "Pa·s", "J/m²"]


def _build_residual(
    models,
    protocols,
    target_flat,
    noise_std,
    *,
    dt,
    T_on,
    T_off,
    T_eq,
    record_every,
    jac_mode,
):
    """Build residual and Jacobian callables for least_squares."""

    n_params = 5

    def signal_all(log10_params):
        parts = []
        for p in protocols:
            m = models[p.name]
            s = m.signal_logparams_np(
                log10_params,
                p.V_abs,
                dt_=dt,
                T_on_=T_on,
                T_off_=T_off,
                T_eq_=T_eq,
                rec_=record_every,
            )
            parts.append(s)
        return np.concatenate(parts)

    def residual_np(lp_np):
        try:
            res = (signal_all(lp_np) - target_flat) / noise_std
            if not np.all(np.isfinite(res)):
                # Non-finite signal (typically explicit-L2/L3 blow-up or a bad
                # region). We still return a large finite residual so TRF can back
                # out, but do NOT do it silently — silent 1e6 plateaus look like
                # convergence stalls. See solve_inverse(strict_stability=...).
                warnings.warn(
                    f"non-finite residual at log10(params)={np.round(lp_np, 3)} "
                    "→ masked to 1e6 (check dt/Nz stability or bounds).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                res = np.nan_to_num(res, nan=1e6, posinf=1e6, neginf=-1e6)
            return res
        except Exception as exc:
            warnings.warn(f"residual evaluation failed → masked to 1e6: {exc}",
                          RuntimeWarning, stacklevel=2)
            return np.full_like(target_flat, 1e6, dtype=float)

    if jac_mode == "jax":

        def jac_np(lp_np):
            try:
                rows = []
                for p in protocols:
                    m = models[p.name]
                    J_p = m.jac_signal_logparams_np(
                        lp_np,
                        p.V_abs,
                        dt_=dt,
                        T_on_=T_on,
                        T_off_=T_off,
                        T_eq_=T_eq,
                        rec_=record_every,
                    )
                    rows.append(J_p)
                J = np.concatenate(rows, axis=0) / noise_std
                MAX_JAC = 1e6
                return np.clip(
                    np.nan_to_num(J, nan=0.0, posinf=MAX_JAC, neginf=-MAX_JAC),
                    -MAX_JAC,
                    MAX_JAC,
                )
            except Exception as exc:
                warnings.warn(f"Jacobian evaluation failed → zero matrix: {exc}",
                              RuntimeWarning, stacklevel=2)
                return np.zeros((target_flat.size, n_params), dtype=float)
    else:

        def jac_np(lp_np):
            eps = 1e-4
            f0 = residual_np(lp_np)
            J = np.zeros((f0.size, n_params), dtype=float)
            for k in range(n_params):
                e = np.zeros(n_params)
                e[k] = eps
                fp = residual_np(lp_np + e)
                fm = residual_np(lp_np - e)
                J[:, k] = (fp - fm) / (2.0 * eps)
            return J

    return residual_np, jac_np


def solve_inverse(
    models,
    protocols,
    target_flat,
    p_true,
    noise_std,
    *,
    n_starts=3,
    max_nfev=100,
    dt=5e-4,
    T_on=0.300,
    T_off=0.200,
    T_eq=0.100,
    record_every=8,
    jac_mode="jax",
    first_start_radius=0.15,
    random_start_radius=0.15,
    seed=42,
    x_scale="jac",
    loss="linear",
    xtol=1e-8,
    ftol=1e-8,
    gtol=1e-8,
    strict_stability=True,
):
    """Multi-start TRF inverse recovery.

    Parameters
    ----------
    models       : dict name → model namespace
    protocols    : list of Protocol
    target_flat  : 1-D noisy Stokes measurement
    p_true       : (5,) true parameter values (for bounds and diagnostics)
    noise_std    : float
    n_starts     : number of optimisation starts
    jac_mode     : "jax" or "fd"
    x_scale      : TRF variable scaling, default "jac" (recommended for the
                   ill-conditioned fringe Jacobian; the old 1.0 stalls).
    loss         : scipy loss, default "linear"; use "soft_l1" for robustness.
    strict_stability : if True (default), raise instead of silently fitting NaN
                   when dt exceeds the explicit-stability limit at p_true.

    Returns
    -------
    dict with keys: ok, x (recovered params), err_pct, result (scipy result),
                    all_results (one per start)
    """
    log_true = np.log10(np.asarray(p_true, dtype=float))
    LOG_LO = log_true - 1.0
    LOG_HI = log_true + 1.0

    # Fail fast in the fitting context: an unstable (dt, Nz) turns every residual
    # into masked 1e6, which looks like a stalled optimiser. Forward runs keep the
    # softer RuntimeWarning; here we raise. (Set strict_stability=False to opt out.)
    if strict_stability:
        p_true_arr = np.asarray(p_true, dtype=float)
        for p in protocols:
            m = models[p.name]
            get_lim = getattr(m, "stability_dt_max", None)
            if get_lim is not None and dt > get_lim(p_true_arr):
                raise ValueError(
                    f"dt={dt:.2e}s exceeds the explicit-stability limit "
                    f"~{get_lim(p_true_arr):.2e}s for cell '{p.name}' at p_true; "
                    "the fit would evaluate NaN residuals. Reduce dt (or Nz), or "
                    "pass strict_stability=False to proceed anyway."
                )

    rng = np.random.default_rng(seed)

    residual_np, jac_np = _build_residual(
        models,
        protocols,
        target_flat,
        noise_std,
        dt=dt,
        T_on=T_on,
        T_off=T_off,
        T_eq=T_eq,
        record_every=record_every,
        jac_mode=jac_mode,
    )

    best_cost = np.inf
    best_result = None
    all_results = []

    for i_start in range(n_starts):
        if i_start == 0:
            x0 = log_true + rng.uniform(
                -first_start_radius, first_start_radius, size=len(log_true)
            )
        else:
            x0 = log_true + rng.uniform(
                -random_start_radius, random_start_radius, size=len(log_true)
            )
        x0 = np.clip(x0, LOG_LO, LOG_HI)

        try:
            result = least_squares(
                residual_np,
                x0=x0,
                jac=jac_np,
                method="trf",
                bounds=(LOG_LO, LOG_HI),
                xtol=xtol,
                ftol=ftol,
                gtol=gtol,
                max_nfev=max_nfev,
                x_scale=x_scale,
                loss=loss,
            )
            cost = float(result.cost)
            all_results.append({"start": i_start, "result": result, "cost": cost})
            if cost < best_cost:
                best_cost = cost
                best_result = result
        except Exception as exc:
            all_results.append({"start": i_start, "error": str(exc), "cost": np.inf})

    if best_result is None:
        return dict(
            ok=False,
            x=np.full(len(p_true), np.nan),
            err_pct=np.full(len(p_true), np.nan),
            result=None,
            all_results=all_results,
        )

    p_rec = 10.0**best_result.x
    err_pct = np.abs(p_rec - p_true) / np.abs(p_true) * 100.0

    return dict(
        ok=True,
        x=p_rec,
        err_pct=err_pct,
        result=best_result,
        all_results=all_results,
    )


def print_recovery_table(result_dict, p_true, param_names=None, param_units=None):
    """Pretty-print parameter recovery results."""
    if param_names is None:
        param_names = PARAM_NAMES
    if param_units is None:
        param_units = PARAM_UNITS

    print(f"{'Param':>8}  {'True':>14}  {'Recovered':>14}  {'Error%':>8}  Unit")
    print("-" * 60)
    p_rec = result_dict["x"]
    err = result_dict["err_pct"]
    for name, pt, pr, er, unit in zip(param_names, p_true, p_rec, err, param_units):
        print(f"{name:>8}  {pt:>14.6e}  {pr:>14.6e}  {er:>7.2f}%  {unit}")
