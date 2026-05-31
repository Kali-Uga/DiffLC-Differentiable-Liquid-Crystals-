"""Differentiable LdG solver — JAX backend.

Full Q-tensor dynamics with:
- Tridiagonal Thomas algorithm for L1 + anchoring (implicit)
- L2, L3, electric field handled explicitly
- Landau-de Gennes thermotropic bulk via local implicit Euler + Newton
- Scalar order S evolves freely (no projection to fixed-S)

Entry point: ``make_model(cfg, cell)`` returns a SimpleNamespace of
JIT-compiled functions ready for forward simulation and JAX autodiff.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax

from .qtensor import (
    ang2Q,
    Q2v,
    v2Q,
    proj_ST,
    angles_from_Q,
)
from .optics import all_stokes
from .utils import E7Config, CellSpec, K_to_L

jax.config.update("jax_enable_x64", True)

# 5-component identity basis — used for analytic Jacobian of bulk term
_I5 = jnp.eye(5, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Bulk (thermotropic) energy: gradient and analytic Jacobian
# ---------------------------------------------------------------------------


def _bulk_gradient_Q(Q, bulk_a, bulk_b, bulk_c):
    """Symmetric-traceless gradient of f_bulk w.r.t. Q.

    f = a tr(Q²) + (2b/3) tr(Q³) + (c/2)[tr(Q²)]²
    δf/δQ = 2a Q + 2b (Q² − I tr(Q²)/3) + 2c tr(Q²) Q  [projected]
    """
    I3 = jnp.eye(3, dtype=jnp.float64)
    tr2 = jnp.sum(Q * Q, axis=(-2, -1))[..., None, None]
    Q2 = Q @ Q
    Q2_st = Q2 - (tr2 / 3.0) * I3
    dF = 2.0 * bulk_a * Q + 2.0 * bulk_b * Q2_st + 2.0 * bulk_c * tr2 * Q
    return proj_ST(dF)


def _bulk_gradient_v(q, bulk_a, bulk_b, bulk_c):
    return Q2v(_bulk_gradient_Q(v2Q(q), bulk_a, bulk_b, bulk_c))


def _bulk_gradient_jac(q, bulk_a, bulk_b, bulk_c):
    """Analytic 5×5 Jacobian of _bulk_gradient_v at point q."""
    I3 = jnp.eye(3, dtype=jnp.float64)
    Q = v2Q(q)
    tr2 = jnp.sum(Q * Q)

    def one_col(dQ):
        dtr2 = 2.0 * jnp.sum(Q * dQ)
        dQ2_st = Q @ dQ + dQ @ Q - (dtr2 / 3.0) * I3
        dG = (
            2.0 * bulk_a * dQ
            + 2.0 * bulk_b * dQ2_st
            + 2.0 * bulk_c * (dtr2 * Q + tr2 * dQ)
        )
        return Q2v(proj_ST(dG))

    basis_Q = v2Q(_I5)
    return jax.vmap(one_col)(basis_Q).T


def _bulk_implicit_update(v_in, dt, mu, bulk_a, bulk_b, bulk_c, n_newton=5):
    """Local implicit Euler at every grid node: Newton solve.

    q_{n+1} + dt·μ·∂f_bulk/∂q(q_{n+1}) = q_star = v_in
    """
    I5 = jnp.eye(5, dtype=jnp.float64)

    def update_node(q0):
        def newton_body(_i, q):
            g = _bulk_gradient_v(q, bulk_a, bulk_b, bulk_c)
            J = I5 + dt * mu * _bulk_gradient_jac(q, bulk_a, bulk_b, bulk_c)
            r = q + dt * mu * g - q0
            dq = jnp.linalg.solve(J, r)
            return q - dq

        return lax.fori_loop(0, n_newton, newton_body, q0)

    return jax.vmap(update_node)(v_in)


# ---------------------------------------------------------------------------
# Elastic + electric molecular field
# ---------------------------------------------------------------------------


def _elastic_electric_field(Q, E, L1, L2, L3, EPS_0, deps, S0, dz):
    """Compute elastic + electric molecular field h and second derivative Qpp.

    Returns (h, Qpp) both shape (Nz, 3, 3).
    Boundary nodes are zeroed out via bulk_mask.
    """
    Nz = Q.shape[0]
    zero = jnp.zeros((1, 3, 3), dtype=jnp.float64)
    mask_col2 = jnp.zeros((3, 3), dtype=jnp.float64).at[:, 2].set(1.0)
    mask_row2 = jnp.zeros((3, 3), dtype=jnp.float64).at[2, :].set(1.0)
    mask_33 = jnp.zeros((3, 3), dtype=jnp.float64).at[2, 2].set(1.0)

    Qpp = jnp.concatenate(
        [
            zero,
            (Q[2:] + Q[:-2] - 2.0 * Q[1:-1]) / dz**2,
            zero,
        ],
        axis=0,
    )
    Qp = jnp.concatenate(
        [
            zero,
            (Q[2:] - Q[:-2]) / (2.0 * dz),
            zero,
        ],
        axis=0,
    )

    h_L1 = L1 * Qpp
    h_L2 = 0.5 * L2 * Qpp[:, :, 2][:, :, None] * mask_col2
    h_L2 = h_L2 + 0.5 * L2 * Qpp[:, 2, :][:, None, :] * mask_row2

    Q33 = Q[:, 2, 2][:, None, None]
    Q33p = Qp[:, 2, 2][:, None, None]
    gsq = jnp.sum(Qp**2, axis=(-2, -1))[:, None, None]
    h_L3 = L3 * Q33 * Qpp + L3 * Q33p * Qp
    h_L3_corr = -0.5 * L3 * gsq * mask_33

    h_E = 0.5 * EPS_0 * (deps / S0) * E**2 * mask_33

    bulk_mask = jnp.ones((Nz, 1, 1), dtype=jnp.float64).at[0].set(0.0).at[-1].set(0.0)
    h = (h_L1 + h_L2 + h_L3 + h_L3_corr + h_E) * bulk_mask
    return h, Qpp


# ---------------------------------------------------------------------------
# Tridiagonal coefficients and Thomas solve
# ---------------------------------------------------------------------------


def _tridiagonal_coefficients(dt, L1, gamma_Q, W, dz, Nz):
    """Build tridiagonal system coefficients for implicit L1 + anchoring.

    Returns (lower, diag, upper, c_w) all as 1-D JAX arrays.
    """
    c_el = dt * gamma_Q * L1 / dz**2
    c_w = dt * gamma_Q * W / (0.5 * dz)

    diag = jnp.ones((Nz,), dtype=jnp.float64) * (1.0 + 2.0 * c_el)
    diag = diag.at[0].set(1.0 + c_w + 2.0 * c_el)
    diag = diag.at[-1].set(1.0 + c_w + 2.0 * c_el)

    lower = jnp.ones((Nz - 1,), dtype=jnp.float64) * (-c_el)
    upper = jnp.ones((Nz - 1,), dtype=jnp.float64) * (-c_el)
    # Ghost-node mirroring at boundaries
    upper = upper.at[0].set(-2.0 * c_el)
    lower = lower.at[-1].set(-2.0 * c_el)

    return lower, diag, upper, c_w


def _solve_tridiagonal(lower, diag, upper, rhs):
    """Thomas algorithm for tridiagonal system (Nz × 5) — O(N).

    Solves A x = rhs where A is tridiagonal.
    rhs shape: (Nz, 5).
    """
    den0 = diag[0]
    cp0 = upper[0] / den0
    dp0 = rhs[0] / den0

    def fwd(carry, elems):
        cp_prev, dp_prev = carry
        lo, di, up, ri = elems
        den = di - lo * cp_prev
        cp = up / den
        dp = (ri - lo * dp_prev) / den
        return (cp, dp), (cp, dp)

    (cp_prev, dp_prev), (cp_mid, dp_mid) = lax.scan(
        fwd,
        (cp0, dp0),
        (lower[:-1], diag[1:-1], upper[1:], rhs[1:-1]),
    )
    den_last = diag[-1] - lower[-1] * cp_prev
    dp_last = (rhs[-1] - lower[-1] * dp_prev) / den_last

    cp_all = jnp.concatenate([jnp.atleast_1d(cp0), cp_mid], axis=0)
    dp_all = jnp.concatenate([dp0[None, :], dp_mid, dp_last[None, :]], axis=0)

    def bwd(x_next, elems):
        cp, dp = elems
        x = dp - cp * x_next
        return x, x

    _u, x_rev = lax.scan(bwd, dp_last, (cp_all[::-1], dp_all[:-1][::-1]))
    return jnp.concatenate([x_rev[::-1], dp_last[None, :]], axis=0)


# ---------------------------------------------------------------------------
# Single time step
# ---------------------------------------------------------------------------


def _step(
    v,
    lower,
    diag,
    upper,
    c_w,
    dt,
    E,
    params_K,
    qsb_v,
    qst_v,
    dz,
    EPS_0,
    deps,
    S0,
    bulk_a,
    bulk_b,
    bulk_c,
):
    """One semi-implicit time step.

    1. Compute explicit L2 + L3 + electric molecular field.
    2. Build RHS including implicit L1 + anchoring boundary terms.
    3. Thomas solve for elastic update.
    4. Local implicit Newton for bulk thermotropic relaxation.
    """
    K11, K22, K33, gamma1, W = params_K
    L1, L2, L3 = K_to_L(K11, K22, K33, S0)
    mu = S0 / gamma1  # Q-tensor mobility = 1/γ_Q

    Q = v2Q(v)
    h_el, Qpp = _elastic_electric_field(Q, E, L1, L2, L3, EPS_0, deps, S0, dz)
    # Explicit part excludes L1 (handled implicitly in Thomas)
    h_explicit = proj_ST(h_el) - proj_ST(L1 * Qpp)
    rhs_full = v + dt * mu * Q2v(h_explicit)

    # Anchor boundary nodes
    rhs = rhs_full.at[0].set(v[0] + c_w * qsb_v)
    rhs = rhs.at[-1].set(v[-1] + c_w * qst_v)

    v_elastic = _solve_tridiagonal(lower, diag, upper, rhs)
    return _bulk_implicit_update(v_elastic, dt, mu, bulk_a, bulk_b, bulk_c)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def make_model(
    cfg: E7Config,
    cell: CellSpec,
    *,
    wavelengths_nm=None,
    incidence_deg=None,
    input_pols=None,
    dt=5e-4,
    record_every=8,
    T_on=0.300,
    T_off=0.200,
    T_eq=0.100,
):
    """Build a JAX model namespace for a single LC cell.

    Returns a SimpleNamespace with:
    - ``run_protocol_np``      — forward simulation → dict of numpy arrays
    - ``run_waveform_np``      — arbitrary waveform forward simulation
    - ``signal_logparams_np``  — normalised Stokes as 1-D array (for inverse)
    - ``jac_signal_logparams_np`` — JAX Jacobian of signal_logparams_np
    - ``counts``               — helper: (n_on_blocks, n_off_blocks, n_eq)
    - ``dz``, ``z_um``

    Parameters
    ----------
    cfg           : E7Config
    cell          : CellSpec
    wavelengths_nm: sequence of floats, default (450, 532, 589, 642, 700)
    incidence_deg : sequence of floats, default (0, 35)
    input_pols    : (n_pol, 2) complex array, default 4 standard polarisations
    """
    from .utils import WAVELENGTHS_NM, INCIDENCE_DEG, DEFAULT_INPUT_POLS

    if wavelengths_nm is None:
        wavelengths_nm = WAVELENGTHS_NM
    if incidence_deg is None:
        incidence_deg = INCIDENCE_DEG
    if input_pols is None:
        input_pols = DEFAULT_INPUT_POLS

    # --- geometry ---
    Nz = cfg.Nz
    d_cell = float(cell.d_cell)
    dz = d_cell / (Nz - 1)
    z_axis = jnp.linspace(0.0, d_cell, Nz, dtype=jnp.float64)

    # --- material derived ---
    S0 = cfg.S0
    bulk_a = cfg.bulk_a
    bulk_b = cfg.bulk_b
    bulk_c = cfg.bulk_c
    EPS_0 = cfg.EPS_0
    deps = cfg.deps
    no = cfg.no
    ne = cfg.ne

    # --- observation grid (JAX arrays) ---
    _wls_m = jnp.asarray([w * 1e-9 for w in wavelengths_nm], dtype=jnp.float64)
    _thetas = jnp.asarray([math.radians(t) for t in incidence_deg], dtype=jnp.float64)
    _pols = jnp.asarray(input_pols, dtype=jnp.complex128)

    # --- step helpers (these capture material constants) ---

    def _v0_init():
        theta0 = jnp.full((Nz,), cfg.pretilt_rad, dtype=jnp.float64)
        phi0 = (z_axis / d_cell) * math.radians(cell.twist_deg)
        return Q2v(ang2Q(theta0, phi0, S_val=S0))

    def _boundary_vectors():
        twist_rad = math.radians(cell.twist_deg)
        Qsb = ang2Q(jnp.array([cfg.pretilt_rad]), jnp.array([0.0]), S_val=S0)[0]
        Qst = ang2Q(jnp.array([cfg.pretilt_rad]), jnp.array([twist_rad]), S_val=S0)[0]
        return Q2v(Qsb), Q2v(Qst)

    def _make_tridiag(dt_val, params_K):
        K11, K22, K33, gamma1, W = params_K
        L1, _, _ = K_to_L(K11, K22, K33, S0)
        mu = S0 / gamma1
        return _tridiagonal_coefficients(dt_val, L1, mu, W, dz, Nz)

    def _step_fn(v, lower, diag, upper, c_w, dt_val, E, params_K, qsb_v, qst_v):
        return _step(
            v,
            lower,
            diag,
            upper,
            c_w,
            dt_val,
            E,
            params_K,
            qsb_v,
            qst_v,
            dz,
            EPS_0,
            deps,
            S0,
            bulk_a,
            bulk_b,
            bulk_c,
        )

    def _all_stokes_fn(Q_field):
        return all_stokes(Q_field, _wls_m, _thetas, _pols, dz, no=no, ne=ne, S0=S0)

    # --- inner protocol (JAX-purefunction, JIT-able) ---

    def _protocol_recorded(
        params_K, V_abs, dt_val, n_on_blocks, n_off_blocks, n_eq, rec_every
    ):
        lower, diag, upper, c_w = _make_tridiag(dt_val, params_K)
        qsb_v, qst_v = _boundary_vectors()
        v = _v0_init()

        # Pre-equilibration at zero field
        def eq_step(v_curr, _):
            return _step_fn(
                v_curr, lower, diag, upper, c_w, dt_val, 0.0, params_K, qsb_v, qst_v
            ), None

        v, _ = lax.scan(eq_step, v, None, length=n_eq)

        def one_record_block(v_curr, E_block):
            def body(vc, _):
                vn = _step_fn(
                    vc, lower, diag, upper, c_w, dt_val, E_block, params_K, qsb_v, qst_v
                )
                return vn, None

            v_end, _ = lax.scan(body, v_curr, None, length=rec_every)
            Q_end = v2Q(v_end)
            st = _all_stokes_fn(Q_end)
            theta, phi, S_eff, beta = angles_from_Q(Q_end)
            diag_out = jnp.stack([theta, phi, S_eff, beta], axis=-1)
            return v_end, (st, diag_out, v_end)

        E_on = V_abs / d_cell
        Es_on = jnp.full((n_on_blocks,), E_on, dtype=jnp.float64)
        Es_off = jnp.zeros((n_off_blocks,), dtype=jnp.float64)
        Es_all = jnp.concatenate([Es_on, Es_off])

        v, (stokes, diag_out, states) = lax.scan(one_record_block, v, Es_all)
        n_rec = n_on_blocks + n_off_blocks
        time_axis = (jnp.arange(n_rec, dtype=jnp.float64) + 1.0) * dt_val * rec_every
        return time_axis, stokes, diag_out, states

    def _protocol_waveform(params_K, V_blocks_abs, dt_val, n_eq, rec_every):
        """Arbitrary block-wise voltage waveform."""
        lower, diag, upper, c_w = _make_tridiag(dt_val, params_K)
        qsb_v, qst_v = _boundary_vectors()
        v = _v0_init()

        def eq_step(v_curr, _):
            return _step_fn(
                v_curr, lower, diag, upper, c_w, dt_val, 0.0, params_K, qsb_v, qst_v
            ), None

        v, _ = lax.scan(eq_step, v, None, length=n_eq)

        def one_block(v_curr, V_block):
            E_block = V_block / d_cell

            def body(vc, _):
                return _step_fn(
                    vc, lower, diag, upper, c_w, dt_val, E_block, params_K, qsb_v, qst_v
                ), None

            v_end, _ = lax.scan(body, v_curr, None, length=rec_every)
            Q_end = v2Q(v_end)
            st = _all_stokes_fn(Q_end)
            theta, phi, S_eff, beta = angles_from_Q(Q_end)
            diag_out = jnp.stack([theta, phi, S_eff, beta], axis=-1)
            return v_end, (st, diag_out, v_end)

        v, (stokes, diag_out, states) = lax.scan(one_block, v, V_blocks_abs)
        n_rec = V_blocks_abs.shape[0]
        time_axis = (jnp.arange(n_rec, dtype=jnp.float64) + 1.0) * dt_val * rec_every
        return time_axis, stokes, diag_out, states

    def _signal_logparams(
        log10_params, V_abs, dt_val, n_on_blocks, n_off_blocks, n_eq, rec_every
    ):
        """Normalised Stokes (S1/S0, S2/S0, S3/S0) as a flat 1-D array."""
        params_K = 10.0**log10_params
        _t, stokes, _diag, _states = _protocol_recorded(
            params_K,
            V_abs,
            dt_val,
            n_on_blocks,
            n_off_blocks,
            n_eq,
            rec_every,
        )
        S0_comp = stokes[..., 0:1]
        Sn = stokes[..., 1:4] / jnp.clip(S0_comp, 1e-30, None)
        return Sn.ravel()

    # --- JIT-compile ---
    _protocol_jit = jax.jit(
        _protocol_recorded,
        static_argnames=("n_on_blocks", "n_off_blocks", "n_eq", "rec_every"),
    )
    _waveform_jit = jax.jit(
        _protocol_waveform,
        static_argnames=("n_eq", "rec_every"),
    )
    _signal_jit = jax.jit(
        _signal_logparams,
        static_argnames=("n_on_blocks", "n_off_blocks", "n_eq", "rec_every"),
    )
    _jac_signal_jit = jax.jit(
        jax.jacfwd(_signal_logparams, argnums=0),
        static_argnames=("n_on_blocks", "n_off_blocks", "n_eq", "rec_every"),
    )

    # --- count helper ---
    def counts(T_on_=T_on, T_off_=T_off, dt_=dt, T_eq_=T_eq, rec_=record_every):
        n_on = int(round(T_on_ / dt_))
        n_off = int(round(T_off_ / dt_))
        n_eq_ = int(round(T_eq_ / dt_))
        assert n_on % rec_ == 0
        assert n_off % rec_ == 0
        return n_on // rec_, n_off // rec_, n_eq_

    # --- numpy-friendly wrappers ---
    import numpy as np

    def run_protocol_np(
        params_K,
        V_abs,
        *,
        dt_=dt,
        T_on_=T_on,
        T_off_=T_off,
        T_eq_=T_eq,
        rec_=record_every,
    ):
        n_on_b, n_off_b, n_eq_ = counts(T_on_, T_off_, dt_, T_eq_, rec_)
        t, st, diag_out, states = _protocol_jit(
            jnp.asarray(params_K, dtype=jnp.float64),
            float(V_abs),
            float(dt_),
            n_on_b,
            n_off_b,
            n_eq_,
            int(rec_),
        )
        return {
            "time": np.asarray(t),
            "stokes": np.asarray(st),
            "diag": np.asarray(diag_out),
            "states": np.asarray(states),
            "z_um": np.asarray(z_axis) * 1e6,
            "cell": cell.name,
            "d_cell": d_cell,
        }

    def run_waveform_np(
        params_K, V_blocks_abs, *, dt_=dt, T_eq_=T_eq, rec_=record_every
    ):
        n_eq_ = int(round(T_eq_ / dt_))
        t, st, diag_out, states = _waveform_jit(
            jnp.asarray(params_K, dtype=jnp.float64),
            jnp.asarray(V_blocks_abs, dtype=jnp.float64),
            float(dt_),
            n_eq_,
            int(rec_),
        )
        return {
            "time": np.asarray(t),
            "stokes": np.asarray(st),
            "diag": np.asarray(diag_out),
            "states": np.asarray(states),
            "z_um": np.asarray(z_axis) * 1e6,
            "V_blocks_abs": np.asarray(V_blocks_abs),
            "cell": cell.name,
            "d_cell": d_cell,
        }

    def signal_logparams_np(
        log10_params,
        V_abs,
        *,
        dt_=dt,
        T_on_=T_on,
        T_off_=T_off,
        T_eq_=T_eq,
        rec_=record_every,
    ):
        n_on_b, n_off_b, n_eq_ = counts(T_on_, T_off_, dt_, T_eq_, rec_)
        out = _signal_jit(
            jnp.asarray(log10_params, dtype=jnp.float64),
            float(V_abs),
            float(dt_),
            n_on_b,
            n_off_b,
            n_eq_,
            int(rec_),
        )
        return np.asarray(out)

    def jac_signal_logparams_np(
        log10_params,
        V_abs,
        *,
        dt_=dt,
        T_on_=T_on,
        T_off_=T_off,
        T_eq_=T_eq,
        rec_=record_every,
    ):
        n_on_b, n_off_b, n_eq_ = counts(T_on_, T_off_, dt_, T_eq_, rec_)
        out = _jac_signal_jit(
            jnp.asarray(log10_params, dtype=jnp.float64),
            float(V_abs),
            float(dt_),
            n_on_b,
            n_off_b,
            n_eq_,
            int(rec_),
        )
        return np.asarray(out)

    def signal_jac_fd_np(log10_params, V_abs, eps=1e-4, **kwargs):
        x = np.asarray(log10_params, dtype=float)
        f0 = signal_logparams_np(x, V_abs, **kwargs)
        J = np.zeros((f0.size, x.size), dtype=float)
        for k in range(x.size):
            e = np.zeros_like(x)
            e[k] = eps
            fp = signal_logparams_np(x + e, V_abs, **kwargs)
            fm = signal_logparams_np(x - e, V_abs, **kwargs)
            J[:, k] = (fp - fm) / (2.0 * eps)
        return J

    return SimpleNamespace(
        cell=cell,
        dz=dz,
        z_um=np.asarray(z_axis) * 1e6,
        run_protocol_np=run_protocol_np,
        run_waveform_np=run_waveform_np,
        signal_logparams_np=signal_logparams_np,
        jac_signal_logparams_np=jac_signal_logparams_np,
        signal_jac_fd_np=signal_jac_fd_np,
        counts=counts,
    )
