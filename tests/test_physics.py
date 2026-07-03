"""Physics / numerics regression tests (added in R&D iteration 3).

Covers: fixed-voltage Poisson electrostatics, autodiff vs finite-difference
gradient agreement, and the explicit-stability warning.
"""

import math
import warnings

import numpy as np
import pytest

from difflc import E7Config, CellSpec, make_model, jones_linear

D_CELL = 12.5e-6
PARAMS = np.array([10e-12, 4e-12, 10e-12, 0.08, 1e-3])  # K11,K22,K33,gamma1,W


def _model(poisson, Nz=21, dt=2.5e-4, rec=2, T_on=0.05, T_off=0.05):
    cfg = E7Config(
        K11=10e-12, K22=4e-12, K33=10e-12, gamma1=0.08, W=1e-3,
        no=1.511, ne=1.691, eps_par=19.2, eps_perp=6.9, Nz=Nz,
        pretilt_deg=2.0, S0=0.6,
    )
    return make_model(
        cfg, CellSpec("t", d_cell=D_CELL, twist_deg=0.0, voltage_ratio=1.0),
        wavelengths_nm=(632.8,), incidence_deg=(0.0,),
        input_pols=np.array([jones_linear(45.0)]),
        dt=dt, record_every=rec, T_on=T_on, T_off=T_off, T_eq=0.02,
        poisson=poisson,
    )


def _poisson_E_from_solver(Nz, theta_rad, V, S0=0.6, eps_perp=6.9, deps=12.3):
    """Extract the interior E(z) that the solver's own field routine produces.

    Uses a *uniform* director so every elastic gradient term (L1/L2/L3) and
    second derivative vanishes; then the molecular field on interior nodes is
    exactly h[·,2,2] = ½ε₀(Δε/S0)E(z)², so E(z) is recovered from the solver
    output rather than reconstructed in the test. This exercises the real
    ``_elastic_electric_field`` quadrature (unlike the old tautological check,
    which rebuilt E with the same formula it then integrated)."""
    from difflc.solver import _elastic_electric_field

    n = np.array([np.cos(theta_rad), 0.0, np.sin(theta_rad)])
    Q_single = S0 * (np.outer(n, n) - np.eye(3) / 3.0)
    Q = np.broadcast_to(Q_single, (Nz, 3, 3)).copy()
    dz = D_CELL / (Nz - 1)
    EPS_0 = 8.854187817e-12
    E_plate = V / D_CELL
    h, _ = _elastic_electric_field(
        Q, E_plate, 0.0, 0.0, 0.0, EPS_0, deps, S0, dz,
        eps_perp=eps_perp, d_cell=D_CELL, poisson=True,
    )
    h_zz = np.asarray(h)[:, 2, 2]
    coeff = 0.5 * EPS_0 * (deps / S0)
    E = np.sqrt(np.clip(h_zz / coeff, 0.0, None))   # boundary nodes are masked → 0
    return E[1:-1], dz                              # interior E(z)


def test_poisson_field_uniform_limit_equals_V_over_d():
    """In the uniform (homeotropic) limit the fixed-V field must be E = V/d
    exactly. The former rectangular quadrature integrated an extra half-cell at
    each end (effective thickness d+dz) → E = V/(d+dz), an O(1/Nz) deficit that
    biases the dielectric torque; the trapezoidal rule removes it. This test
    would FAIL on the old ``jnp.sum(dz/eps_zz)`` quadrature."""
    for Nz in (21, 41, 81):
        E, _ = _poisson_E_from_solver(Nz, np.radians(75.0), V=5.0)
        E_expected = 5.0 / D_CELL
        assert np.allclose(E, E_expected, rtol=1e-9), (
            f"Nz={Nz}: E/(V/d)={float(E.mean())/E_expected:.6f} (want 1.0)"
        )


def test_poisson_homeotropic_limit_matches_uniform():
    """At very high voltage the cell is uniformly homeotropic → ε_zz≈const →
    field is uniform, so the Poisson and uniform ON-state coincide.
    (Measured on the ON steady level: the relaxation sweeps through intermediate
    tilts where the field redistributes, so it differs even when ON matches.)"""
    Mu, Mp = _model(False), _model(True)
    Vr = 30.0
    assert abs(_on_level(Mu, Vr) - _on_level(Mp, Vr)) < 0.01


def test_poisson_differs_at_intermediate_voltage():
    """At intermediate tilt the field redistributes → Poisson ON-state must
    differ from the uniform-field model by a measurable (but bounded) amount."""
    Mu, Mp = _model(False), _model(True)
    Vr = 5.0
    d = abs(_on_level(Mu, Vr) - _on_level(Mp, Vr))
    assert 0.01 < d < 0.5


def _on_level(M, Vr):
    """Steady ON-state transmitted intensity (last ON record)."""
    o = M.run_protocol_np(PARAMS, Vr)
    n_on = M.counts(0.05, 0.05, 2.5e-4, 0.02, 2)[0]
    st = o["stokes"]
    s2 = st[..., 2] / np.clip(st[..., 0], 1e-30, None)
    I = np.nan_to_num(np.squeeze(0.5 * (1.0 - s2)))
    return float(I[n_on - 1])


def _relax_I(M, Vr):
    o = M.run_protocol_np(PARAMS, Vr)
    st = o["stokes"]
    s2 = st[..., 2] / np.clip(st[..., 0], 1e-30, None)
    return np.nan_to_num(np.squeeze(0.5 * (1.0 - s2)))


def test_jax_jacobian_matches_finite_difference():
    """Autodiff Jacobian must match central finite differences in the smooth
    (low-voltage) regime — guards the eigh-free differentiable optics."""
    M = _model(False, Nz=21)
    lp = np.log10(PARAMS)
    Jj = M.jac_signal_logparams_np(lp, 4.0, dt_=2.5e-4, T_on_=0.05, T_off_=0.05, T_eq_=0.02, rec_=2)
    Jf = M.signal_jac_fd_np(lp, 4.0, eps=1e-3, dt_=2.5e-4, T_on_=0.05, T_off_=0.05, T_eq_=0.02, rec_=2)
    assert np.all(np.isfinite(Jj))
    rel = np.linalg.norm(Jj - Jf) / (np.linalg.norm(Jf) + 1e-30)
    assert rel < 1e-2


def test_stability_warning_fires_when_dt_too_large():
    """Refining the grid (Nz=81) at the coarse default dt must trip the
    explicit-stability RuntimeWarning rather than silently producing NaN."""
    M = _model(False, Nz=81, dt=2.5e-4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        M.run_protocol_np(PARAMS, 5.0)
        assert any("stability" in str(x.message) for x in w)


def test_stability_warning_silent_when_safe():
    M = _model(False, Nz=41, dt=2.5e-4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        M.run_protocol_np(PARAMS, 5.0)
        assert not any("stability" in str(x.message) for x in w)


def _relax_vec(M, n_on, K11, K33, g1, pt, Vrs):
    out = []
    for Vr in Vrs:
        o = M.run_protocol_np(np.array([K11, 4e-12, K33, g1, 1e-3]), Vr, pretilt_deg=pt)
        st = o["stokes"]
        s2 = st[..., 2] / np.clip(st[..., 0], 1e-30, None)
        I = np.nan_to_num(np.squeeze(0.5 * (1.0 - s2)))[n_on:]
        out.append(I)
    return np.concatenate(out)


def _relax_S2_jac_rows(M, n_on, n_rec):
    """Indices into jac_signal_logparams output selecting the S2/S0 component
    on the relaxation records (signal raveled as [S1,S2,S3] per record)."""
    return [r * 3 + 1 for r in range(n_on, n_rec)]


def _analytic_jac(M, n_on, K11, K33, g1, pt, Vrs):
    """Exact d(I_relax)/d[log10 K11, log10 K33, log10 γ1] via the model's JAX
    Jacobian of the normalised Stokes signal. I = ½(1−S2/S0) ⇒ dI = −½ d(S2/S0).
    The finite-difference Jacobian is inaccurate on the oscillatory fringe
    signal and stalls TRF (trust region collapses at large gradient); the
    analytic Jacobian converges in a handful of iterations."""
    log5 = np.array([np.log10(K11), np.log10(4e-12), np.log10(K33), np.log10(g1), np.log10(1e-3)])
    blocks = []
    for Vr in Vrs:
        Jf = M.jac_signal_logparams_np(log5, Vr, dt_=2.5e-4, T_on_=0.1, T_off_=0.2,
                                       T_eq_=0.02, rec_=2, pretilt_deg=pt)
        n_rec = Jf.shape[0] // 3
        rows = _relax_S2_jac_rows(M, n_on, n_rec)
        blocks.append(-0.5 * Jf[rows][:, [0, 2, 3]])
    return np.vstack(blocks)


def test_inverse_roundtrip_converges_with_analytic_jacobian():
    """Synthetic round-trip on noise-free data. With the EXACT JAX Jacobian a
    properly-working optimiser must drive cost→0, ‖Jᵀr‖→0 and recover the truth
    to ≪1 %. (With finite differences the same fit stalls at xtol with a large
    gradient — this test pins the analytic-Jacobian fix.)"""
    from scipy.optimize import least_squares

    M = _model(False, Nz=21, dt=2.5e-4, rec=2, T_on=0.1, T_off=0.2)
    n_on = M.counts(0.1, 0.2, 2.5e-4, 0.02, 2)[0]
    Vrs = [3.5, 7.0]
    true = [10e-12, 12e-12, 0.085]
    pt = 2.5
    data = _relax_vec(M, n_on, true[0], true[1], true[2], pt, Vrs)

    def resid(x):
        return _relax_vec(M, n_on, 10 ** x[0], 10 ** x[1], 10 ** x[2], pt, Vrs) - data

    def jac(x):
        return _analytic_jac(M, n_on, 10 ** x[0], 10 ** x[1], 10 ** x[2], pt, Vrs)

    x0 = [math.log10(9e-12), math.log10(11e-12), math.log10(0.078)]  # same fringe winding
    lb = [math.log10(2e-12)] * 2 + [math.log10(0.02)]
    ub = [math.log10(5e-11)] * 2 + [math.log10(0.5)]
    r = least_squares(resid, x0, jac=jac, method="trf", bounds=(lb, ub), x_scale="jac",
                      xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)
    rec = 10 ** r.x
    rel = np.abs(rec - np.array(true)) / np.array(true)
    assert r.cost < 1e-8, f"cost not driven to 0: {r.cost}"
    assert r.optimality < 1e-4, f"not gradient-converged: ||Jᵀr||={r.optimality}"
    assert np.all(rel < 1e-3), f"params not recovered: rel={rel}"


# ---------------------------------------------------------------------------
# Oblique optics (Berreman): Snell + Fresnel boundary vs exact Airy (R&D iter 4)
# ---------------------------------------------------------------------------


def _airy_T(n, d, lam, theta_amb, pol, n_amb=1.0):
    """Exact transmittance of a lossless isotropic slab (index n, thickness d)
    in an ambient of index n_amb, incidence angle theta_amb, s/p polarisation."""
    sa, ca = math.sin(theta_amb), math.cos(theta_amb)
    st = n_amb * sa / n
    ct = math.sqrt(1.0 - st * st)
    if pol == "s":
        r = (n_amb * ca - n * ct) / (n_amb * ca + n * ct)
    else:
        r = (n * ca - n_amb * ct) / (n * ca + n_amb * ct)
    R = r * r
    delta = 2.0 * math.pi / lam * n * d * ct
    F = 4.0 * R / (1.0 - R) ** 2
    return 1.0 / (1.0 + F * math.sin(delta) ** 2)


@pytest.mark.parametrize("n_amb", [1.0, 1.52])
@pytest.mark.parametrize("theta_deg", [0.0, 20.0, 35.0])
def test_berreman_oblique_matches_airy(n_amb, theta_deg):
    """The oblique Berreman path (Snell ξ=n_amb·sinθ + Fresnel boundary) must
    reproduce the exact Airy transmittance of an isotropic slab. This pins both
    the Snell factor (former ξ=n0·sinθ error) and the boundary matching (former
    vacuum-normal vectors gave ~no reflection). Isotropic Q=0 ⇒ ε = n²·I."""
    import jax.numpy as jnp
    from difflc.optics import stokes_oblique

    n, d, lam, Nz = 1.573, 2e-6, 532e-9, 201
    dz = d / (Nz - 1)
    Q = jnp.zeros((Nz, 3, 3))
    th = math.radians(theta_deg)
    S0_p = float(stokes_oblique(Q, lam, th, jnp.array([1, 0], dtype=complex),
                                dz, no=n, ne=n, S0=0.6, n_ambient=n_amb)[0])
    S0_s = float(stokes_oblique(Q, lam, th, jnp.array([0, 1], dtype=complex),
                                dz, no=n, ne=n, S0=0.6, n_ambient=n_amb)[0])
    assert abs(S0_p - _airy_T(n, d, lam, th, "p", n_amb)) < 1e-4
    assert abs(S0_s - _airy_T(n, d, lam, th, "s", n_amb)) < 1e-4


def test_backflow_kappa_changes_dynamics_and_is_finite():
    """The backflow branch (backflow_kappa>0) must run without NaN and produce a
    relaxation that differs from κ=0 (guards the per-node μ(θ) tridiagonal /
    bulk path, which had no test)."""
    def relax(kappa):
        M = make_model(
            E7Config(K11=10e-12, K22=4e-12, K33=10e-12, gamma1=0.08, W=1e-3,
                     no=1.511, ne=1.691, eps_par=19.2, eps_perp=6.9, Nz=41,
                     pretilt_deg=2.0, S0=0.6),
            CellSpec("t", d_cell=12.5e-6, twist_deg=0.0, voltage_ratio=1.0),
            wavelengths_nm=(632.8,), incidence_deg=(0.0,),
            input_pols=np.array([jones_linear(45.0)]),
            dt=2.5e-4, record_every=2, T_on=0.1, T_off=0.2, T_eq=0.02,
            backflow_kappa=kappa,
        )
        o = M.run_protocol_np(PARAMS, 6.0)
        st = o["stokes"]
        s2 = st[..., 2] / np.clip(st[..., 0], 1e-30, None)
        return np.nan_to_num(np.squeeze(0.5 * (1.0 - s2)))

    I0, Ik = relax(0.0), relax(0.4)
    assert np.all(np.isfinite(I0)) and np.all(np.isfinite(Ik))
    assert np.max(np.abs(I0 - Ik)) > 1e-3, "backflow κ=0.4 did not change dynamics"


# ---------------------------------------------------------------------------
# R&D iteration 5: gradient safety + vectorisation
# ---------------------------------------------------------------------------


def test_jones_normal_grad_finite_at_homeotropic():
    """jones_layer_normal must have a finite autodiff gradient at the exact
    homeotropic director, where the in-plane dielectric tensor is isotropic and
    the internal sqrt argument hits 0 (the 1e-30 lower clip removes the NaN)."""
    import jax
    import jax.numpy as jnp
    from difflc.optics import jones_layer_normal

    S0, no, ne = 0.6, 1.511, 1.691

    def scalar(theta):
        n = jnp.array([jnp.cos(theta), 0.0, jnp.sin(theta)])
        Q = S0 * (jnp.outer(n, n) - jnp.eye(3) / 3.0)
        M = jones_layer_normal(Q, 632.8e-9, 3e-7, no=no, ne=ne, S0=S0)
        return jnp.sum(jnp.abs(M) ** 2)

    g = jax.grad(scalar)(jnp.array(jnp.pi / 2))   # exact homeotropic
    assert jnp.isfinite(g)


def test_run_protocols_batch_matches_loop():
    """run_protocols_np (vmap over V) must match looping run_protocol_np. It is
    exact on a single backend, but XLA re-associates the batched reductions
    (non-deterministically across runs), and that ~1e-15/step noise accumulates
    over the relaxation and is amplified by the fringe-sensitive optics — observed
    up to ~1e-7 on JAX 0.10 / numpy 2. Assert numerical equivalence (< 1e-5, still
    5+ orders below any wiring bug, which would give O(1) differences), not
    bit-equality. Also exercises the optics-out-of-scan refactor and empty guard."""
    cfg = E7Config(K11=10e-12, K22=4e-12, K33=12e-12, gamma1=0.09, W=1e-3,
                   no=1.511, ne=1.691, eps_par=19.2, eps_perp=6.9, Nz=41,
                   pretilt_deg=2.5, S0=0.6)
    M = make_model(cfg, CellSpec("c", d_cell=12.5e-6, twist_deg=0.0, voltage_ratio=1.0),
                   wavelengths_nm=(632.8,), incidence_deg=(0.0,),
                   input_pols=np.array([jones_linear(45.0)]),
                   dt=2.5e-4, record_every=2, T_on=0.05, T_off=0.05, T_eq=0.02,
                   poisson=True, n_ambient=1.52)
    P = np.array([10e-12, 4e-12, 12e-12, 0.09, 1e-3])
    Vs = [3.5, 5.0, 7.0]
    batch = M.run_protocols_np(P, Vs)
    for i, V in enumerate(Vs):
        one = M.run_protocol_np(P, V)
        assert np.max(np.abs(batch["stokes"][i] - one["stokes"])) < 1e-5
        assert np.max(np.abs(batch["diag"][i] - one["diag"])) < 1e-5
    assert np.all(np.isfinite(batch["stokes"]))
    with pytest.raises(ValueError):
        M.run_protocols_np(P, [])


def test_solve_inverse_strict_stability_raises():
    """solve_inverse(strict_stability=True) must fail fast when dt exceeds the
    explicit-stability limit, instead of silently fitting masked-1e6 residuals."""
    from types import SimpleNamespace
    from difflc.inverse import solve_inverse

    cfg = E7Config(K11=10e-12, K22=4e-12, K33=16e-12, gamma1=0.08, W=1e-3,
                   no=1.511, ne=1.691, eps_par=19.2, eps_perp=6.9, Nz=81,
                   pretilt_deg=2.0, S0=0.6)
    M = make_model(cfg, CellSpec("t", d_cell=12.5e-6, twist_deg=0.0, voltage_ratio=1.0),
                   wavelengths_nm=(632.8,), incidence_deg=(0.0,),
                   input_pols=np.array([jones_linear(45.0)]))
    proto = SimpleNamespace(name="t", V_abs=5.0)
    p_true = np.array([10e-12, 4e-12, 16e-12, 0.08, 1e-3])
    # Nz=81 with the coarse default dt=2.5e-4 is above the explicit-stability limit.
    with pytest.raises(ValueError, match="stability"):
        solve_inverse({"t": M}, [proto], np.zeros(10), p_true, 0.01,
                      dt=2.5e-4, T_on=0.05, T_off=0.05, T_eq=0.02, record_every=2)
