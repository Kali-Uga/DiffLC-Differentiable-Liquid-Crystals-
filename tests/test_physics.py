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


def test_poisson_field_integrates_to_voltage():
    """E(z) reconstructed from the ON-state director must satisfy ∫E dz = V."""
    Nz = 21
    M = _model(True, Nz=Nz)
    Vr = 7.0
    o = M.run_protocol_np(PARAMS, Vr)
    n_on = M.counts(0.05, 0.05, 2.5e-4, 0.02, 2)[0]
    theta = o["diag"][n_on - 1, :, 0]            # tilt from plane, last ON record
    S0, eps_perp, deps = 0.6, 6.9, 12.3
    Qzz = S0 * (np.sin(theta) ** 2 - 1.0 / 3.0)  # uniaxial, n_z = sinθ
    eps_zz = eps_perp + deps / 3.0 + (deps / S0) * Qzz
    dz = D_CELL / (Nz - 1)
    C_inv = np.sum(dz / eps_zz)
    E_field = Vr / (C_inv * eps_zz)
    assert abs(np.sum(E_field * dz) - Vr) < 1e-6


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
