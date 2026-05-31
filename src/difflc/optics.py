"""Full Berreman 4×4 transfer-matrix optics (JAX backend).

Supports normal and oblique incidence, multiple wavelengths,
incidence angles, and input polarizations.  The boundary matching
solves for reflected and transmitted amplitudes so that the full
entrance/exit interface is modelled.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


# ---------------------------------------------------------------------------
# Dielectric tensor from Q
# ---------------------------------------------------------------------------


def eps_from_Q(Q, *, no: float, ne: float, S0: float):
    """Optical dielectric tensor ε(Q) for each spatial layer.

    For uniaxial Q = S(nn − I/3):
        ε = no² I + Δn² nn = ε_iso I + (Δn²/S₀) Q
    Eigenvalues are clipped to a positive physical interval to prevent
    sqrt-of-negative issues during dynamics.
    """
    I3 = jnp.eye(3, dtype=jnp.float64)
    delta_n2 = ne**2 - no**2
    eps_iso = no**2 + delta_n2 / 3.0
    eps_raw = eps_iso * I3 + (delta_n2 / S0) * Q
    eps_sym = 0.5 * (eps_raw + jnp.swapaxes(eps_raw, -2, -1))
    vals, vecs = jnp.linalg.eigh(eps_sym)
    vals = jnp.clip(vals, no**2 * 0.5, ne**2 * 1.5)
    return (vecs * vals[..., None, :]) @ jnp.swapaxes(vecs, -2, -1)


# ---------------------------------------------------------------------------
# Berreman 4×4 propagation matrix
# ---------------------------------------------------------------------------


def berreman_B_from_eps(eps_tilde, theta_rad):
    """4×4 Berreman Δ-matrix from normalised dielectric tensor and incidence angle."""
    lam = jnp.cos(theta_rad)
    ss = jnp.sin(theta_rad)
    e11 = eps_tilde[0, 0]
    e12 = eps_tilde[0, 1]
    e13 = eps_tilde[0, 2]
    e22 = eps_tilde[1, 1]
    e23 = eps_tilde[1, 2]
    e33 = eps_tilde[2, 2]

    B = jnp.zeros((4, 4), dtype=jnp.float64)
    B = B.at[0, 0].set(-ss * e13 / e33)
    B = B.at[0, 1].set((lam * lam + e33 - 1.0) / (e33 * lam))
    B = B.at[0, 3].set(-ss * e23 / (e33 * lam))
    B = B.at[1, 0].set(lam * (e11 * e33 - e13 * e13) / e33)
    B = B.at[1, 1].set(-ss * e13 / e33)
    B = B.at[1, 3].set((e12 * e33 - e13 * e23) / e33)
    B = B.at[2, 0].set((e12 * e33 - e13 * e23) / e33)
    B = B.at[2, 1].set(-ss * e23 / (e33 * lam))
    B = B.at[2, 3].set(((lam * lam + e22 - 1.0) * e33 - e23 * e23) / (e33 * lam))
    B = B.at[3, 2].set(lam)
    return B


def berreman_layer_transfer(
    Q_layer, wavelength_m, theta_rad, dz, *, no: float, ne: float, S0: float
):
    """4×4 transfer matrix for a single layer at oblique incidence via expm."""
    n0 = jnp.sqrt(no * ne)
    eps = eps_from_Q(Q_layer, no=no, ne=ne, S0=S0)
    eps_tilde = eps / n0**2
    B = berreman_B_from_eps(eps_tilde, theta_rad)
    phase = (2.0 * math.pi / wavelength_m) * n0 * dz
    return expm((1j * phase) * B.astype(jnp.complex128))


# ---------------------------------------------------------------------------
# Stokes vector
# ---------------------------------------------------------------------------


def stokes_from_jones2(E):
    """Full 4-component Stokes vector from a 2-component Jones vector."""
    Ex, Ey = E[0], E[1]
    S0 = jnp.real(Ex * jnp.conj(Ex) + Ey * jnp.conj(Ey))
    S1 = jnp.real(Ex * jnp.conj(Ex) - Ey * jnp.conj(Ey))
    S2 = 2.0 * jnp.real(Ex * jnp.conj(Ey))
    S3 = -2.0 * jnp.imag(Ex * jnp.conj(Ey))
    return jnp.stack([S0, S1, S2, S3])


# ---------------------------------------------------------------------------
# Normal-incidence Jones chain (fast fallback)
# ---------------------------------------------------------------------------


def jones_layer_normal(
    Q_layer,
    wavelength_m,
    dz,
    *,
    no: float,
    ne: float,
    S0: float,
    path_factor: float = 1.0,
):
    """2×2 Jones matrix for one layer at normal incidence."""
    I3 = jnp.eye(3, dtype=jnp.float64)
    delta_n2 = ne**2 - no**2
    eps_iso = no**2 + delta_n2 / 3.0

    eps = eps_iso * I3 + (delta_n2 / S0) * Q_layer
    eps_sym = 0.5 * (eps + eps.T)
    vals_3d, vecs_3d = jnp.linalg.eigh(eps_sym)
    vals_3d = jnp.clip(vals_3d, no**2 * 0.5, ne**2 * 1.5)

    # Effective 2×2 in-plane dielectric tensor
    eps_full = (vecs_3d * vals_3d[None, :]) @ vecs_3d.T
    eps_tt = eps_full[:2, :2]
    eps_tz = eps_full[:2, 2:3]
    eps_zt = eps_full[2:3, :2]
    eps_zz = eps_full[2:3, 2:3]
    eps_eff = eps_tt - eps_tz @ eps_zt / jnp.clip(eps_zz, 1e-30, None)

    a = eps_eff[0, 0]
    b = 0.5 * (eps_eff[0, 1] + eps_eff[1, 0])
    d = eps_eff[1, 1]
    det = jnp.clip(a * d - b * b, 1e-30, None)
    sdet = jnp.sqrt(det)
    denom = jnp.sqrt(jnp.clip(a + d + 2.0 * sdet, 1e-30, None))
    I2 = jnp.eye(2, dtype=jnp.float64)
    N = (jnp.array([[a, b], [b, d]]) + sdet * I2) / denom

    k0_dz = 2.0 * math.pi / wavelength_m * dz * path_factor
    tr_half = 0.5 * (N[0, 0] + N[1, 1])
    B0 = N - tr_half * I2
    r = jnp.sqrt(jnp.clip(B0[0, 0] ** 2 + B0[0, 1] ** 2, 0.0, None))
    x = k0_dz * r
    sin_over_r = k0_dz * jnp.sinc(x / jnp.pi)
    scalar = jnp.exp(1j * k0_dz * tr_half)
    return scalar * (
        jnp.cos(x).astype(jnp.complex128) * I2.astype(jnp.complex128)
        + 1j * sin_over_r.astype(jnp.complex128) * B0.astype(jnp.complex128)
    )


# ---------------------------------------------------------------------------
# Full Berreman oblique propagation with boundary matching
# ---------------------------------------------------------------------------


def stokes_oblique(
    Q_field, wavelength_m, theta_rad, input_pol, dz, *, no: float, ne: float, S0: float
):
    """Compute transmitted Stokes vector using full Berreman 4×4 transfer
    with entrance/exit boundary matching.

    Parameters
    ----------
    Q_field : (Nz, 3, 3) array — Q-tensor at each grid node
    wavelength_m : scalar — wavelength in metres
    theta_rad : scalar — incidence angle in radians
    input_pol : (2,) complex — input Jones vector
    dz : scalar — layer spacing

    Returns
    -------
    (4,) Stokes vector
    """
    # Midpoint Q for each layer
    Q_mid = 0.5 * (Q_field[:-1] + Q_field[1:])

    def mul(P, Ql):
        return berreman_layer_transfer(
            Ql, wavelength_m, theta_rad, dz, no=no, ne=ne, S0=S0
        ) @ P, None

    from jax import lax

    P_total, _ = lax.scan(mul, jnp.eye(4, dtype=jnp.complex128), Q_mid)

    # Boundary matching: incident, reflected, transmitted
    S_i = jnp.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=jnp.complex128)
    S_r = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.complex128)
    S_t = S_i

    A = jnp.concatenate([P_total @ S_r, -S_t], axis=1)
    rhs = -(P_total @ (S_i @ input_pol))
    sol = jnp.linalg.solve(A, rhs)
    return stokes_from_jones2(sol[2:4])


def stokes_normal(
    Q_field, wavelength_m, input_pol, dz, *, no: float, ne: float, S0: float
):
    """Normal-incidence Stokes via Jones chain (faster than Berreman)."""
    Q_mid = 0.5 * (Q_field[:-1] + Q_field[1:])

    def mul(M, Ql):
        return jones_layer_normal(Ql, wavelength_m, dz, no=no, ne=ne, S0=S0) @ M, None

    from jax import lax

    M_total, _ = lax.scan(mul, jnp.eye(2, dtype=jnp.complex128), Q_mid)
    return stokes_from_jones2(M_total @ input_pol)


# ---------------------------------------------------------------------------
# Vectorised multi-condition Stokes
# ---------------------------------------------------------------------------


def all_stokes(
    Q_field,
    wavelengths_m,
    theta_rads,
    input_pols,
    dz,
    *,
    no: float,
    ne: float,
    S0: float,
):
    """Compute Stokes vectors over all (wavelength × angle × polarization).

    Returns array of shape (n_wl, n_theta, n_pol, 4).
    """

    def per_wl(wl):
        def per_theta(th):
            def per_pol(p):
                return stokes_oblique(Q_field, wl, th, p, dz, no=no, ne=ne, S0=S0)

            return jax.vmap(per_pol)(input_pols)

        return jax.vmap(per_theta)(theta_rads)

    return jax.vmap(per_wl)(wavelengths_m)
