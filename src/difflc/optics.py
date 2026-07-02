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
    # NOTE: for physical Q (S ≤ S0) the eigenvalues of eps_sym already lie in
    # [no², ne²]; the former eigh-based clip to [0.5 no², 1.5 ne²] was therefore
    # never active, while eigh has a singular gradient at degenerate eigenvalues
    # (uniform / homeotropic director under strong field) → NaN in autodiff.
    # Returning the symmetric tensor directly is mathematically equivalent here
    # and keeps the optics differentiable.
    return eps_sym


# ---------------------------------------------------------------------------
# Berreman 4×4 propagation matrix
# ---------------------------------------------------------------------------


def berreman_B_from_eps(eps, theta_rad, n_ambient: float = 1.0):
    """Standard Berreman 4×4 Δ-matrix for the field vector ψ = (Ex, Hy, Ey, −Hx),
    ``dψ/dz = i k0 Δ ψ``.

    Uses the **un-normalised** dielectric tensor ``eps`` and the reduced in-plane
    wavevector ξ = n_ambient·sinθ with θ the incidence angle **in the ambient**
    medium (index ``n_ambient``) — so Snell's law ξ = k_x/k0 is conserved across
    the entrance interface. The layer eigen-indices are then ±√(ε − ξ²), as
    required (the former version used ε/n0² together with ξ = sinθ, equivalent to
    ξ = n0·sinθ — i.e. θ mis-read as an *internal* angle in a medium of index
    n0 — a Snell error of O(n0)).
    """
    xi = n_ambient * jnp.sin(theta_rad)
    exx, exy, exz = eps[0, 0], eps[0, 1], eps[0, 2]
    eyx, eyy, eyz = eps[1, 0], eps[1, 1], eps[1, 2]
    ezx, ezy, ezz = eps[2, 0], eps[2, 1], eps[2, 2]

    B = jnp.zeros((4, 4), dtype=jnp.float64)
    B = B.at[0, 0].set(-xi * ezx / ezz)
    B = B.at[0, 1].set(1.0 - xi * xi / ezz)
    B = B.at[0, 2].set(-xi * ezy / ezz)
    B = B.at[1, 0].set(exx - exz * ezx / ezz)
    B = B.at[1, 1].set(-xi * exz / ezz)
    B = B.at[1, 2].set(exy - exz * ezy / ezz)
    B = B.at[2, 3].set(1.0)
    B = B.at[3, 0].set(eyx - eyz * ezx / ezz)
    B = B.at[3, 1].set(-xi * eyz / ezz)
    B = B.at[3, 2].set(eyy - eyz * ezy / ezz - xi * xi)
    return B


def berreman_layer_transfer(
    Q_layer, wavelength_m, theta_rad, dz, *, no: float, ne: float, S0: float,
    n_ambient: float = 1.0,
):
    """4×4 transfer matrix for a single layer at oblique incidence via expm."""
    eps = eps_from_Q(Q_layer, no=no, ne=ne, S0=S0)
    B = berreman_B_from_eps(eps, theta_rad, n_ambient)
    phase = (2.0 * math.pi / wavelength_m) * dz
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
    # Symmetric dielectric tensor used directly (see eps_from_Q): the previous
    # eigh-clip was inactive for physical Q and introduced NaN gradients at
    # degenerate eigenvalues. eps is symmetric by construction.
    eps_full = 0.5 * (eps + eps.T)
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
    Q_field, wavelength_m, theta_rad, input_pol, dz, *, no: float, ne: float, S0: float,
    n_ambient: float = 1.0,
):
    """Compute transmitted Stokes vector using full Berreman 4×4 transfer
    with entrance/exit boundary matching.

    Parameters
    ----------
    Q_field : (Nz, 3, 3) array — Q-tensor at each grid node
    wavelength_m : scalar — wavelength in metres
    theta_rad : scalar — incidence angle in the ambient medium, radians
    input_pol : (2,) complex — input Jones vector (p, s mode amplitudes)
    dz : scalar — layer spacing
    n_ambient : refractive index of the bounding medium (default 1.0 = air).
        For a glass-clad cell pass ≈1.52: the glass/LC index steps are small
        (n_glass ≈ n_o), so Fresnel reflection is weak and the entrance/exit
        interfaces are nearly index-matched. ξ = n_ambient·sinθ (Snell).

    Returns
    -------
    (4,) Stokes vector
    """
    # Midpoint Q for each layer
    Q_mid = 0.5 * (Q_field[:-1] + Q_field[1:])

    def mul(P, Ql):
        return berreman_layer_transfer(
            Ql, wavelength_m, theta_rad, dz, no=no, ne=ne, S0=S0, n_ambient=n_ambient
        ) @ P, None

    from jax import lax

    P_total, _ = lax.scan(mul, jnp.eye(4, dtype=jnp.complex128), Q_mid)

    # Boundary matching in the ambient medium (index n_ambient) at incidence
    # angle θ. Partial-wave basis for ψ = (Ex, Hy, Ey, −Hx); c = cosθ and
    # n_ambient carry the correct obliquity and Fresnel matching (the former
    # vacuum-normal vectors omitted both, so the interfaces behaved as
    # index-matched with no Fresnel reflection). Forward modes:
    #   p = (c, n_a, 0, 0),   s = (0, 0, 1, n_a·c);  backward flip the ± sign.
    # Verified against the exact Airy transmittance for an isotropic slab (T_s,
    # T_p to 5 decimals at θ = 0/20/35°, n_ambient = 1). At θ=0, c=1.
    c = jnp.cos(theta_rad).astype(jnp.complex128)
    na = jnp.asarray(n_ambient, dtype=jnp.complex128)
    nac = na * c
    z = jnp.zeros((), dtype=jnp.complex128)
    o = jnp.ones((), dtype=jnp.complex128)
    S_i = jnp.array([[c, z], [na, z], [z, o], [z, nac]], dtype=jnp.complex128)
    S_r = jnp.array([[-c, z], [na, z], [z, o], [z, -nac]], dtype=jnp.complex128)
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
    n_ambient: float = 1.0,
):
    """Compute Stokes vectors over all (wavelength × angle × polarization).

    Returns array of shape (n_wl, n_theta, n_pol, 4).
    """

    def per_wl(wl):
        def per_theta(th):
            def per_pol(p):
                return stokes_oblique(
                    Q_field, wl, th, p, dz, no=no, ne=ne, S0=S0, n_ambient=n_ambient
                )

            return jax.vmap(per_pol)(input_pols)

        return jax.vmap(per_theta)(theta_rads)

    return jax.vmap(per_wl)(wavelengths_m)
