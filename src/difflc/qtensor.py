"""Q-tensor and director conversion helpers (JAX backend).

All functions operate on JAX arrays.  The Q-tensor is stored either as a
full symmetric-traceless 3×3 matrix or as a 5-component vector
``v = (Q00, Q01, Q02, Q11, Q12)`` with ``Q22 = -Q00 - Q11``.
"""

from __future__ import annotations

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Fundamental conversions
# ---------------------------------------------------------------------------


def director_from_angles(theta, phi):
    """Director n from polar (θ) and azimuthal (φ) angles.

    Convention: n = (cos θ cos φ, cos θ sin φ, sin θ).
    θ is the angle FROM the substrate plane (pretilt convention).
    """
    return jnp.stack(
        [
            jnp.cos(theta) * jnp.cos(phi),
            jnp.cos(theta) * jnp.sin(phi),
            jnp.sin(theta),
        ],
        axis=-1,
    )


def Q_from_director(n, S_val):
    """Q = S (n⊗n − I/3)  from a unit director n."""
    I3 = jnp.eye(3, dtype=jnp.float64)
    return S_val * (n[..., :, None] * n[..., None, :] - I3 / 3.0)


def ang2Q(theta, phi, S_val=0.6):
    """Build Q-tensor from angles θ, φ and scalar order S."""
    return Q_from_director(director_from_angles(theta, phi), S_val)


# ---------------------------------------------------------------------------
# 5-component ↔ 3×3 packing
# ---------------------------------------------------------------------------


def Q2v(Q):
    """Pack symmetric-traceless 3×3 → 5-component vector."""
    return jnp.stack(
        [
            Q[..., 0, 0],
            Q[..., 0, 1],
            Q[..., 0, 2],
            Q[..., 1, 1],
            Q[..., 1, 2],
        ],
        axis=-1,
    )


def v2Q(v):
    """Unpack 5-component vector → symmetric-traceless 3×3."""
    row0 = jnp.stack([v[..., 0], v[..., 1], v[..., 2]], axis=-1)
    row1 = jnp.stack([v[..., 1], v[..., 3], v[..., 4]], axis=-1)
    row2 = jnp.stack([v[..., 2], v[..., 4], -v[..., 0] - v[..., 3]], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def proj_ST(H):
    """Project a (batch of) 3×3 matrices onto the symmetric-traceless subspace."""
    Hs = 0.5 * (H + jnp.swapaxes(H, -2, -1))
    tr = jnp.trace(Hs, axis1=-2, axis2=-1) / 3.0
    I3 = jnp.eye(3, dtype=jnp.float64)
    return Hs - tr[..., None, None] * I3


# ---------------------------------------------------------------------------
# Director / order extraction from Q
# ---------------------------------------------------------------------------


def extract_director(Q):
    """Extract the director (eigenvector of largest eigenvalue) from Q.

    Returns (n, eigvals) where n has nz ≥ 0 (head-tail symmetry).
    """
    eigvals, eigvecs = jnp.linalg.eigh(Q)
    n = eigvecs[..., :, -1]
    return jnp.where(n[..., 2:3] < 0.0, -n, n), eigvals


def scalar_order_from_eigvals(eigvals):
    """Scalar order parameter S from eigenvalues: S = 1.5 · λ_max."""
    return 1.5 * eigvals[..., -1]


def biaxiality_from_eigvals(eigvals):
    """Biaxiality parameter β ∈ [0, 1].  β = 0 uniaxial, β → 1 biaxial.

    β = 1 − 6 (tr Q³)² / (tr Q²)³
    """
    tr2 = jnp.sum(eigvals**2, axis=-1)
    tr3 = jnp.sum(eigvals**3, axis=-1)
    return 1.0 - 6.0 * tr3**2 / jnp.clip(tr2**3, 1e-30, None)


def angles_from_Q(Q):
    """Extract (θ, φ, S_eff, β) from Q-tensor field.

    Returns
    -------
    theta : polar angle (from substrate plane)
    phi   : azimuthal angle
    S_eff : effective scalar order
    beta  : biaxiality
    """
    n, eigvals = extract_director(Q)
    theta = jnp.arcsin(jnp.clip(n[..., 2], -1.0, 1.0))
    phi = jnp.arctan2(n[..., 1], n[..., 0])
    S_eff = scalar_order_from_eigvals(eigvals)
    beta = biaxiality_from_eigvals(eigvals)
    return theta, phi, S_eff, beta


def angles_from_v(v):
    """Convenience: angles_from_Q(v2Q(v))."""
    return angles_from_Q(v2Q(v))
