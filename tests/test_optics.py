import math
import jax.numpy as jnp

from difflc import (
    default_cfg,
    ang2Q,
    stokes_normal,
    stokes_oblique,
    jones_linear,
)


def test_stokes_energy_conservation():
    cfg = default_cfg()

    # Create a simple uniform Q-tensor field (Nz=50)
    Nz = 50
    th = jnp.full((Nz,), 0.3, dtype=jnp.float64)
    ph = jnp.full((Nz,), math.pi / 4, dtype=jnp.float64)
    Q_field = ang2Q(th, ph, S_val=cfg.S0)

    wl = 633e-9
    dz = 10e-6 / (Nz - 1)

    # Incident linearly polarized light
    pol_in = jones_linear(45.0)

    # Normal incidence via Jones fallback
    st_norm = stokes_normal(Q_field, wl, pol_in, dz, no=cfg.no, ne=cfg.ne, S0=cfg.S0)

    # For non-absorbing media, total energy S0 should be conserved (close to 1.0)
    assert abs(st_norm[0] - 1.0) < 1e-5

    # Oblique incidence via Berreman 4x4
    st_obliq = stokes_oblique(
        Q_field, wl, math.radians(10.0), pol_in, dz, no=cfg.no, ne=cfg.ne, S0=cfg.S0
    )

    # In oblique, some light is reflected, so S0 of transmitted is <= 1.0,
    # but still positive and reasonable
    assert 0.0 < st_obliq[0] <= 1.0 + 1e-10
