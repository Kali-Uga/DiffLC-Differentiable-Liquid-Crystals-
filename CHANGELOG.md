# Changelog

## 0.3.1

Gradient-safety and vectorisation pass (no change to the physics; the forward
and Jacobian outputs are unchanged — bit-for-bit on the same backend, ~1e-12
across backends where `vmap` re-associates floating-point ops).

### Added
- `model.run_protocols_np(params, V_array)` — batches a protocol over several
  voltages in one `vmap` kernel (≈n_V× faster than looping `run_protocol_np` on a
  parallel backend; matches the loop to ~1e-7 across backends — the whole
  trajectory is re-associated in the batched kernel and accumulates over the
  relaxation). `model.stability_dt_max(params)` exposes the
  explicit-stability limit.

### Changed
- **Optics and angle diagnostics moved out of the time `scan`.** They depend
  only on the recorded states, so they are now batched with `vmap` after the
  scan instead of serialised inside it (far fewer sequential `expm` calls and
  Jacobian tangents). Forward/Jacobian outputs unchanged (0 ULP on the same
  backend; ~1e-12 across backends).
- `angles_from_Q` (eigh) diagnostics wrapped in `stop_gradient` — output-only,
  and `eigh` has a singular JVP at degenerate eigenvalues; keeps them out of the
  `jacfwd` graph.
- `solve_inverse` defaults: `x_scale="jac"` (was `1.0`, which stalls on the
  ill-conditioned fringe Jacobian) and `strict_stability=True` (raise instead of
  silently fitting NaN-masked residuals when `dt` exceeds the stability limit);
  `loss`/`xtol`/`ftol`/`gtol` are now arguments. Residual/Jacobian failures warn
  instead of masking to 1e6 silently.

### Fixed
- `jones_layer_normal`: sqrt argument lower-clipped to 1e-30 (was 0.0) — removes
  a NaN autodiff tangent at the exact homeotropic (in-plane isotropic) layer.
- `eps_from_Q` docstring no longer claims eigenvalue clipping (removed in 0.3.0).
- Documented that importing DiffLC flips JAX to float64 globally.

## 0.3.0

Physics/optics correctness pass (see `tests/test_physics.py`).

### Breaking changes
- **Oblique Berreman optics rewritten.** `berreman_B_from_eps(eps, theta_rad, n_ambient=1.0)`
  now takes the **un-normalised** dielectric tensor and builds the standard Berreman
  Δ-matrix with ξ = `n_ambient`·sinθ; `berreman_layer_transfer` drops the internal
  `n0` phase factor. The previous version used ε/n0² with ξ=sinθ (equivalent to
  ξ=n0·sinθ — a Snell error). **Any code calling these two functions directly will
  behave differently.** Validated against the exact Airy transmittance to 5 decimals.
- **Boundary matching now models Fresnel reflection** with a bounding medium index.
  `make_model`, `all_stokes`, `stokes_oblique` gain `n_ambient` (default **1.0 = air**).
  The old code was effectively index-matched (no reflection); the new default changes
  the normalised Stokes at θ=0 by up to ~0.03–0.08 depending on cell geometry
  (≈0.03 in ΔI on a 12.5 µm cell, up to ~0.08 in ΔS2 on a thin plate). **Glass-clad cells should pass
  `n_ambient≈1.52`.** Synthetic data / FIM / fits generated before 0.3.0 are not
  directly comparable.
- **Rotational-viscosity mapping fixed:** μ = 1/γ_Q = **2·S0²/γ1** (was S0/γ1). The old
  value made dynamics 2·S0≈1.2× too slow and gave recovered γ1 a spurious ∝1/S0
  dependence. Recovered γ1 from earlier fits is low by ×2·S0; elastic constants are
  unaffected (exact reparameterisation).

### Fixed
- Poisson field quadrature: trapezoidal ∫dz/ε_zz (was rectangular → effective
  thickness d+dz, an O(1/Nz) field deficit).
- Backflow: μ(θ) no longer scales the scalar-order relaxation; docstring corrected
  (the sin²2θ form is a phenomenological surrogate, not an Ericksen–Leslie reduction).
- `stokes_normal` documented as a retardation-only (no-interface) path that diverges
  from `stokes_oblique(θ=0)` by the Fresnel boundary term.

### Notes
- `oed.py` is local/frequentist (Fisher + D-optimality), not Bayesian — README wording
  corrected accordingly.

## 0.2.0
- JAX backend, E7 default, Berreman optics.
