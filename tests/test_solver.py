import numpy as np

from difflc import default_cfg, default_cells, build_protocols, make_model


def test_run_protocol_is_finite():
    cfg = default_cfg()
    cells = default_cells()
    protocols = build_protocols(cells, cfg)
    p0 = protocols[0]  # TN90 cell

    # Create the model for this cell
    model = make_model(cfg, p0.cell)

    p_true = np.array([cfg.K11, cfg.K22, cfg.K33, cfg.gamma1, cfg.W])

    # Run a short protocol to check for NaNs
    out = model.run_protocol_np(
        p_true, p0.V_abs, dt_=1e-3, T_on_=0.02, T_off_=0.0, T_eq_=0.01, rec_=4
    )

    assert np.isfinite(out["states"]).all()
    assert np.isfinite(out["stokes"]).all()

    # Stokes should have 4 components, check that S0 is positive
    assert np.all(out["stokes"][..., 0] >= 0.0)


def test_final_state_has_angles():
    cfg = default_cfg()
    cell = default_cells()[1]  # PLANAR0 cell
    model = make_model(cfg, cell)

    p_true = np.array([cfg.K11, cfg.K22, cfg.K33, cfg.gamma1, cfg.W])
    out = model.run_protocol_np(
        p_true, 3.0, dt_=1e-3, T_on_=0.02, T_off_=0.0, T_eq_=0.01, rec_=4
    )

    diag = out["diag"]  # (n_rec, Nz, 4): theta, phi, S_eff, beta

    assert np.isfinite(diag).all()

    S_eff = diag[..., 2]
    beta = diag[..., 3]

    # S_eff should be around S0 (0.6)
    assert np.all((S_eff > 0.0) & (S_eff < 1.0))

    # Biaxiality should be between 0 and 1
    assert np.all((beta >= 0.0) & (beta <= 1.0))
