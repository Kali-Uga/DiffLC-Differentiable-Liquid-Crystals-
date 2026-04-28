import numpy as np
import torch

from difflc import default_params, run_dc_protocol_diff, v2Q, get_angles


def test_direct_run_is_finite():
    params = default_params()
    run = run_dc_protocol_diff(
        params.L1,
        params.L2,
        params.L3,
        params.gamma1,
        params.W_surf,
        V_factor=3.0,
        T_on=0.02,
        T_off=0.0,
        dt_fw=1e-3,
        params=params,
    )
    assert torch.isfinite(run["states"][-1]).all()
    assert torch.isfinite(run["I_cross"]).all()


def test_final_state_has_angles():
    params = default_params()
    run = run_dc_protocol_diff(
        params.L1,
        params.L2,
        params.L3,
        params.gamma1,
        params.W_surf,
        V_factor=2.0,
        T_on=0.01,
        T_off=0.0,
        dt_fw=1e-3,
        params=params,
    )
    Q_end = v2Q(run["states"][-1])
    theta, phi = get_angles(Q_end)
    assert np.isfinite(theta.detach().cpu().numpy()).all()
    assert np.isfinite(phi.detach().cpu().numpy()).all()
