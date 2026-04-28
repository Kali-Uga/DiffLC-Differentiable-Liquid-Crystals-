import numpy as np
import torch

from difflc import ang2Q, jones_diff_from_Q, default_params


def test_jones_energy_conservation():
    params = default_params()
    th = torch.full((50,), 0.3, dtype=torch.float64)
    ph = torch.full((50,), np.pi / 4, dtype=torch.float64)
    Q = ang2Q(th, ph, S_val=params.S, params=params)
    Ic, Ip = jones_diff_from_Q(Q, params=params)
    assert abs((Ic + Ip).item() - 1.0) < 1e-10
