"""Physical parameters and small helpers for TN 5CB simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class TNParameters:
    Nz: int = 80
    d_cell: float = 20e-6
    L1: float = 8.037e-12
    L2: float = 8.096e-12
    L3: float = 6.613e-12
    gamma1: float = 0.077
    EPS_0: float = 8.854e-12
    S: float = 0.533
    deps: float = 13.0
    ne: float = 1.736
    no: float = 1.5442
    wl: float = 633e-9
    pretilt_deg: float = 2.0
    twist_deg: float = 90.0
    W_surf: float = 5e-5

    @property
    def pretilt(self) -> float:
        return float(np.deg2rad(self.pretilt_deg))

    @property
    def twist_tot(self) -> float:
        return float(np.deg2rad(self.twist_deg))


DEFAULT_PARAMS = TNParameters()


def default_params() -> TNParameters:
    return DEFAULT_PARAMS


def with_updates(params: TNParameters, **kwargs) -> TNParameters:
    return replace(params, **kwargs)


def compute_K_constants(params: TNParameters) -> Tuple[float, float, float]:
    S = params.S
    K1 = S**2 * (2 * params.L1 + params.L2) - (2.0 / 3.0) * S**3 * params.L3
    K2 = 2 * S**2 * params.L1 - (2.0 / 3.0) * S**3 * params.L3
    K3 = S**2 * (2 * params.L1 + params.L2) + (4.0 / 3.0) * S**3 * params.L3
    return K1, K2, K3


def compute_voltage_thresholds(params: TNParameters) -> Tuple[float, float]:
    K1, K2, K3 = compute_K_constants(params)
    V_th_planar = np.pi * np.sqrt(K1 / (params.EPS_0 * params.deps))
    V_th_tn = np.pi * np.sqrt((K1 + (K3 - 2 * K2) / 4.0) / (params.EPS_0 * params.deps))
    return float(V_th_planar), float(V_th_tn)


def build_z_axis(params: TNParameters, *, device=None, dtype=torch.float64) -> torch.Tensor:
    return torch.linspace(0.0, params.d_cell, params.Nz, dtype=dtype, device=device)


def as_torch_scalar(value, *, device=None, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)
