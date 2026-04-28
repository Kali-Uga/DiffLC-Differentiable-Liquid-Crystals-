"""Differential Jones optics for twisted-nematic cells."""

from __future__ import annotations

import numpy as np
import torch

from .qtensor import get_angles
from .utils import TNParameters, default_params


def _resolve_params(params: TNParameters | None) -> TNParameters:
    return default_params() if params is None else params


def effective_index_yamauchi(theta: torch.Tensor, *, params: TNParameters | None = None) -> torch.Tensor:
    params = _resolve_params(params)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return params.no * params.ne / torch.sqrt(params.no**2 * cos_theta**2 + params.ne**2 * sin_theta**2 + 1e-14)


def jones_slice_yamauchi(theta: torch.Tensor, phi: torch.Tensor, dz_val, *, params: TNParameters | None = None) -> torch.Tensor:
    params = _resolve_params(params)
    n_eff = effective_index_yamauchi(theta, params=params)
    retardance = (2.0 * np.pi / params.wl) * (n_eff - params.no) * dz_val

    cp = torch.cos(phi).to(torch.complex128)
    sp = torch.sin(phi).to(torch.complex128)
    e_pos = torch.exp(0.5j * retardance.to(torch.complex128))
    e_neg = torch.exp(-0.5j * retardance.to(torch.complex128))

    row0 = torch.stack([cp**2 * e_pos + sp**2 * e_neg, cp * sp * (e_pos - e_neg)], dim=-1)
    row1 = torch.stack([cp * sp * (e_pos - e_neg), sp**2 * e_pos + cp**2 * e_neg], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def multiply_jones_stack(jones_stack: torch.Tensor) -> torch.Tensor:
    M = jones_stack
    while M.shape[0] > 1:
        n_layers = M.shape[0]
        if n_layers % 2 == 1:
            last_prod = torch.matmul(M[-1], M[-2]).unsqueeze(0)
            M = torch.cat([M[:-2], last_prod], dim=0) if n_layers > 3 else torch.stack([M[0], last_prod[0]], dim=0)
        else:
            M = torch.bmm(M[1::2], M[0::2])
    return M[0]


def jones_matrix_diff(Q: torch.Tensor, *, params: TNParameters | None = None) -> torch.Tensor:
    params = _resolve_params(params)
    theta, phi = get_angles(Q)
    jones_stack = jones_slice_yamauchi(theta, phi, params.d_cell / (params.Nz - 1), params=params)
    return multiply_jones_stack(jones_stack)


def jones_diff_from_Q(Q: torch.Tensor, pol_angle=np.pi / 4, *, params: TNParameters | None = None):
    params = _resolve_params(params)
    J = jones_matrix_diff(Q, params=params)
    cos_p = float(np.cos(pol_angle))
    sin_p = float(np.sin(pol_angle))
    P = torch.tensor([cos_p, sin_p], dtype=torch.complex128, device=Q.device)
    A_cross = torch.tensor([-sin_p, cos_p], dtype=torch.complex128, device=Q.device)
    A_parallel = torch.tensor([cos_p, sin_p], dtype=torch.complex128, device=Q.device)
    E_out = torch.mv(J, P)
    I_cross = (torch.abs(torch.dot(A_cross.conj(), E_out))**2).real.to(torch.float64)
    I_parallel = (torch.abs(torch.dot(A_parallel.conj(), E_out))**2).real.to(torch.float64)
    return I_cross, I_parallel
