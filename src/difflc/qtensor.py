"""Q-tensor and director conversion helpers."""

from __future__ import annotations

import torch

from .utils import TNParameters, default_params


def _resolve_params(params: TNParameters | None) -> TNParameters:
    return default_params() if params is None else params


def ang2Q(th, ph, S_val=None, *, params: TNParameters | None = None):
    params = _resolve_params(params)
    if S_val is None:
        S_val = params.S
    nx = torch.cos(th) * torch.cos(ph)
    ny = torch.cos(th) * torch.sin(ph)
    nz = torch.sin(th)
    Q = torch.zeros(len(th), 3, 3, dtype=th.dtype, device=th.device)
    Q[:, 0, 0] = S_val * (nx * nx - 1 / 3)
    Q[:, 1, 1] = S_val * (ny * ny - 1 / 3)
    Q[:, 2, 2] = S_val * (nz * nz - 1 / 3)
    Q[:, 0, 1] = Q[:, 1, 0] = S_val * nx * ny
    Q[:, 0, 2] = Q[:, 2, 0] = S_val * nx * nz
    Q[:, 1, 2] = Q[:, 2, 1] = S_val * ny * nz
    return Q


def Q2v(Q: torch.Tensor) -> torch.Tensor:
    return torch.stack([Q[:, 0, 0], Q[:, 0, 1], Q[:, 0, 2], Q[:, 1, 1], Q[:, 1, 2]], dim=1)


def v2Q(v: torch.Tensor) -> torch.Tensor:
    row0 = torch.stack([v[:, 0], v[:, 1], v[:, 2]], dim=-1)
    row1 = torch.stack([v[:, 1], v[:, 3], v[:, 4]], dim=-1)
    row2 = torch.stack([v[:, 2], v[:, 4], -v[:, 0] - v[:, 3]], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def extract_director(Q: torch.Tensor) -> torch.Tensor:
    _, eigvecs = torch.linalg.eigh(Q)
    n = eigvecs[:, :, -1]
    sign = torch.where(n[:, 2:3] < 0.0, -torch.ones_like(n[:, 2:3]), torch.ones_like(n[:, 2:3]))
    return n * sign


def renorm_Q(Q: torch.Tensor, *, params: TNParameters | None = None) -> torch.Tensor:
    params = _resolve_params(params)
    n = extract_director(Q)
    I3 = torch.eye(3, dtype=Q.dtype, device=Q.device)
    nn_outer = n.unsqueeze(-1) * n.unsqueeze(-2)
    return params.S * (nn_outer - (1.0 / 3.0) * I3.unsqueeze(0).expand_as(nn_outer))


def get_angles(Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = extract_director(Q)
    theta = torch.asin(torch.clamp(n[:, 2], -1.0, 1.0))
    phi = torch.atan2(n[:, 1], n[:, 0])
    return theta, phi
