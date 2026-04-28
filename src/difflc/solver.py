"""Differentiable Landau-de Gennes solver with weak anchoring."""

from __future__ import annotations

import numpy as np
import torch

from .qtensor import ang2Q, Q2v, v2Q, renorm_Q, extract_director
from .utils import TNParameters, default_params, build_z_axis, as_torch_scalar, compute_voltage_thresholds


def _resolve_params(params: TNParameters | None) -> TNParameters:
    return default_params() if params is None else params


def proj_ST(H: torch.Tensor) -> torch.Tensor:
    Hs = 0.5 * (H + H.transpose(-2, -1))
    tr = torch.einsum('...ii', Hs) / 3.0
    eye = torch.eye(3, dtype=Hs.dtype, device=Hs.device)
    return Hs - tr.view(-1, 1, 1) * eye


def mol_field_diff(
    Q: torch.Tensor,
    E,
    L1v,
    L2v,
    L3v,
    *,
    params: TNParameters | None = None,
):
    params = _resolve_params(params)
    L1v = as_torch_scalar(L1v, device=Q.device, dtype=Q.dtype)
    L2v = as_torch_scalar(L2v, device=Q.device, dtype=Q.dtype)
    L3v = as_torch_scalar(L3v, device=Q.device, dtype=Q.dtype)
    E = as_torch_scalar(E, device=Q.device, dtype=Q.dtype)

    dz = params.d_cell / (params.Nz - 1)
    zero_pad = torch.zeros(1, 3, 3, dtype=Q.dtype, device=Q.device)
    Qpp = torch.cat([zero_pad, (Q[2:] + Q[:-2] - 2.0 * Q[1:-1]) / dz**2, zero_pad], dim=0)
    Qp = torch.cat([zero_pad, (Q[2:] - Q[:-2]) / (2.0 * dz), zero_pad], dim=0)

    mask_col2 = torch.zeros(3, 3, dtype=Q.dtype, device=Q.device)
    mask_col2[:, 2] = 1.0
    mask_row2 = torch.zeros(3, 3, dtype=Q.dtype, device=Q.device)
    mask_row2[2, :] = 1.0
    mask_33 = torch.zeros(3, 3, dtype=Q.dtype, device=Q.device)
    mask_33[2, 2] = 1.0

    h_L1 = L1v * Qpp
    h_L2 = 0.5 * L2v * Qpp[:, :, 2].unsqueeze(-1) * mask_col2
    h_L2 = h_L2 + 0.5 * L2v * Qpp[:, 2, :].unsqueeze(-2) * mask_row2

    Q33 = Q[:, 2, 2].view(-1, 1, 1)
    Q33p = Qp[:, 2, 2].view(-1, 1, 1)
    gsq = (Qp**2).sum(dim=(-2, -1))

    h_L3 = L3v * Q33 * Qpp + L3v * Q33p * Qp
    h_L3_corr = -0.5 * L3v * gsq.view(-1, 1, 1) * mask_33
    h_E = 0.5 * params.EPS_0 * (params.deps / params.S) * (E**2) * mask_33
    h = h_L1 + h_L2 + h_L3 + h_L3_corr + h_E

    bulk_mask = torch.ones(params.Nz, 1, 1, dtype=Q.dtype, device=Q.device)
    bulk_mask[0] = 0.0
    bulk_mask[-1] = 0.0
    h = h * bulk_mask
    return h, Qpp


def build_A_diff(dt_val, L1v, gamma1v, Wv, *, params: TNParameters | None = None) -> torch.Tensor:
    params = _resolve_params(params)
    dz = params.d_cell / (params.Nz - 1)
    c = dt_val * L1v / (gamma1v * dz**2)
    wc = dt_val * Wv / (gamma1v * dz * 0.5)

    dtype = c.dtype if torch.is_tensor(c) else torch.float64
    device = c.device if torch.is_tensor(c) else None
    interior_main = torch.ones(params.Nz - 2, dtype=dtype, device=device) + 2.0 * c
    bdry_main = torch.ones(1, dtype=interior_main.dtype, device=interior_main.device) * (1.0 + wc + c)
    diag_main = torch.cat([bdry_main, interior_main, bdry_main])
    diag_off = torch.ones(params.Nz - 1, dtype=diag_main.dtype, device=diag_main.device) * (-c)
    A = torch.diag(diag_main) + torch.diag(diag_off, 1) + torch.diag(diag_off, -1)
    return A


def extract_director_diff(Q: torch.Tensor, S_val: float | None = None) -> torch.Tensor:
    return extract_director(Q)


def renorm_Q_diff(Q: torch.Tensor, S_val: float, *, params: TNParameters | None = None) -> torch.Tensor:
    params = _resolve_params(params)
    n = extract_director_diff(Q, S_val)
    I3 = torch.eye(3, dtype=Q.dtype, device=Q.device)
    nn_outer = n.unsqueeze(-1) * n.unsqueeze(-2)
    return S_val * (nn_outer - (1.0 / 3.0) * I3.unsqueeze(0).expand_as(nn_outer))


def v0_init_fn(twist_angle, *, params: TNParameters | None = None) -> torch.Tensor:
    params = _resolve_params(params)
    z_axis = build_z_axis(params, dtype=torch.float64)
    tilt0 = torch.full((params.Nz,), params.pretilt, dtype=torch.float64, device=z_axis.device)
    twist0 = (z_axis / params.d_cell) * twist_angle
    return Q2v(ang2Q(tilt0, twist0, S_val=params.S, params=params))


def get_boundary_Q(twist_angle, *, params: TNParameters | None = None):
    params = _resolve_params(params)
    Qsb = ang2Q(torch.tensor([params.pretilt], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), S_val=params.S, params=params)[0]
    Qst = ang2Q(torch.tensor([params.pretilt], dtype=torch.float64), torch.tensor([twist_angle], dtype=torch.float64), S_val=params.S, params=params)[0]
    return Qsb, Qst


def step_diff(
    v: torch.Tensor,
    A_mat: torch.Tensor,
    dt_val,
    E_val,
    L1v,
    L2v,
    L3v,
    gamma1v,
    Wv,
    twist_angle,
    *,
    params: TNParameters | None = None,
):
    params = _resolve_params(params)
    L1v = as_torch_scalar(L1v, device=v.device, dtype=v.dtype)
    L2v = as_torch_scalar(L2v, device=v.device, dtype=v.dtype)
    L3v = as_torch_scalar(L3v, device=v.device, dtype=v.dtype)
    gamma1v = as_torch_scalar(gamma1v, device=v.device, dtype=v.dtype)
    Wv = as_torch_scalar(Wv, device=v.device, dtype=v.dtype)
    E_val = as_torch_scalar(E_val, device=v.device, dtype=v.dtype)
    dt_val = as_torch_scalar(dt_val, device=v.device, dtype=v.dtype)

    Q = v2Q(v)
    h, Qpp = mol_field_diff(Q, E_val, L1v, L2v, L3v, params=params)
    h_ex = proj_ST(h) - proj_ST(L1v * Qpp)

    rhs_full = v + (dt_val / gamma1v) * Q2v(h_ex)
    dz = params.d_cell / (params.Nz - 1)
    wc = dt_val * Wv / (gamma1v * dz * 0.5)
    Qsb, Qst = get_boundary_Q(twist_angle, params=params)
    rhs_bot = (v[0] + wc * Q2v(Qsb.unsqueeze(0))[0]).unsqueeze(0)
    rhs_top = (v[-1] + wc * Q2v(Qst.unsqueeze(0))[0]).unsqueeze(0)
    rhs = torch.cat([rhs_bot, rhs_full[1:-1], rhs_top], dim=0)

    v_new = torch.linalg.solve(A_mat, rhs)
    return Q2v(renorm_Q_diff(v2Q(v_new), params.S, params=params))


def get_angles_from_Q(Q: torch.Tensor, *, params: TNParameters | None = None):
    params = _resolve_params(params)
    n = extract_director(Q)
    theta = torch.asin(torch.clamp(n[:, 2], -1.0, 1.0))
    phi = torch.atan2(n[:, 1], n[:, 0])
    return theta, phi


def run_dc_protocol_diff(
    L1v,
    L2v,
    L3v,
    gamma1v,
    Wv,
    V_factor,
    T_on,
    T_off,
    dt_fw,
    pol_angle=np.pi / 4,
    twist_angle=np.pi / 2,
    *,
    params: TNParameters | None = None,
):
    from .optics import jones_diff_from_Q

    params = _resolve_params(params)
    V_th_planar, V_th_tn = compute_voltage_thresholds(params)
    A_mat = build_A_diff(dt_fw, L1v, gamma1v, Wv, params=params)
    v_th_ref = V_th_planar if float(twist_angle) == 0.0 else V_th_tn
    Ei = torch.tensor(V_factor * v_th_ref / params.d_cell, dtype=torch.float64, device=A_mat.device)
    E_zero = torch.tensor(0.0, dtype=torch.float64, device=A_mat.device)

    v = v0_init_fn(twist_angle, params=params).to(dtype=torch.float64, device=A_mat.device)
    I_cross_list = []
    I_parallel_list = []
    states = []
    time_axis = []

    n_on = int(round(float(T_on) / float(dt_fw)))
    n_off = int(round(float(T_off) / float(dt_fw)))
    t_curr = 0.0

    for _ in range(n_on):
        v = step_diff(v, A_mat, dt_fw, Ei, L1v, L2v, L3v, gamma1v, Wv, twist_angle, params=params)
        ic, ip = jones_diff_from_Q(v2Q(v), pol_angle=pol_angle, params=params)
        I_cross_list.append(ic)
        I_parallel_list.append(ip)
        states.append(v)
        time_axis.append(t_curr)
        t_curr += float(dt_fw)

    for _ in range(n_off):
        v = step_diff(v, A_mat, dt_fw, E_zero, L1v, L2v, L3v, gamma1v, Wv, twist_angle, params=params)
        ic, ip = jones_diff_from_Q(v2Q(v), pol_angle=pol_angle, params=params)
        I_cross_list.append(ic)
        I_parallel_list.append(ip)
        states.append(v)
        time_axis.append(t_curr)
        t_curr += float(dt_fw)

    return {
        "time": torch.tensor(time_axis, dtype=torch.float64, device=A_mat.device),
        "I_cross": torch.stack(I_cross_list),
        "I_parallel": torch.stack(I_parallel_list),
        "states": torch.stack(states),
    }
