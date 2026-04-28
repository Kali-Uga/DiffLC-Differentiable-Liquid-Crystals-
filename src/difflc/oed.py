"""Optional OED helpers for constant-field experiment design."""

from __future__ import annotations

import numpy as np
import torch

from .solver import run_dc_protocol_diff
from .utils import TNParameters, default_params


def _resolve_params(params: TNParameters | None) -> TNParameters:
    return default_params() if params is None else params


def run_single_np(param_vec, V_factor, twist_a, *, T_on=0.2, T_off=0.4, dt_fw=1e-3, params: TNParameters | None = None):
    params = _resolve_params(params)
    param_vec = np.asarray(param_vec, dtype=float)
    run = run_dc_protocol_diff(
        param_vec[0],
        param_vec[1],
        param_vec[2],
        param_vec[3],
        param_vec[4],
        V_factor=V_factor,
        T_on=T_on,
        T_off=T_off,
        dt_fw=dt_fw,
        twist_angle=twist_a,
        params=params,
    )
    return run["I_cross"].detach().cpu().numpy()


def compute_J_fd_joint(
    p_true,
    Vf_planar_list,
    Vf_twist_list,
    *,
    rel_eps=1e-3,
    T_on=0.2,
    T_off=0.4,
    dt_fw=1e-3,
    params: TNParameters | None = None,
):
    params = _resolve_params(params)
    n_params = len(p_true)

    def _sensitivities(vf, twist_angle):
        I_base = run_single_np(p_true, vf, twist_angle, T_on=T_on, T_off=T_off, dt_fw=dt_fw, params=params)
        J = np.zeros((len(I_base), n_params))
        for k in range(n_params):
            p_p = np.array(p_true, dtype=float).copy()
            p_m = np.array(p_true, dtype=float).copy()
            p_p[k] *= (1.0 + rel_eps)
            p_m[k] *= (1.0 - rel_eps)
            J[:, k] = (
                run_single_np(p_p, vf, twist_angle, T_on=T_on, T_off=T_off, dt_fw=dt_fw, params=params)
                - run_single_np(p_m, vf, twist_angle, T_on=T_on, T_off=T_off, dt_fw=dt_fw, params=params)
            ) / (2.0 * rel_eps * p_true[k])
        return J

    J_p_list = [_sensitivities(Vf, 0.0) for Vf in Vf_planar_list]
    J_t_list = [_sensitivities(Vf, np.pi / 2) for Vf in Vf_twist_list]
    return J_p_list, J_t_list


def build_fim_joint(p_true, Vf_planar_list, Vf_twist_list, noise_std, *, params: TNParameters | None = None):
    J_p_list, J_t_list = compute_J_fd_joint(p_true, Vf_planar_list, Vf_twist_list, params=params)
    n_params = len(p_true)
    FIM = np.zeros((n_params, n_params))
    for J in J_p_list:
        FIM += (J / noise_std).T @ (J / noise_std)
    for J in J_t_list:
        FIM += (J / noise_std).T @ (J / noise_std)
    return FIM


def evaluate_joint_design_tensor(
    V_tensor: torch.Tensor,
    p_true,
    Vf_planar_count: int,
    noise_std: float,
    *,
    T_on=0.2,
    T_off=0.4,
    dt_fw=1e-3,
    params: TNParameters | None = None,
):
    params = _resolve_params(params)
    Vf_p = V_tensor[:Vf_planar_count].tolist()
    Vf_t = V_tensor[Vf_planar_count:].tolist()
    p_true = np.asarray(p_true, dtype=float)
    F_norm = build_fim_joint(p_true, Vf_p, Vf_t, noise_std, params=params)
    D_scale = np.diag(p_true)
    F_scaled = D_scale @ F_norm @ D_scale + 1e-8 * np.eye(len(p_true))
    return torch.tensor([np.log10(np.linalg.det(F_scaled))], dtype=torch.float64)
