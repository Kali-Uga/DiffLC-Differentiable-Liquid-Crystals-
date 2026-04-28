"""Optional TRF inverse solver helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch.func import jacfwd

from .oed import run_single_np
from .solver import run_dc_protocol_diff
from .utils import TNParameters, default_params


def _resolve_params(params: TNParameters | None) -> TNParameters:
    return default_params() if params is None else params


def build_targets(p_true, Vf_planar_opt, Vf_twist_opt, noise_std, *, seed=42, params: TNParameters | None = None):
    params = _resolve_params(params)
    p_true = np.asarray(p_true, dtype=float)
    rng = np.random.default_rng(seed)
    I_p_opt = [run_single_np(p_true, Vf, 0.0, params=params) for Vf in Vf_planar_opt]
    I_t_opt = [run_single_np(p_true, Vf, np.pi / 2, params=params) for Vf in Vf_twist_opt]
    target_p_list = [Ip + rng.normal(scale=noise_std, size=Ip.shape) for Ip in I_p_opt]
    target_t_list = [It + rng.normal(scale=noise_std, size=It.shape) for It in I_t_opt]
    target_full = np.concatenate(target_p_list + target_t_list)
    return I_p_opt, I_t_opt, target_p_list, target_t_list, target_full


def solve_trf_worker(
    x0_log10,
    target_full,
    p_true,
    Vf_planar_opt,
    Vf_twist_opt,
    noise_std,
    *,
    T_on=0.2,
    T_off=0.4,
    dt_fw=1e-3,
    max_nfev=80,
    params: TNParameters | None = None,
):
    from scipy.optimize import least_squares

    params = _resolve_params(params)
    p_true = np.asarray(p_true, dtype=float)
    n_params = len(p_true)
    LOG_LO = np.log10(p_true) - 1.0
    LOG_HI = np.log10(p_true) + 1.0
    MAX_JAC = 1e8
    t_full = torch.tensor(target_full, dtype=torch.float64)

    def res_torch(lp_t):
        p_t = torch.pow(torch.tensor(10.0, dtype=torch.float64), lp_t)
        preds = []
        for Vf in Vf_planar_opt:
            run_p = run_dc_protocol_diff(p_t[0], p_t[1], p_t[2], p_t[3], p_t[4], Vf, T_on, T_off, dt_fw, twist_angle=0.0, params=params)
            preds.append(run_p["I_cross"])
        for Vf in Vf_twist_opt:
            run_t = run_dc_protocol_diff(p_t[0], p_t[1], p_t[2], p_t[3], p_t[4], Vf, T_on, T_off, dt_fw, twist_angle=np.pi / 2, params=params)
            preds.append(run_t["I_cross"])
        return (torch.cat(preds) - t_full) / noise_std

    def res_np(lp_np):
        try:
            lp_t = torch.tensor(lp_np, dtype=torch.float64)
            res = res_torch(lp_t).detach().cpu().numpy()
            return np.nan_to_num(res, nan=1e6, posinf=1e6, neginf=-1e6)
        except Exception:
            return np.full_like(target_full, 1e6, dtype=float)

    def jac_np(lp_np):
        try:
            lp_t = torch.tensor(lp_np, dtype=torch.float64)
            J = jacfwd(res_torch)(lp_t).detach().cpu().numpy()
            J = np.nan_to_num(J, nan=0.0, posinf=MAX_JAC, neginf=-MAX_JAC)
            return np.clip(J, -MAX_JAC, MAX_JAC)
        except Exception:
            return np.zeros((target_full.size, n_params), dtype=float)

    try:
        result = least_squares(
            res_np,
            x0=np.clip(x0_log10, LOG_LO, LOG_HI),
            jac=jac_np,
            method="trf",
            bounds=(LOG_LO, LOG_HI),
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=max_nfev,
            x_scale=1.0,
            loss="linear",
        )

        p_rec = 10**result.x
        rel_err = np.abs(p_rec - p_true) / p_true * 100.0
        return {
            "ok": True,
            "result": result,
            "x": p_rec,
            "err": rel_err,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "result": None,
            "x": np.full(n_params, np.nan),
            "err": np.full(n_params, np.nan),
        }
