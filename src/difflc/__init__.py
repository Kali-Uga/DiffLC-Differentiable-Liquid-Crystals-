"""DiffLC public API."""

from .utils import TNParameters, default_params, compute_K_constants, compute_voltage_thresholds, build_z_axis, with_updates
from .qtensor import ang2Q, Q2v, v2Q, renorm_Q, get_angles, extract_director
from .solver import (
    proj_ST,
    mol_field_diff,
    build_A_diff,
    step_diff,
    run_dc_protocol_diff,
    v0_init_fn,
    get_boundary_Q,
    get_angles_from_Q,
)
from .optics import (
    effective_index_yamauchi,
    jones_slice_yamauchi,
    multiply_jones_stack,
    jones_matrix_diff,
    jones_diff_from_Q,
)
from .oed import run_single_np, compute_J_fd_joint, build_fim_joint, evaluate_joint_design_tensor
from .inverse import build_targets, solve_trf_worker

__version__ = "0.1.0"

__all__ = [
    "TNParameters",
    "default_params",
    "compute_K_constants",
    "compute_voltage_thresholds",
    "build_z_axis",
    "with_updates",
    "ang2Q",
    "Q2v",
    "v2Q",
    "renorm_Q",
    "get_angles",
    "extract_director",
    "proj_ST",
    "mol_field_diff",
    "build_A_diff",
    "step_diff",
    "run_dc_protocol_diff",
    "v0_init_fn",
    "get_boundary_Q",
    "get_angles_from_Q",
    "effective_index_yamauchi",
    "jones_slice_yamauchi",
    "multiply_jones_stack",
    "jones_matrix_diff",
    "jones_diff_from_Q",
    "run_single_np",
    "compute_J_fd_joint",
    "build_fim_joint",
    "evaluate_joint_design_tensor",
    "build_targets",
    "solve_trf_worker",
    "__version__",
]
