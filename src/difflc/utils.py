"""Physical parameters and configuration for LC simulations.

Provides E7 material configuration, cell specifications, and protocol
definitions used throughout the DiffLC framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Material configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E7Config:
    """E7 liquid crystal material and simulation parameters."""

    # Spatial grid
    Nz: int = 41

    # Frank elastic constants [N]
    K11: float = 10.64e-12
    K22: float = 5.18e-12
    K33: float = 16.50e-12

    # Rotational viscosity [Pa·s]
    gamma1: float = 0.2036

    # Surface anchoring energy [J/m²]
    W: float = 1.388e-4

    # Equilibrium scalar order parameter
    S0: float = 0.6

    # Dielectric constants
    eps_par: float = 19.54
    eps_perp: float = 5.17

    # Optical refractive indices (fixed over wavelength in this benchmark)
    ne: float = 1.7287
    no: float = 1.5182

    # Landau-de Gennes thermotropic coefficients [J/m³]
    # Convention: f_bulk = a tr(Q²) + (2b/3) tr(Q³) + (c/2) [tr(Q²)]²
    bulk_b: float = -0.64e6
    bulk_c: float = 0.40e6

    # Boundary geometry
    pretilt_deg: float = 2.0

    # Vacuum permittivity [F/m]
    EPS_0: float = 8.854187817e-12

    # --- Derived properties ---------------------------------------------------

    @property
    def deps(self) -> float:
        """Dielectric anisotropy Δε = ε∥ − ε⊥."""
        return self.eps_par - self.eps_perp

    @property
    def delta_n2(self) -> float:
        """Optical birefringence squared: Δn² = ne² − no²."""
        return self.ne**2 - self.no**2

    @property
    def eps_iso_opt(self) -> float:
        """Isotropic average of optical dielectric tensor."""
        return self.no**2 + self.delta_n2 / 3.0

    @property
    def pretilt_rad(self) -> float:
        return math.radians(self.pretilt_deg)

    @property
    def bulk_a(self) -> float:
        """Thermotropic coefficient *a* chosen so that S0 is a stationary point:
        3a + b·S0 + 2c·S0² = 0."""
        return -(self.bulk_b * self.S0 + 2.0 * self.bulk_c * self.S0**2) / 3.0

    @property
    def gamma_Q(self) -> float:
        """Q-tensor rotational viscosity γ_Q = γ1 / (2 S0²).

        For a uniaxial Q = S(nn − I/3) with fixed S, |Q̇|² = 2S²|ṅ|², so
        equating the Q-dissipation ½γ_Q|Q̇|² to the director dissipation
        ½γ1|ṅ|² gives γ_Q = γ1/(2S²). The corresponding mobility used in the
        solver is μ = 1/γ_Q = 2S0²/γ1. (The earlier γ1/S0 was too large by a
        factor 2S0, making the dynamics 2S0≈1.2× too slow at S0=0.6 and giving
        recovered γ1 a spurious ∝1/S0 dependence.) Consistent with the anchoring
        conversion W_Q = W_RP/(2S0²)."""
        return self.gamma1 / (2.0 * self.S0**2)


# ---------------------------------------------------------------------------
# Cell and protocol specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    """Physical specification of a single LC cell."""

    name: str
    d_cell: float  # cell gap [m]
    twist_deg: float  # total twist angle [°]
    voltage_ratio: float  # V / V_threshold (used once to compute V_abs)


@dataclass(frozen=True)
class Protocol:
    """Fully resolved voltage protocol for a cell."""

    name: str
    cell: CellSpec
    V_threshold_true: float
    V_abs: float
    E_abs: float


# ---------------------------------------------------------------------------
# Timing configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingConfig:
    """Default time-stepping and recording parameters."""

    dt: float = 5e-4  # solver time step [s]  (0.5 ms)
    record_every: int = 8  # record every N solver steps  (→ 4 ms)
    T_on: float = 0.300  # voltage-on duration [s]
    T_off: float = 0.200  # relaxation duration [s]
    T_eq: float = 0.100  # zero-field pre-equilibration [s]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def K_to_L(K11: float, K22: float, K33: float, S0: float) -> Tuple[float, float, float]:
    """Convert Frank constants (K) to Landau-de Gennes elastic constants (L)."""
    L1 = (K33 - K11 + 3.0 * K22) / (6.0 * S0**2)
    L2 = (K11 - K22) / S0**2
    L3 = (K33 - K11) / (2.0 * S0**3)
    return L1, L2, L3


def threshold_voltage(cell: CellSpec, cfg: E7Config) -> float:
    """Analytical Fréedericksz threshold voltage for a cell."""
    if abs(cell.twist_deg) < 1e-12:
        K_eff = cfg.K11
    else:
        K_eff = cfg.K11 + 0.25 * (cfg.K33 - 2.0 * cfg.K22)
    return math.pi * math.sqrt(K_eff / (cfg.EPS_0 * cfg.deps))


def build_protocols(cells, cfg: E7Config):
    """Build resolved Protocol list from CellSpec list."""
    protocols = []
    for cell in cells:
        Vth = threshold_voltage(cell, cfg)
        Vabs = cell.voltage_ratio * Vth
        protocols.append(
            Protocol(
                name=f"{cell.name}_u{cell.voltage_ratio:.3g}",
                cell=cell,
                V_threshold_true=Vth,
                V_abs=Vabs,
                E_abs=Vabs / cell.d_cell,
            )
        )
    return protocols


# ---------------------------------------------------------------------------
# Default instances
# ---------------------------------------------------------------------------

DEFAULT_CFG = E7Config()

DEFAULT_CELLS = [
    CellSpec("TN90_10um", d_cell=10e-6, twist_deg=90.0, voltage_ratio=2.19),
    CellSpec("PLANAR0_10um", d_cell=10e-6, twist_deg=0.0, voltage_ratio=2.11),
    CellSpec("PLANAR0_5um_FAST", d_cell=5e-6, twist_deg=0.0, voltage_ratio=1.209),
]

DEFAULT_TIMING = TimingConfig()


def default_cfg() -> E7Config:
    """Return the default E7 configuration."""
    return DEFAULT_CFG


def default_cells():
    """Return the default three-cell campaign."""
    return list(DEFAULT_CELLS)


def default_timing() -> TimingConfig:
    """Return default timing configuration."""
    return DEFAULT_TIMING


# ---------------------------------------------------------------------------
# Observation grid defaults
# ---------------------------------------------------------------------------

WAVELENGTHS_NM = (450.0, 532.0, 589.0, 642.0, 700.0)
INCIDENCE_DEG = (0.0, 35.0)


def jones_linear(angle_deg: float):
    """Jones vector for linearly polarized light."""
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=complex)


def jones_circular(handedness: str = "R"):
    """Jones vector for circularly polarized light."""
    if handedness.upper().startswith("R"):
        return np.array([1.0, -1.0j], dtype=complex) / math.sqrt(2)
    return np.array([1.0, 1.0j], dtype=complex) / math.sqrt(2)


DEFAULT_INPUT_POLS = np.array(
    [
        jones_linear(0.0),
        jones_linear(45.0),
        jones_linear(90.0),
        jones_circular("R"),
    ],
    dtype=complex,
)

POL_LABELS = ("0°", "45°", "90°", "RCP")
