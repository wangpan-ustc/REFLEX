"""Numpy-only physics backend for the CCAA antenna task.

Mirrors the formulas used in paper/plot_array_factor_surfaces.py and adds
SLL / FNBW computation. No torch dependency.

Design vector layout (normalized, all entries in [0,1]). Three accepted
lengths, detected automatically (N = num_elements, R = num_rings):

    base  (N + R)    : [amp_codes (N), radius_codes (R)]
    rot   (N + 2R)   : base + [ring_rotation_codes (R)]
    full  (2N + 2R)  : rot  + [element_phase_codes (N)]

Physics:
    amps   = amp_min + (1 - amp_min) * amp_code
    radii  = cumsum( min_spacing + extra_spacing * radius_code )   (lambda)
    ring rotation: ring k's elements are offset by
        delta_k = rot_code_k * (2*pi / count_k)     (one inter-element period)
    element phase: psi_n = (phase_code_n - 0.5) * 2*pi   in [-pi, pi]
    AF(theta, phi) = sum_n a_n * exp(j * (k r_n sin(theta) cos(phi - phi_n) + psi_n))

Legacy behaviour is exactly preserved: base-length vectors decode to zero
rotations and zero phases, and for zero-phase designs the broadside peak
equals (sum a)^2 so all historical scores are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CCAAConfig:
    ring_counts: tuple[int, ...]
    min_spacing: float = 0.25
    extra_spacing: float = 0.65
    amp_min: float = 0.05
    fnbw_target_deg: float = 28.0
    score_alpha: float = 0.3       # weight on directivity gain (dB)
    score_beta: float = 0.05       # weight on FNBW deviation (deg)

    @property
    def num_rings(self) -> int:
        return len(self.ring_counts)

    @property
    def num_elements(self) -> int:
        return int(sum(self.ring_counts))

    @property
    def design_dim(self) -> int:
        return self.num_elements + self.num_rings

    @property
    def design_dim_rot(self) -> int:
        """base + per-ring rotation codes."""
        return self.num_elements + 2 * self.num_rings

    @property
    def design_dim_full(self) -> int:
        """base + rotations + per-element phase codes."""
        return 2 * self.num_elements + 2 * self.num_rings


# ---------------------------------------------------------------------------
def decode_design(x: np.ndarray, cfg: CCAAConfig) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n, r = cfg.num_elements, cfg.num_rings
    if x.size not in (cfg.design_dim, cfg.design_dim_rot, cfg.design_dim_full):
        raise ValueError(
            f"design vector has {x.size} entries, expected one of "
            f"{cfg.design_dim} (base), {cfg.design_dim_rot} (+rotations), "
            f"{cfg.design_dim_full} (+rotations+phases)"
        )
    amp_code = np.clip(x[:n], 0.0, 1.0)
    radius_code = np.clip(x[n : n + r], 0.0, 1.0)
    if x.size >= cfg.design_dim_rot:
        rot_code = np.clip(x[n + r : n + 2 * r], 0.0, 1.0)
    else:
        rot_code = np.zeros(r)
    if x.size == cfg.design_dim_full:
        phase_code = np.clip(x[n + 2 * r :], 0.0, 1.0)
        phases = (phase_code - 0.5) * 2.0 * np.pi   # psi_n in [-pi, pi]
    else:
        phases = np.zeros(n)

    amps = cfg.amp_min + (1.0 - cfg.amp_min) * amp_code
    radii = np.cumsum(cfg.min_spacing + cfg.extra_spacing * radius_code)
    element_radii = []
    element_phi = []
    for ring_r, count, rc in zip(radii, cfg.ring_counts, rot_code):
        # rotation offset spans one inter-element period, so rot_code=0 and
        # rot_code=1 give the same geometry (continuous, no discontinuity).
        delta = rc * 2.0 * np.pi / count
        phis = 2.0 * np.pi * np.arange(count) / count + delta
        element_radii.append(np.full(count, ring_r))
        element_phi.append(phis)
    return {
        "amps": amps,
        "radii": radii,
        "rotations": rot_code,
        "phases": phases,
        "element_radii": np.concatenate(element_radii),
        "element_phi": np.concatenate(element_phi),
    }


# ---------------------------------------------------------------------------
def array_factor_power(
    amps: np.ndarray,
    element_radii: np.ndarray,
    element_phi: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    phases: np.ndarray | None = None,
) -> np.ndarray:
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    field = np.zeros_like(theta_grid, dtype=np.complex128)
    k = 2.0 * np.pi
    if phases is None:
        phases = np.zeros(len(amps))
    for amp, r, ep, psi in zip(amps, element_radii, element_phi, phases):
        phase = k * r * np.sin(theta_grid) * np.cos(phi_grid - ep) + psi
        field += amp * np.exp(1j * phase)
    return np.abs(field) ** 2


def directivity_db(
    amps: np.ndarray,
    element_radii: np.ndarray,
    element_phi: np.ndarray,
    phases: np.ndarray | None = None,
) -> float:
    theta = np.linspace(0.0, np.pi, 121)
    phi = np.linspace(0.0, 2.0 * np.pi, 241)
    power = array_factor_power(amps, element_radii, element_phi, theta, phi, phases)
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x renamed trapz
    integral = _trapz(
        _trapz(power * np.sin(theta)[:, None], phi, axis=1), theta, axis=0
    )
    # Peak power over the sampled sphere. For zero-phase (broadside) designs
    # this equals (sum amps)^2 exactly, preserving all legacy scores; for
    # phased designs it is the physically correct peak.
    main_power = max(float(np.max(power)), 1e-12)
    d = 4.0 * np.pi * main_power / max(float(integral), 1e-12)
    return 10.0 * np.log10(max(d, 1e-12))


def elevation_cut_db(
    amps: np.ndarray,
    element_radii: np.ndarray,
    element_phi: np.ndarray,
    phi_cut: float = 0.0,
    n_theta: int = 721,
    phases: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta_deg, AF in dB normalized to peak)."""
    theta = np.linspace(0.0, np.pi / 2.0, n_theta)
    phi = np.array([phi_cut])
    p = array_factor_power(amps, element_radii, element_phi, theta, phi, phases)[:, 0]
    p_peak = max(float(np.max(p)), 1e-12)
    af_db = 10.0 * np.log10(np.maximum(p / p_peak, 1e-8))
    return np.degrees(theta), af_db


def sll_and_fnbw(theta_deg: np.ndarray, af_db: np.ndarray) -> tuple[float, float]:
    """Compute SLL (peak sidelobe level, dB; negative is better) and
    FNBW (first-null beamwidth, degrees) from a single elevation cut.

    Strategy:
        - find the main-beam null on the right side (first theta where
          af_db crosses below -20 dB after the global peak).
        - SLL = max(af_db) within [first_null, end].
        - FNBW = 2 * theta_first_null  (since cut is symmetric around 0).
    """
    af_db = np.asarray(af_db, dtype=np.float64)
    peak_idx = int(np.argmax(af_db))
    null_idx = peak_idx + 1
    while null_idx < len(af_db) - 1 and af_db[null_idx] > -20.0:
        null_idx += 1
        if af_db[null_idx] - af_db[null_idx - 1] > 0 and af_db[null_idx] < -10.0:
            break
    fnbw = 2.0 * float(theta_deg[null_idx] - theta_deg[peak_idx])
    sll = float(np.max(af_db[null_idx:])) if null_idx < len(af_db) else float("nan")
    return sll, fnbw


# ---------------------------------------------------------------------------
def evaluate_design(x: np.ndarray, cfg: CCAAConfig) -> dict[str, Any]:
    dec = decode_design(x, cfg)
    theta_deg, af_db_phi0 = elevation_cut_db(
        dec["amps"], dec["element_radii"], dec["element_phi"], phi_cut=0.0,
        phases=dec["phases"],
    )
    _, af_db_phi90 = elevation_cut_db(
        dec["amps"], dec["element_radii"], dec["element_phi"], phi_cut=np.pi / 2,
        phases=dec["phases"],
    )
    sll0, fnbw0 = sll_and_fnbw(theta_deg, af_db_phi0)
    sll90, fnbw90 = sll_and_fnbw(theta_deg, af_db_phi90)
    sll = float(max(sll0, sll90))           # worst-case SLL across cuts
    fnbw = float(0.5 * (fnbw0 + fnbw90))
    d_db = directivity_db(
        dec["amps"], dec["element_radii"], dec["element_phi"], dec["phases"]
    )

    score = (
        -sll                             # smaller (more negative) SLL is better
        + cfg.score_alpha * d_db
        - cfg.score_beta * abs(fnbw - cfg.fnbw_target_deg)
    )

    return {
        "score": score,
        "sll_db": sll,
        "fnbw_deg": fnbw,
        "directivity_db": d_db,
        "amps": dec["amps"],
        "phases": dec["phases"],
        "rotations": dec["rotations"],
        "radii": dec["radii"],
        "element_radii": dec["element_radii"],
        "element_phi": dec["element_phi"],
        "theta_deg": theta_deg,
        "af_db_phi0": af_db_phi0,
        "af_db_phi90": af_db_phi90,
    }
