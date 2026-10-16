"""Classical textbook baselines for the CCAA design task.

Each seed is a self-contained `design_array(spec)` function string that
initializes the REFLEX population with standard analytical tapers.

References:
    - Taylor (1953), "Design of line-source antennas for narrow beamwidth
      and low side lobes."
    - Dolph (1946), "A current distribution for broadside arrays..."
    - Stutzman & Thiele, "Antenna Theory and Design."
"""

# 1) Taylor-like per-ring linear taper, near-uniform spacing.
SEED_TAYLOR_LINEAR = '''
import numpy as np


def design_array(spec):
    """Smooth linear amplitude taper from inner to outer ring; near-uniform spacing."""
    num_rings   = spec['num_rings']
    ring_counts = spec['ring_counts']

    amps = []
    for k, count in enumerate(ring_counts):
        ring_pos = k / max(num_rings - 1, 1)
        a_k = 0.95 - 0.55 * ring_pos
        amps.extend([a_k] * count)
    amps = np.array(amps)

    radius_codes = np.linspace(0.45, 0.55, num_rings)
    return np.clip(np.concatenate([amps, radius_codes]), 0.0, 1.0)
'''


# 2) Dolph-Chebyshev-inspired taper: outer-ring suppression + super-linear spacing.
SEED_DOLPH_CHEB = '''
import numpy as np


def design_array(spec):
    """Dolph-Chebyshev-inspired: outer-ring suppression + super-linear spacing."""
    num_rings   = spec['num_rings']
    ring_counts = spec['ring_counts']

    amps = []
    for k, count in enumerate(ring_counts):
        ring_pos = k / max(num_rings - 1, 1)
        a_k = float(np.cos(0.5 * np.pi * ring_pos ** 0.7) ** 1.5)
        a_k = max(0.10, min(1.0, a_k))
        amps.extend([a_k] * count)
    amps = np.array(amps)

    rp = np.linspace(0.0, 1.0, num_rings)
    radius_codes = 0.30 + 0.55 * (rp ** 1.5)
    return np.clip(np.concatenate([amps, radius_codes]), 0.0, 1.0)
'''


# 3) Cosine-squared (raised cosine, n=2) taper.
SEED_COSINE_SQ = '''
import numpy as np


def design_array(spec):
    """Raised-cosine^2 amplitude taper; low-SLL baseline."""
    num_rings   = spec['num_rings']
    ring_counts = spec['ring_counts']

    amps = []
    for k, count in enumerate(ring_counts):
        ring_pos = k / max(num_rings - 1, 1)
        a_k = float(np.cos(0.5 * np.pi * ring_pos) ** 2)
        a_k = max(0.05, a_k)
        amps.extend([a_k] * count)
    amps = np.array(amps)

    radius_codes = np.linspace(0.40, 0.60, num_rings)
    return np.clip(np.concatenate([amps, radius_codes]), 0.0, 1.0)
'''


# 4) Truncated Gaussian taper centered on the inner ring.
SEED_GAUSSIAN = '''
import numpy as np


def design_array(spec):
    """Truncated Gaussian amplitude taper across rings; super-linear spacing."""
    num_rings   = spec['num_rings']
    ring_counts = spec['ring_counts']

    sigma = max(num_rings * 0.55, 1.0)
    amps = []
    for k, count in enumerate(ring_counts):
        a_k = float(np.exp(-0.5 * (k / sigma) ** 2))
        a_k = max(0.05, a_k)
        amps.extend([a_k] * count)
    amps = np.array(amps)

    rp = np.linspace(0.0, 1.0, num_rings)
    radius_codes = 0.35 + 0.50 * (rp ** 1.3)
    return np.clip(np.concatenate([amps, radius_codes]), 0.0, 1.0)
'''


# 5) Radial Taylor taper based on physical ring radius.
SEED_PER_ELEMENT_TAYLOR = '''
import numpy as np


def design_array(spec):
    """Taylor-style taper based on physical ring radius."""
    num_rings   = spec['num_rings']
    ring_counts = spec['ring_counts']
    min_sp = spec['min_spacing']
    ext_sp = spec['extra_spacing']

    rp = np.linspace(0.0, 1.0, num_rings)
    radius_codes = 0.40 + 0.40 * (rp ** 1.2)
    radii_lambda = np.cumsum(min_sp + ext_sp * radius_codes)
    R = float(radii_lambda[-1])

    amp_codes = []
    for r_lambda, count in zip(radii_lambda, ring_counts):
        u = float(r_lambda) / R
        a = 0.20 + 0.80 * (np.cos(0.5 * np.pi * u) ** 1.2)
        amp_codes.extend([float(np.clip(a, 0.05, 1.0))] * int(count))

    x = np.concatenate([np.array(amp_codes, dtype=float), radius_codes])
    return np.clip(x, 0.0, 1.0)
'''


# 6) Bessel-basis radial taper.
SEED_BESSEL_TAPER = '''
import numpy as np


def design_array(spec):
    """Bessel-style radial taper."""
    num_rings   = spec['num_rings']
    ring_counts = spec['ring_counts']
    min_sp = spec['min_spacing']
    ext_sp = spec['extra_spacing']

    rp = np.linspace(0.0, 1.0, num_rings)
    radius_codes = 0.40 + 0.40 * (rp ** 1.1)
    radii_lambda = np.cumsum(min_sp + ext_sp * radius_codes)
    R = float(radii_lambda[-1])

    amp_codes = []
    for r_lambda, count in zip(radii_lambda, ring_counts):
        u = float(r_lambda) / R
        a = 0.10 + 0.90 * max(np.cos(0.5 * np.pi * u) ** 1.5, 0.0)
        amp_codes.extend([float(np.clip(a, 0.05, 1.0))] * int(count))

    x = np.concatenate([np.array(amp_codes, dtype=float), radius_codes])
    return np.clip(x, 0.0, 1.0)
'''


CLASSIC_SEEDS = [
    {"name": "taylor-linear",
     "code": SEED_TAYLOR_LINEAR,
     "concept": "Linear per-ring amplitude taper with near-uniform spacing (Taylor-like baseline)."},
    {"name": "dolph-chebyshev",
     "code": SEED_DOLPH_CHEB,
     "concept": "Dolph-Chebyshev-inspired outer-ring suppression and super-linear ring spacing."},
    {"name": "cosine-squared",
     "code": SEED_COSINE_SQ,
     "concept": "Raised-cosine-squared amplitude taper for low side-lobes."},
    {"name": "gaussian",
     "code": SEED_GAUSSIAN,
     "concept": "Truncated Gaussian taper across rings with super-linear spacing."},
    {"name": "per-element-taylor",
     "code": SEED_PER_ELEMENT_TAYLOR,
     "concept": "Radial Taylor-style amplitude taper based on physical ring radius."},
    {"name": "bessel-taper",
     "code": SEED_BESSEL_TAPER,
     "concept": "Radial cosine-power taper approximating a Bessel aperture profile."},
]
