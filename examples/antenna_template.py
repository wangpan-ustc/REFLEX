"""Task description and policy template for the CCAA antenna task.

The LLM is asked to write a function `design_array(spec)` that returns a
1-D numpy array `x` of length `spec['design_dim']` with all entries in
[0, 1]. The first `spec['num_elements']` entries are amplitude codes; the
remaining `spec['num_rings']` are radial-increment codes.

We deliberately keep the contract simple so the LLM can encode rich
heuristics (Taylor / Dolph-Chebyshev / aperture tapers, ring-thinning,
importance-weighted spacing, etc.) without leaving the function body.
"""

ANTENNA_TASK_DESCRIPTION = (
    "Design a synthesis policy for a Concentric Circular Antenna Array (CCAA).\n\n"
    "## Geometry\n"
    "- Concentric rings; each ring has a fixed number of evenly-spaced isotropic "
    "elements.\n"
    "- The design vector x accepts THREE layouts (N = num_elements, R = num_rings):\n"
    "    * base  (N+R,  spec['design_dim'])      : [amp_codes (N), radius_codes (R)]\n"
    "    * rot   (N+2R, spec['design_dim_rot'])  : base + [ring_rotation_codes (R)]\n"
    "    * full  (2N+2R, spec['design_dim_full']): rot + [element_phase_codes (N)]\n"
    "- Decoded amplitudes a_n = amp_min + (1 - amp_min) * amp_code, with amp_min = 0.05.\n"
    "- Decoded ring radii r_k = cumsum(min_spacing + extra_spacing * radius_code) "
    "in units of wavelength, with min_spacing = 0.25 lambda and extra_spacing = 0.65 lambda.\n"
    "- Ring rotation: ring k's elements are all rotated by "
    "delta_k = rot_code_k * 2*pi/count_k (staggering rings breaks the aligned "
    "azimuthal grating structure that creates coherent sidelobes).\n"
    "- Element excitation PHASE: psi_n = (phase_code_n - 0.5) * 2*pi in [-pi, pi]. "
    "phase_code = 0.5 means zero phase (broadside). Phase tapering / azimuthal "
    "phase modes psi_n = beta * cos(m * phi_n + delta) can destructively cancel "
    "specific sidelobes that amplitude tapering alone cannot reach.\n\n"
    "## Physics\n"
    "AF(theta, phi) = sum_n a_n * exp(j * (2*pi * r_n * sin(theta) * cos(phi - phi_n) + psi_n)).\n"
    "CAUTION: nonzero phases steal power from the broadside main beam; keep "
    "phase perturbations SMALL (|psi| <~ 0.5 rad) or precisely structured, and "
    "always check the directivity column of the feedback table.\n"
    "Metrics evaluated on the elevation cuts at phi = 0 and phi = pi/2:\n"
    "    SLL  : peak side-lobe level (dB; more negative is better).\n"
    "    FNBW : first-null beamwidth (deg).\n"
    "    D    : 3-D directivity (dB).\n\n"
    "## Objective\n"
    "Maximize  score = -SLL + 0.3 * D - 0.05 * |FNBW - FNBW_target|.\n"
    "(Equivalent to: deeply suppress side-lobes, raise directivity, keep the\n"
    "main beamwidth near the target.)\n\n"
    "## Engineering hints (use any combination)\n"
    "- Amplitude tapers: Taylor n-bar, Dolph-Chebyshev, raised-cosine, or\n"
    "  truncated Gaussian over ring index. They trade SLL vs FNBW.\n"
    "- Ring spacings often grow super-linearly to suppress grating side-lobes.\n"
    "- The two outermost rings dominate side-lobe location; the inner rings\n"
    "  dominate main-beam width and directivity.\n"
    "- Per-ring uniform amplitudes are a strong baseline; per-element or\n"
    "  non-monotonic radial profiles can further improve side-lobe suppression.\n"
)


ANTENNA_STAGNATION_HINT = (
    "If per-ring uniform tapers plateau, explore structural variations:\n"
    "  (a) Non-monotonic radial amplitude profiles across rings.\n"
    "  (b) Per-element amplitude or azimuthal modulation in outer rings.\n"
    "  (c) Aperiodic radial spacing increments.\n"
    "  (d) Ring rotations or small excitation phase perturbations to cancel\n"
    "      dominant sidelobes while preserving broadside directivity."
)


ANTENNA_TEMPLATE = '''
import numpy as np


def design_array(spec):
    """Return a normalized design vector for a concentric circular array.

    Args:
        spec (dict): contains
            - 'num_rings'    (int)
            - 'ring_counts'  (tuple[int, ...]): elements per ring, outer last
            - 'num_elements' (int): sum(ring_counts)
            - 'design_dim'   (int): num_elements + num_rings
            - 'amp_min'      (float)
            - 'min_spacing'  (float, in wavelengths)
            - 'extra_spacing'(float, in wavelengths)
            - 'fnbw_target_deg' (float)

    Returns:
        np.ndarray with entries in [0, 1] whose length is one of:
          spec['design_dim']       = N+R   [amps, radii]
          spec['design_dim_rot']   = N+2R  [amps, radii, ring_rotations]
          spec['design_dim_full']  = 2N+2R [amps, radii, ring_rotations,
                                            element_phases]  (0.5 = zero phase)
        or {"x": np.ndarray, "refine": {"iters": int, "scale": float}} to
        request a bounded runner-side numerical polish.
    """
    num_elements = spec['num_elements']
    num_rings    = spec['num_rings']
    ring_counts  = spec['ring_counts']

    # Default baseline: per-ring smooth Taylor-like taper, near-uniform spacing.
    amp_codes = []
    for k, count in enumerate(ring_counts):
        ring_pos = k / max(num_rings - 1, 1)
        # Smooth taper: outer rings get smaller amplitude.
        a_k = 0.95 - 0.55 * ring_pos
        amp_codes.extend([a_k] * count)
    amp_codes = np.array(amp_codes, dtype=np.float64)

    # Uniform-ish ring increments (pull slightly outward for outer rings).
    radius_codes = np.linspace(0.45, 0.55, num_rings)

    x = np.concatenate([amp_codes, radius_codes])
    return np.clip(x, 0.0, 1.0)
'''
