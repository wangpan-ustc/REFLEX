"""Pendulum-v1 task description and template (loaded from bundled MLES templates)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_PATH = REPO_ROOT / "third_party" / "mles_templates" / "pendulum" / "template.py"

_spec = importlib.util.spec_from_file_location("_mles_pendulum_template", _TEMPLATE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PENDULUM_TASK_DESCRIPTION = _mod.task_description + (
    "\n\nObjective scalar (REFLEX): higher is better, range roughly [-3, 0).\n"
    "  score = -(1 + final_angle_error + final_stability_error)  if not perfectly upright\n"
    "        = -(steps_to_upright / max_steps)                   if perfectly stable upright\n"
    "Optimal score is close to 0 (e.g. swing up & stabilize within 100 / 500 steps → -0.20).\n"
    "Pendulum-v1 has continuous torque action in [-2.0, 2.0]."
)
PENDULUM_TEMPLATE = _mod.template_program
