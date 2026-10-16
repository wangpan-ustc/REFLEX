"""Acrobot task description and template (loaded from bundled MLES templates).

Bypasses `llm4ad` package init (which pulls codebleu/gymnasium-old/etc.).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_PATH = REPO_ROOT / "third_party" / "mles_templates" / "acrobot" / "template.py"

_spec = importlib.util.spec_from_file_location("_mles_acrobot_template", _TEMPLATE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ACROBOT_TASK_DESCRIPTION = _mod.task_description + (
    "\n\nObjective scalar (REFLEX): higher is better.\n"
    "  score = -(steps_to_goal / max_steps)  if the upper link reaches target height\n"
    "        =  -1 - mean(distance_to_goal)  otherwise\n"
    "Optimal score is close to 0 (e.g. solve in 80 steps out of 500 → score ≈ -0.16)."
)
ACROBOT_TEMPLATE = _mod.template_program
