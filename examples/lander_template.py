"""Lunar Lander task description and template (loaded from bundled MLES templates).

We deliberately bypass the `llm4ad` package's `__init__.py` (which eagerly
imports `codebleu` / `gymnasium` / etc.) by loading the standalone
`template.py` file via importlib.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_PATH = REPO_ROOT / "third_party" / "mles_templates" / "moon_lander" / "template.py"

_spec = importlib.util.spec_from_file_location("_mles_lander_template", _TEMPLATE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MLES_TASK_DESCRIPTION = _mod.task_description
MLES_TEMPLATE = _mod.template_program

LANDER_TASK_DESCRIPTION = MLES_TASK_DESCRIPTION + (
    "\n\nObjective scalar: Normalized Weighted Score (NWS) in [0, 1.4]:\n"
    "  NWS = 0.6 * mean_reward/200 + 0.2 * (1 - min(mean_fuel/100, 1)) + 0.2 * success_rate\n"
    "Higher is better. The published MLES paper reports NWS ~ 0.95+ on a single\n"
    "training seed and >0.85 averaged over unseen seeds."
)

LANDER_TEMPLATE = MLES_TEMPLATE
