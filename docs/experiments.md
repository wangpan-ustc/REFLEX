# Running Experiments

## Task Objectives

| Runner | Task | Objective |
|---|---|---|
| `examples/run_acrobot.py` | Acrobot-v1 | Solve-time score with a distance penalty for unsolved episodes |
| `examples/run_pendulum.py` | Pendulum-v1 | Final-state stability score over a 200-step episode |
| `examples/run_lander.py` | LunarLander-v3 | Normalized weighted score combining reward, fuel efficiency, and landing success |
| `examples/run_antenna.py` | CCAA synthesis | Composite array score (sidelobe level, directivity, and beamwidth) |
| `scripts/ccaa/run_ccaa_reflex_formal.py` | CCAA strict-36D | Strict 36-dimensional antenna objective under a fixed evaluation budget |

## Budgets and Stopping Flags

- `--budget` sets the candidate evaluation budget for the evolutionary search loop.
- `--llm-call-budget` caps total model requests across Critic, Actor, skill distillation, and repair steps.
- `--run-to-call-budget` continues search until `--llm-call-budget` is reached rather than stopping early at the target threshold.
- `--polish-mode` (`none` by default, or `cem`) controls optional Cross-Entropy Method constant polishing on elite policies.

Run any script with `--help` for its full command-line interface.

## Seeds and Reproducibility

By default, control and antenna runners sample a campaign seed and environment seeds at startup and record them as `seed` and `training_seeds` in `history.json`. Pass `--seed <seed>` and `--training-seeds <s1,s2,...>` to replay the exact same evaluation environments.

For the CCAA equal-budget benchmark (`scripts/ccaa/`), each run is indexed by `run_0`, `run_1`, etc., and seeded deterministically from its run identifier so that REFLEX and classical baselines share identical initializations.
