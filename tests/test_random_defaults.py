import json

from reflex.randomness import environment_seeds, prior_run_ids, resolve_seed, seed_argument


def test_random_defaults_and_explicit_override(monkeypatch, tmp_path):
    from examples import run_acrobot, run_lander, run_pendulum

    monkeypatch.setenv('LLM_HOST', 'https://example.invalid/v1')
    monkeypatch.setenv('LLM_KEY', 'fixture-key')
    monkeypatch.setenv('LLM_MODEL', 'fixture-model')
    for runner in (run_acrobot, run_lander, run_pendulum):
        args = runner.parse_args(['--log-dir', str(tmp_path)])
        assert args.seed is None
        assert args.training_seeds == 'random'
        assert args.polish_mode == 'none'
        assert runner.parse_args(['--log-dir', str(tmp_path), '--seed', 'random']).seed is None
        assert runner.parse_args(['--log-dir', str(tmp_path), '--seed', '123']).seed == 123


def test_seeds_use_entropy_and_explicit_values_are_preserved(monkeypatch):
    import reflex.randomness as randomness
    values = iter([11, 22, 33, 44, 55, 66, 77])
    monkeypatch.setattr(randomness.secrets, 'randbits', lambda bits: next(values))
    assert environment_seeds('random') == (11, 22, 33)
    assert environment_seeds() == (44, 55, 66)
    assert environment_seeds('1,2,3') == (1, 2, 3)
    assert seed_argument('random') is None
    assert resolve_seed() == 77
    assert resolve_seed(5) == 5


def test_shared_priors_reuse_run_ids(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'run_ids': [0, 1, 2]}))
    assert prior_run_ids(tmp_path) == [0, 1, 2]


def test_classical_run_is_seeded_with_its_run_identifier(tmp_path, monkeypatch):
    from scripts.ccaa import run_ccaa_classical_baselines as baseline

    monkeypatch.setattr(baseline, 'ROOT', tmp_path)
    original = baseline.np.random.default_rng
    arguments = []

    def generator(*args, **kwargs):
        arguments.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(baseline.np.random, 'default_rng', generator)
    baseline.run_one('random_search', 'random_vector', 3, tmp_path / 'runs/check', 2, None)
    assert arguments == [((3,), {})]
    record = json.loads((tmp_path / 'runs/check/budget_2/run_3/random_search/random_vector/history.json').read_text())
    assert record['run_id'] == 3 and record['rng_seed'] == 3
    assert record['objective_evaluations'] == 2


def test_reflex_and_baselines_share_the_first_random_vector():
    import numpy as np

    reflex_vector = np.random.default_rng(4).random(36)
    ga_population = np.random.default_rng(4).random((16, 36))
    cma_mean = np.random.default_rng(4).random(36)
    np.testing.assert_array_equal(reflex_vector, ga_population[0])
    np.testing.assert_array_equal(reflex_vector, cma_mean)


def test_classical_antenna_seeds_are_deterministic():
    import numpy as np
    from examples.antenna_seeds import CLASSIC_SEEDS
    from examples.run_antenna import REGIMES, cfg_to_spec, _safe_exec_design

    for seed in CLASSIC_SEEDS:
        for config in REGIMES.values():
            spec = cfg_to_spec(config)
            first, _ = _safe_exec_design(seed["code"], spec)
            second, _ = _safe_exec_design(seed["code"], spec)
            np.testing.assert_array_equal(first, second)
