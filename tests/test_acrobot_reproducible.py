import random

import numpy as np
import pytest

from examples.acrobot_reproducible import evaluate_acrobot_code_reproducible, policy_rng

RANDOM_POLICY = '''
import numpy as np
import random
offset = np.random.randint(3)
def choose_action(ct1, st1, ct2, st2, avt1, avt2, last_action):
    return (np.random.randint(3) + random.randint(0, 2) + offset) % 3
'''


def test_random_policy_is_reproducible_across_search_rng_states():
    np.random.seed(11)
    random.seed(11)
    first = evaluate_acrobot_code_reproducible(RANDOM_POLICY, [1, 2], max_steps=25, render_be=False)
    np.random.seed(99)
    random.seed(99)
    second = evaluate_acrobot_code_reproducible(RANDOM_POLICY, [1, 2], max_steps=25, render_be=False)
    assert first[0] == second[0]
    np.testing.assert_equal(first[1], second[1])
    assert first[2] == second[2]
    assert not first[2]


def test_episode_global_state_is_reset_and_rendering_does_not_change_score():
    code = '''
counter = 0
def choose_action(ct1, st1, ct2, st2, avt1, avt2, last_action):
    global counter
    counter += 1
    return counter % 3
'''
    doubled = evaluate_acrobot_code_reproducible(code, [2, 2], max_steps=20, render_be=False)
    single = evaluate_acrobot_code_reproducible(code, [2], max_steps=20, render_be=True)
    assert doubled[0] == single[0]
    assert doubled[1]["episode_steps"] == single[1]["episode_steps"] * 2
    assert single[3]
    assert doubled[3] is None


@pytest.mark.parametrize("code", [RANDOM_POLICY, "raise RuntimeError('candidate failure')"])
def test_evaluation_restores_search_random_state(code):
    np.random.seed(23)
    random.seed(23)
    before_np, before_py = np.random.get_state(), random.getstate()
    evaluate_acrobot_code_reproducible(code, [1], max_steps=20, render_be=False)
    after_np = np.random.get_state()
    assert before_np[0] == after_np[0]
    assert np.array_equal(before_np[1], after_np[1])
    assert before_np[2:] == after_np[2:]
    assert before_py == random.getstate()


def test_nested_rng_context_restores_outer_rng():
    with policy_rng(7):
        before = np.random.get_state()
        with policy_rng(99):
            np.random.random(10)
        assert np.array_equal(before[1], np.random.get_state()[1])


