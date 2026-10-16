import re
from types import SimpleNamespace

from reflex.llm_client import LLMClient
from reflex.population import Individual


def test_acrobot_runner_logs_seeds_and_skips_polish_by_default(tmp_path, monkeypatch):
    from examples import run_acrobot

    class Completions:
        def __init__(self): self.index = 0
        def create(self, **kwargs):
            self.index += 1
            content = f'def choose_action(*args):\n    return 0.5\n#MOCKCODE_{self.index}\n'
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
                                   choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class Ops:
        def __init__(self, description, template, mdc, skills, **kwargs): self.mdc = mdc
        def child(self, kind, parent=None):
            code = self.mdc.llm.chat('proposal', usage_role='actor')
            return Individual(code=code, operator=kind, parent_id=parent.iid if parent else None)
        def init_zero_shot(self): return self.child('fresh')
        def mutate_diagnostic(self, parent): return self.child('diagnostic', parent)
        def mutate_text_only(self, parent): return self.child('text', parent)

    origins = []

    def evaluate(code, seeds, **kwargs):
        origins.append(kwargs.get('render_be'))
        found = re.search(r'MOCKCODE_(\d+)', code)
        return -2.0 + (int(found[1]) * .01 if found else 0), {'solve_rate': 0.0}, '', 'aW1hZ2U='

    monkeypatch.setattr(run_acrobot, 'Operators', Ops)
    monkeypatch.setattr(run_acrobot, 'evaluate_acrobot_code_reproducible', evaluate)
    args = run_acrobot.parse_args([
        '--llm-host', 'https://example.invalid/v1', '--llm-key', 'test-only', '--llm-model', 'fake',
        '--log-dir', str(tmp_path), '--skill-bank', str(tmp_path / 'bank.json'),
        '--reproducible-evaluation', '--llm-call-budget', '2', '--budget', '4', '--run-to-call-budget',
        '--freeze-skill-bank', '--retrieval-mode', 'none', '--seed', '42', '--training-seeds', '7,8,9'])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client = LLMClient(host=args.llm_host, key=args.llm_key, model='fake', call_budget=2)
    monkeypatch.setattr(client, '_client', lambda vision=False, fake=fake: fake)
    result = run_acrobot.run_campaign(args, llm_client=client)

    assert result['seed'] == 42
    assert result['training_seeds'] == [7, 8, 9]
    assert result['polish_mode'] == 'none'
    assert result['inner_cem_evaluator_invocations'] == 0
    assert False not in origins
    assert result['llm_usage']['attempted'] == 2
