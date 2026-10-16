from types import SimpleNamespace

import pytest

from reflex.llm_client import LLMClient


def test_credentials_are_hidden_from_repr_and_failure_records(monkeypatch, capsys):
    # Synthetic fixtures, never usable credentials.
    key = 'fixture-' + 'primary-credential'
    vision_key = 'fixture-' + 'vision-credential'
    authorization = 'Bearer fixture-' + 'header-credential'
    client = LLMClient(host='https://example.invalid/v1', key=key, model='fixture',
                       vision_key=vision_key, extra_headers={'Authorization': authorization},
                       max_retries=1, debug=True)

    def failure(**kwargs):
        raise RuntimeError(f'upstream echoed {key}; {vision_key}; {authorization}')

    backend = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=failure)))
    monkeypatch.setattr(client, '_client', lambda **kwargs: backend)
    with pytest.raises(RuntimeError) as error:
        client.chat('fixture')
    surfaces = [repr(client), str(error.value), str(client.call_trace()), capsys.readouterr().out]
    for secret in (key, vision_key, authorization):
        assert all(secret not in surface for surface in surfaces)
    assert client.stats()['failed'] == 1
    assert '[REDACTED]' in client.call_trace()[0]['error']


def test_example_reads_process_environment(monkeypatch, tmp_path):
    from examples.run_acrobot import parse_args

    monkeypatch.setenv('LLM_HOST', 'https://example.invalid/v1')
    monkeypatch.setenv('LLM_KEY', 'fixture-key')
    monkeypatch.setenv('LLM_MODEL', 'fixture-model')
    args = parse_args(['--log-dir', str(tmp_path)])
    assert (args.llm_host, args.llm_key, args.llm_model) == (
        'https://example.invalid/v1', 'fixture-key', 'fixture-model')
    assert args.diagnostic_mode == 'full'
