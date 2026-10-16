from reflex.utils import extract_python_block


def test_extract_python_block_with_closed_fence():
    assert extract_python_block("```python\ndef choose_action():\n    return 1\n```") == (
        "def choose_action():\n    return 1"
    )


def test_extract_python_block_with_missing_closing_fence():
    assert extract_python_block("idea\n```python\ndef choose_action():\n    return 1\n") == (
        "def choose_action():\n    return 1"
    )

