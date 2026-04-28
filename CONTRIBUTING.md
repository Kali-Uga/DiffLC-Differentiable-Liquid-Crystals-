# Contributing

## Working rules

- Keep changes focused and physics-preserving.
- Prefer small, reviewable pull requests.
- Add or update tests for behavior changes.
- Avoid notebook-only edits when a library module can hold the logic.

## Local validation

```bash
pip install -e '.[dev,research]'
pytest
python -m py_compile $(find src tests -name '*.py')
```

## Suggested PR content

- What changed.
- Why the change is needed.
- How you validated it.
- Whether it changes numerical behavior or only organization.
