# Contributing

1. Create an isolated Python 3.10+ environment.
2. Install `pip install -e '.[test]'`.
3. Keep test providers inside `tests/`; runtime code uses the OpenAI-compatible client or a framework adapter.
4. Preserve the buyer/owner file boundary and failure-atomic output behavior.
5. Add tests for every protocol, security, or gate change.
6. Run `pytest` and `python -m build` before submitting changes.

Changes to cue construction, extraction thresholds, ECC rules, or acceptance gates must document the threat-model impact and include active, decoy, and normal cases.
