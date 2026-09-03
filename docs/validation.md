# Verification

The repository verifies the implementation at three layers.

## Core contracts

`pytest` covers semantic anchoring, controlled-vocabulary generation, private codebooks, carrier distribution, bounded fidelity optimization, behavior gates, matched probing, ECC decoding, release authentication, suspect probing, package safety, and atomic output behavior.

## Framework adapters

The LangChain and CAMEL extras run against the same `ProbeTarget` contract as the direct runtime. CI installs each adapter independently and executes the complete test suite, including fresh-agent boundaries and normalized response extraction.

## Distribution

CI builds both wheel and source distributions and tests Python 3.10, 3.11, 3.12, and 3.13. Release contents are controlled by `MANIFEST.in`; research datasets stay in the source repository and runtime secrets stay outside every distribution.

Run the full local verification set with:

```bash
python -m pytest
python -m mypy --no-incremental skillcoder
python -m build
```
