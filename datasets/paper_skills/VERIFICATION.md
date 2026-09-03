# Verification Record

## Result

- Import status: `partially_verified`.
- Public-source snapshots: `verified_exact`.
- Paper-generated samples: `unavailable`; their Appendix E prompts and a pinned public generator reference are present.
- Reproduction coverage: the three public Skills are complete; the three generated Skills require a labeled reproduction run.

## Checks performed

The three public Skill trees were exported again from their recorded Git commits. Recursive byte comparisons against this directory returned no differences:

| Artifact | Commit | Source tree | Result |
|---|---|---|---|
| Trail of Bits differential-review | `540111a52a2b76009fb279fed2b8f5d3eaa97adc` | `90210497beddff346abd4fd43175a2702adb37a5` | exact |
| CSV Data Summarizer | `9b3affd270f85aaf1c8a7a457f510034d5736cde` | `968cd3433a26e13544845dcbcb5bb67c89c86065` | exact |
| ErlebnisW Travel Planner | `a7a43cc811ac723566d68c3985bc19920dc95000` | `e0eb1f542f93ca4182e3196bce5082902347a898` | exact |

The public Skill Creator revision used for future reproduction is pinned in `manifest.json`; its source tree is intentionally not vendored.

Additional checks confirmed:

- every imported package has a non-empty `SKILL.md`;
- no `.git` directories, cache directories, macOS metadata files, or symbolic links are present;
- `manifest.json` parses as valid JSON;
- imported helper scripts were not executed;
- `SHA256SUMS` covers every dataset file except itself.

## Known gap

The paper does not publish the three realized Skill Creator outputs or their model transcripts. The paper also cites an Anthropic repository access date earlier than the public repository history available during this import. These gaps are recorded in `manifest.json`; no synthetic artifact is represented as a paper original.

The CSV snapshot declares MIT in its upstream README but contains no standalone license text at the pinned revision. The dataset tree therefore remains a source-workspace research asset and is excluded from Python distributions pending independent resolution of that notice.

## Reproduction boundary

Reproducing the missing generated samples requires running each `generated/*/PROMPT.md` through the pinned Skill Creator workflow and recording the model, base URL, temperature, seed when supported, timestamp, and transcript. Any such outputs belong in a separate `reproduced/` namespace and must be labeled as reproductions.
