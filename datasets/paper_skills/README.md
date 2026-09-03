# SkillCODER Paper Skills

This directory materializes the six main-study Skill identities described in Section 4.1 and Appendix E of the companion paper.

## Dataset composition

| Domain | Paper variant | Local path | Availability |
|---|---|---|---|
| Code review | Anthropic Skill Creator | `generated/code_review/` | Prompt available; paper output unavailable |
| Code review | Trail of Bits differential-review | `real_world/code_review/differential-review/` | Commit-locked source snapshot |
| Data science | Anthropic Skill Creator | `generated/data_science/` | Prompt available; paper output unavailable |
| Data science | CSV Data Summarizer | `real_world/data_science/csv-data-summarizer/` | Commit-locked source snapshot |
| Travel planning | Anthropic Skill Creator | `generated/travel_planning/` | Prompt available; paper output unavailable |
| Travel planning | ErlebnisW Travel Planner | `real_world/travel_planning/travel-planner/` | Commit-locked source snapshot |

`manifest.json` pins the public Skill Creator commit needed to reproduce the three generated variants. The generator is not vendored because it is not one of the six evaluated Skills and is not used by the SkillCODER runtime.

## Provenance policy

- Files under `real_world/` are unmodified exports from the commits in `manifest.json`.
- Generated prompts are transcribed from Appendix E. No generated `SKILL.md` is supplied because the paper does not provide the three realized outputs or their model transcripts.
- The paper cites the Anthropic repository as accessed in 2025-06, but that repository's public history starts in 2025-10. The mismatch is recorded rather than silently resolved.
- Imported scripts, permissions, archives, and instructions under `real_world/` are inert research data. They have not been executed or installed.
- Upstream licenses remain controlling. The CSV repository declares MIT in its README but does not contain the linked `LICENSE` file at the recorded commit.

## Distribution boundary

This research tree is distributed as repository research data but excluded from Python wheels and source distributions. It is outside the root Apache-2.0 grant and remains subject to the upstream terms recorded in `manifest.json`. The CSV snapshot retains its upstream README declaration; the pinned revision does not contain the linked standalone license file.

Use `manifest.json` for machine-readable identities and `SHA256SUMS` for byte-level verification.
