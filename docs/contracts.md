# CLI and Artifact Contract

## Environment

| Variable | Required | Purpose |
|---|---:|---|
| `SKILLCODER_OWNER_KEY` | yes | Private owner secret, at least 32 UTF-8 bytes. |
| `SKILLCODER_MODEL_API_KEY` | yes | API key for the selected model endpoint. |
| `SKILLCODER_MODEL_BASE_URL` | optional | OpenAI-compatible HTTPS base URL; defaults to `https://api.openai.com/v1`. |
| `SKILLCODER_MODEL` | unless `--model` is set | Endpoint model identifier. |

## Source contract

`--source` accepts either one Markdown file or a Skill Package directory. Directory inputs use `SKILL.md` by default; `--entrypoint` selects another relative Markdown entrypoint.

Every package Markdown document is normalized into one SkillIR graph. Carrier spans may come from multiple documents; each span remains exact, disjoint, and contained inside one document. Non-Markdown files are copied without execution or modification. Version-control/cache directories are ignored. Symbolic links, non-Markdown single-file inputs, environment files, private keys, and common credential filenames fail closed. The authenticated manifest binds every delivered file path, size, and SHA-256 digest.

## `run`

Input:

- source Markdown or Skill Package directory;
- stable `skill_id`;
- target `buyer_id`;
- buyer population and power-of-two codeword length;
- at least ten generated normal queries and at least five active/decoy pairs.

Output:

```text
<output>/
├── normal_queries.json
├── report.json
├── release.json
└── package/
    ├── build.json
    ├── buyer_delivery/<original package tree>
    └── owner_audit/audit.json
```

The lightweight default population is 8 buyers with a 4-bit codeword. `--buyer-count 16 --codeword-length 32` selects the paper-capacity profile.

## `build`

`build` accepts an existing normal-query JSON array and creates one candidate buyer package. The query file must contain at least ten unique, non-empty strings. The output path must not exist.

`build.json` contains public build metadata and hashes. `owner_audit/audit.json` contains private mappings, model-call audit data, gates, and an HMAC authentication field.

Ordinary-behavior evaluation uses two mandatory blind judge orientations per query: clean output as answer A and candidate output as answer B, then the swapped order. Both receive the clean execution Skill as a shared inert policy reference. Scores are normalized by answer identity and averaged per dimension before the fixed gate is applied. Invalid judge JSON is retried at most twice per orientation; a valid adverse score is never resampled.

Python callers can catch `skillcoder.watermark.BehaviorGateRejected` and serialize its `report` attribute. The report contains query and output digests, identity-normalized dimension scores, orientation disagreement, bounded format-failure records, sanitized model-call audits, and the exact failed predicates. It deliberately contains no raw Skill, query, answer, or arbitrary model-provided audit fields. CLI builds remain atomic and do not publish a buyer tree or a partial output directory after this rejection; applications that need a durable rejected-build record should serialize the exception report to an owner-only location outside the requested package path.

## Buyer-family commands

`build-family` prepares the semantic parse, activation profile, controlled vocabulary binding, carrier selection, codebook, and term-neutral controlled-vocabulary fragment templates once. Buyer terms are substituted locally; carrier fusion and fidelity evaluation then run for each candidate. It renders all configured buyers, or the subset selected with repeated `--buyer-id` options.

```text
<family>/
├── family.json
├── owner_audit/family.json
└── buyers/
    ├── buyer_1/
    │   ├── build.json
    │   ├── buyer_delivery/<original package tree>
    │   └── owner_audit/audit.json
    └── ...
```

`probe-family` runs matched active/decoy probes and normal controls for each candidate. It reports Owner Verification, Buyer Attribution, Top-1 accuracy, and the exact buyers that pass the release gate. `verify-family` performs no model calls and verifies the family audit, shared plan identifier, buyer identities, and every delivery tree.

`run-family` composes query generation, family construction, batch probing, ECC decoding, aggregation, and `release.json` generation in one atomic command. Query generation and construction share one immutable source snapshot and record the same source-tree digest.

The `skillcoder-release/2` manifest contains an Owner HMAC, the exact report digest, Skill and watermark-plan identifiers, and a complete authenticated manifest for each entry in `approved_deliveries`. `verify-release --run <output>` verifies these bindings and returns no model calls. A valid rejected manifest confirms integrity; it does not approve any delivery.

## Issuance QA: `probe` and `probe-family`

`probe` requires the pristine owner-retained package, its bound normal-query file, the matching owner key, a model endpoint, and at least five active/decoy pairs. It verifies the exact delivery manifest and canonical hash before the first model call. Its output path must not exist and must be outside the input package. The report is published through an atomic file replacement and contains:

- sample counts and per-response continuous capsule validity;
- authenticated Owner policy, matched differential score, and Owner decision;
- active, decoy, normal, and paired-differential release-quality rates;
- observed bits, errors, erasures, decode margin, and decoded buyer;
- independent Buyer Attribution status and release decision;
- response and model-call audit records.

Owner Verification uses `mean(ν_active − λν_decoy)`. The scorer computes each `ν` from four private capsule phrases (mode, route, checkpoint, and decision) with sentence-window token-cosine similarity. The public slot label anchors controlled-vocabulary extraction and contributes no Owner evidence. The built-in threshold policy is recorded as `builtin_reference_v1`; `calibrate-owner` can freeze a threshold from independently authored same-domain Skills. The 60/20/10 checks are release-quality checks and do not define family membership.

The authenticated Owner audit freezes `matched_probe_plan`. Every entry records `pair_id`, one of the five required intents, a human-readable purpose, the ordinary base request, one LLM-generated template containing `[[CUE_1]]` through `[[CUE_3]]`, and its positive/negative renderings. Generation uses at most three generate → independent judge → revise/select rounds; deterministic invariants run before judgment, and the path fails closed if no candidate passes. Both renderings come from that same template, so the controlled difference is limited to the private cue values. Detection reports preserve the active/decoy compatibility labels and additionally expose positive/negative roles plus per-pair validity differentials.

`calibrate-owner --clean-scores <json> --target-fpr <rate> --calibration-source <id>` accepts a JSON array of precomputed clean differential scores. It returns a conservative threshold above the empirical boundary. Pass that threshold and provenance to build/run with `--owner-threshold`, `--owner-negative-weight`, and `--owner-calibration-source`.

The direct runtime is the default. LangChain and CAMEL are selected with `--runtime langchain` and `--runtime camel`. All runtimes receive the same serialized Skill and produce the same probe report contract. CAMEL uses one fresh, tool-free `ChatAgent` per query so probe requests do not share conversational memory.

CLI stdout contains only a log-safe decision summary and artifact paths. Raw active/decoy queries, responses, and model-call records remain in the owner-side report file.

## Post-distribution detection: `probe-suspect`

`probe-suspect` deliberately has two independent inputs:

- `--reference` is an owner-retained `run` or `run-family` root containing `release.json`. The release HMAC, exact report digest, `approved_deliveries`, protocol, original delivery trees, private audit, frozen Owner policy, codebook, and normal-query digest must verify. Only approved Buyer IDs remain eligible attribution results.
- `--suspect` is a separate untrusted Markdown document or Skill Package. It is loaded with the normal file-count, size, path, credential, and symlink protections, but its text and tree hashes are not compared with the reference. `--entrypoint` selects its package entrypoint.

This boundary permits local evaluation after paraphrase, compression, clause deletion, or section reorganization without weakening issuance integrity. The output is an atomic owner-side report containing:

- suspect canonical/tree digests as evidence provenance;
- active, decoy, normal, and paired-differential probe statistics;
- continuous Owner Verification and its authenticated policy;
- raw errors-and-erasures ECC observations;
- decoded approved buyer or attribution abstention;
- complete response and model-call records.

The suspect report has `scope: post_distribution_suspect_probe`. It does not compare the decoded identity with an expected Buyer and has no `release_ready` field. If Owner Verification rejects, raw ECC observations remain available but Buyer Attribution returns `not_evaluated_owner_not_supported` with no decoded identity. A completed negative Owner decision is a detection result, not an integrity error.

The Python function `skillcoder.pipeline.probe_released_target` accepts a retained run root and any independent object implementing `ProbeTarget.invoke(query, purpose=...)`. It verifies the release and invokes the same scoring/decoding core with attribution restricted to approved buyers. Production deployments use this boundary for a remote black-box Agent adapter; the local CLI constructs the target from `--suspect` through the selected direct, LangChain, or CAMEL runtime. Target calls use one neutral `behavior_probe` purpose, while active/decoy/normal labels and the owner-keyed mixed schedule remain owner-side.

## `verify`

`verify` performs no model calls. It checks protocol compatibility, owner-key fingerprint, audit HMAC, canonical Markdown hash, complete buyer-delivery manifest, and the public file surface.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Command completed and required checks passed. |
| `1` | Input, integrity, provider, or other execution failure. |
| `2` | Issuance QA rejected a candidate, or a suspect probe completed with a negative Owner decision. |

## Deployment mapping

The open-source `probe-suspect` command executes a separately supplied local suspect through a configured model runtime. A production black-box service implements the same `ProbeTarget` contract for the suspected remote Agent and calls `probe_released_target`, reusing release-bound differential scoring and ECC decoding without reading or hash-matching the suspect artifact.
