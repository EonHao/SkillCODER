# Architecture

SkillCODER separates buyer-visible content, owner-only state, and model execution. The separation is enforced by the package layout and authenticated audit record.

## Components

| Module | Responsibility |
|---|---|
| `package.py` | Load a complete Skill Package safely, normalize Markdown documents, preserve inert assets, and hash the delivery tree. |
| `semantic.py` | Parse exact package spans into typed SkillIR nodes and validated semantic edges. |
| `watermark.py` | Prepare one shared semantic/cryptographic plan, render buyer-specific controlled terms, run FidelityOpt, and evaluate ordinary behavior. |
| `crypto.py` | Validate the owner key, randomize codewords and mappings, authenticate audits, and bind query sets. |
| `targets.py` | Execute a local Skill through the direct OpenAI-compatible client, LangChain, or CAMEL, and define the independent `ProbeTarget` boundary used by remote adapters. |
| `detection.py` | Score owner capsules from sentence windows, compute matched differential evidence, and decode buyer bits separately. |
| `pipeline.py` | Build packages atomically, keep issuance integrity separate from suspect detection, aggregate Top-1 results, and verify whole-tree integrity. |
| `cli.py` | Expose build, issuance QA, post-distribution suspect detection, and verification commands. |

## Build path

1. Validate the owner key and normal-query set.
2. Load the package without executing its assets and parse all Markdown documents into exact, disjoint SkillIR nodes and semantic edges.
3. Ask the configured model for domain-native cue pairs, capsule bundles, and controlled adjective pairs.
4. Apply deterministic validation and discard semantically unsafe vocabulary pairs.
5. Freeze one owner-keyed watermark plan containing the graph, codebook, vocabulary direction, cue direction, and edge-constrained carrier selection.
6. Generate term-neutral controlled-vocabulary fragment templates once and substitute each buyer's keyed terms locally.
7. For each buyer candidate, run two or three carrier-fusion and fidelity rounds against the real candidate, then select the best accepted version.
8. Compare original and watermarked behavior on the frozen normal-query set using the exact multi-document serialization used by probing. Each query is judged in both answer orders against the same clean Skill policy reference; identity-normalized scores are averaged before applying the fixed gate.
9. Write each watermarked Markdown document back to its original path, preserve non-Markdown assets byte-for-byte, and authenticate the complete delivery tree.

Query generation and construction consume the same in-memory package snapshot. A source-tree digest is bound to the generated query audit, so mid-run filesystem changes cannot silently change the build input.

## Issuance QA path

1. Verify the owner key fingerprint, audit authentication, complete delivery-tree manifest, canonical Markdown hash, and query-set digest.
2. Generate one natural task template per matched pair across policy checking, response generation, next-step reasoning, escalation, and clarification. A separate bounded judgment call checks task relevance, intent alignment, naturalness, and semantic cue placement before selection. Render the accepted template's positive and negative members by substituting the private active or decoy cue conjunction into the same three placeholders.
3. Execute those pairs and the bound normal requests with an Owner-keyed mixed schedule.
4. Compute continuous validity from the four private mode, route, checkpoint, and decision phrases in each response, then evaluate `mean(ν_positive − λν_negative)` against the frozen Owner threshold. The public slot label is used only to locate controlled-vocabulary positions.
5. If ownership is supported, extract CV positions, aggregate bit votes, and apply errors-and-erasures ECC decoding. Owner support remains valid when buyer decoding abstains.
6. Mark a buyer release-ready only when Owner Verification, expected-buyer attribution, and the 60/20/10 release-quality checks pass.

`probe` and `probe-family` use this strict path. A candidate modified after construction fails before any model call, which prevents an unverified tree from being approved for release.

## Post-distribution detection path

1. Authenticate an owner-retained run root: verify `release.json`, its report digest and approved delivery trees, then verify the package/family audit, frozen Owner policy, codebook, and query digest.
2. Load a separately supplied suspect Skill as untrusted execution input. Apply filesystem safety checks, but do not compare its content hashes with the reference.
3. Construct a local direct, LangChain, or CAMEL target, or accept an independent remote `ProbeTarget` adapter.
4. Mix active, decoy, and normal requests with an Owner-keyed schedule. The target receives only a neutral invocation purpose; condition labels remain Owner-side.
5. Apply the authenticated continuous Owner test. Only after it passes, run ECC Buyer Attribution over buyers approved by the authenticated release.
6. Report Owner support and the decoded released Buyer or an attribution abstention. Do not apply an expected-buyer check or create a new release decision.

`probe-suspect` implements the local path. `probe_released_target` is the shared release-bound entry point for both local and black-box targets. This separation reflects the trust model: defender evidence and the original distribution decision must be exact, while a suspected leaked copy is expected to be modified.

## Trust boundaries

| Data | Buyer | Owner backend | Model endpoint |
|---|---:|---:|---:|
| Buyer `buyer_delivery/` tree | yes | yes | Markdown documents during configured execution |
| Owner key | no | yes | no |
| Model API key | no | secret manager | transport authentication only |
| Full codebook and token pairs | no | yes | no |
| Frozen normal queries | no | yes | during configured execution |
| Active/decoy query plan | no | yes | during probing |
| Owner audit | no | yes | no |
| Suspect Skill or remote responses | adversary-controlled | detection input only | during suspect probing |

All candidate copies in one family carry the same `watermark_plan_sha256`. Their buyer record, controlled lexical realization, delivery hash, and delivery tree hash differ. The private family audit authenticates the shared codebook and every candidate package. A copy becomes releasable only after its probe result appears in `release.json`.

The configured endpoint is trusted to process the prompts and responses it receives. Carrier generation uses opaque placeholders and never receives the owner key, Buyer-ID/codeword records, or an explicit keyed phrase-binding table. Fidelity and behavior evaluation receive the real candidate so their gates measure the delivered wording. Local `probe-family` executes every selected candidate; `probe-suspect` executes only the suspect input. Use separate buyer deployments when cross-copy visibility matters.

The public algorithm is assumed known. Security relies on private keyed mappings, matched controls, suppression, and multi-position decoding.

## Failure semantics

- Invalid model output fails closed after bounded retries.
- A build publishes an owner-side candidate with `security_status: pending_probe` after all build gates pass.
- `run` and `run-family` write a `skillcoder-release/2` manifest authenticated by the Owner key. `verify-release` checks its report digest and every approved delivery tree before distribution.
- Owner rejection, buyer abstention, buyer mismatch, and release-quality rejection remain distinct completed results.
- Authentication, integrity, endpoint, or parsing failures are execution errors.
- `detection_result.supported` reports Owner membership and `buyer_attribution` reports attribution. `release_ready` exists only on issuance QA and is the distribution decision; suspect reports intentionally omit it.
