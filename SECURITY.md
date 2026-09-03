# Security Policy

## Secrets

- Keep `SKILLCODER_OWNER_KEY`, model API keys, target-service tokens, and owner audits outside the buyer delivery.
- Use a secret manager in deployed systems.
- Rotate any credential pasted into chat, logs, tickets, or shell history.
- Pair `SKILLCODER_MODEL_API_KEY` with the endpoint configured by `SKILLCODER_MODEL_BASE_URL` or `--base-url`.

## Buyer delivery

Only the relevant `buyer_delivery/` tree is buyer-visible. The codebook, token pairs, query set, owner authentication, and buyer mapping remain owner-only.

## Endpoint policy

Remote model endpoints require HTTPS. URL credentials, query strings, and fragments are rejected. Loopback HTTP requires the explicit development override `SKILLCODER_ALLOW_INSECURE_LOCAL_HTTP=1`.

Treat the configured endpoint as a trusted evaluator within the deployment boundary. Carrier generation transports keyed phrases as opaque placeholders; fidelity and behavior checks evaluate the real candidate. Local family probing sends each selected buyer delivery to that endpoint. Probe separate buyer deployments when cross-copy visibility is sensitive.

`probe-suspect` requires an authenticated retained run root and limits attribution to Buyer IDs in its verified `approved_deliveries`. It treats the separately supplied suspect Skill as untrusted input, applying package path, size, credential, and symbolic-link checks without requiring its hashes to match the reference. The suspect text is sent to the configured model endpoint; the Owner key, codebook, audit, and active/decoy condition labels are not.

## Reports

Probe reports can contain model outputs and private query evidence. Treat them as owner-side security records. The CLI prints only a compact decision summary; detailed records remain at the requested output path. Redact endpoint-specific sensitive content before sharing.

Run `verify-release` before distribution. It verifies the Owner-authenticated release decision, exact report digest, and complete file manifest of every approved delivery. A bare or edited list of Buyer IDs is not a release authorization.

## Vulnerability reporting

When publishing a fork, configure a private security contact in the repository hosting platform. Include the affected version, reproduction steps, impact, and whether owner or model credentials may have been exposed.
