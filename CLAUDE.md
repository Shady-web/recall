<!--
  AWS guidance installed by the Agent Toolkit for AWS (Step 7 of setup).
  Source: https://raw.githubusercontent.com/aws/agent-toolkit-for-aws/refs/heads/main/rules/aws-agent-rules.md
  Retrieved: 2026-08-04. The rules below are verbatim; re-fetch to update.

  Project-specific instructions for Recall can be added below the AWS section.
-->

# AWS Guidance

- Prefer the AWS MCP Server for AWS interactions — it provides sandboxed
  execution, observability, and audit logging. If unavailable, use the
  AWS CLI directly.
- Before starting a task, check whether a relevant AWS skill is available.
  Load the skill with `retrieve_skill` and prefer its guidance over
  general knowledge.
- When uncertain about specific AWS details (API parameters, permissions,
  limits, error codes), verify against documentation rather than guessing.
  State uncertainty explicitly if you cannot confirm.
- When creating infrastructure, prefer infrastructure-as-code (AWS CDK or
  CloudFormation) over direct CLI commands.
- When working with infrastructure, follow AWS Well-Architected Framework
  principles.
- Do not use em dashes in AWS resource names or descriptions. Use
  hyphens instead.

## Secret Safety

- MUST load the `aws-secrets-manager` skill first for any secret,
  credential, API key, token, or password task. MUST NOT call
  `secretsmanager get-secret-value` or `batch-get-secret-value`, and MUST
  NOT hit the Secrets Manager Agent daemon directly. MUST use
  `{{resolve:secretsmanager:secret-id:SecretString:json-key}}` with
  `asm-exec` so the secret resolves at runtime without entering context.

---

# Recall — project rules

Architecture, data model, and the phase plan are in **CONTEXT.md** — read it for
what the system *is*. What follows is only the set of things a competent agent
will otherwise get wrong while acting reasonably.

## The kernel is never bypassed

- All SQL for branches, memories, decisions, and audit lives in `kernel/`.
  `api/`, `agent/`, `mcp_server/`, and `scripts/` call kernel methods only.
- Need data the kernel does not expose? **Add a kernel method** (see
  `list_branches`, `list_decisions`) rather than writing SQL at the call site.
  Writing SQL in `api/` is the obvious move and it is wrong.
- Exception to the AWS rule above: the kernel calls Bedrock through `boto3`
  directly, by design. Do **not** re-route the embedding or reasoning paths
  through the AWS MCP server — they are the kernel's own dependency, and the
  bearer-token auth path in `Settings.export_bedrock_auth()` is deliberate.

## Every operation is audited in its own transaction

- `kernel.audit.record()` runs inside the operation's transaction, so a failed
  operation leaves zero audit rows. Keep it that way.
- One documented exception: physical replay (`AS OF SYSTEM TIME` transactions
  are read-only) writes its audit row immediately afterwards.

## Never mix embedding spaces

- Titan and `FakeEmbeddingProvider` both emit 1024-dimension unit vectors and
  are **mutually meaningless**. Comparing across them yields orthogonal noise
  scored near zero — plausible-looking, entirely fake.
- **Do not switch providers to dodge a `ThrottlingException`.** That is the
  exact bug that produced a page of confident nonsense.
- **Seed and record with the same provider.** Never seed with fake and then
  record live.
- `memories.embedding_model` records the space; `verify_embedding_provider`
  refuses a mismatch at startup. Do not weaken that guard to make a run proceed.

## Everything on screen is read live

- No hardcoded values, placeholder numbers, or fallback constants in `ui/` or
  `api/`. If a number renders, it was read from CockroachDB on that request.
- On failure, **show the error** — never substitute a plausible-looking value.
  Defensive smoothing here silently destroys the project's central claim.
- Do **not** present fake-mode similarity scores as evidence of recall quality.
  The fake provider ranks a deliberately unrelated control query as highly as
  real hits (it matches on lexical overlap, including stopwords). Fake mode
  demonstrates *mechanism*, never *semantic quality*.

## Bedrock reality (verified 2026-08-04, account 211125777641)

- **Claude models are geo-blocked** at this egress location. Not fixable by
  changing region, credentials, or model id. Offline mode is the fallback.
- `BEDROCK_REASONING_MODEL` **must be an inference profile** with a routing
  prefix (`us.` / `global.`). Bare `anthropic.*` ids are rejected outright.
- **Titan on-demand RPM quota is 0 and non-adjustable.** Throttling is capacity,
  not auth — SigV4 and bearer token throttle identically. Batch inference or
  provisioned throughput are the only real paths; retry tuning is not.
- Offline mode (`RECALL_OFFLINE=1`, or `--offline`) = fake embeddings + the rule
  reasoner, for zero-cost iteration.

## Demo

- The incident query must **name the mechanism**, not ask politely. Measured:
  a keyword-dense query ("pgbouncer connection pool exhaustion, cl_waiting,
  max_client_conn") put 6/6 relevant memories in the top 6 with a positive
  separation margin; phrasing it as "what should we do to restore latency?"
  scored materially worse.
- `benchmarks/recall_quality.py` measures ranking quality (rank, margin,
  spread) and declares its relevance labels up front. Use it instead of
  eyeballing scores.
