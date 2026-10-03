# Azure-Rooted Multi-Provider Proxy

## Status

Accepted.

## Context

The existing public product is the Cursor Azure GPT-5 Flask proxy. A separate Codex proxy proved useful for routing Cursor traffic to the Codex/ChatGPT backend, backed by a ChatGPT monthly subscription through `codex login`.

Keeping the providers as separate services duplicated the Cursor-facing contract, deployment surface, and documentation.

## Decision

In single mode and environment-configured tenant mode, this repository remains Azure-rooted for backward compatibility. Root (`/`) and explicit `/azure` paths route to Azure. Codex is available separately through `/codex` (single mode only). Provider selection is path-based rather than model-based because providers can expose overlapping model IDs:

- `/v1/models` and `/azure/v1/models` return Azure models.
- `/codex/v1/models` returns Codex/ChatGPT subscription models.

Database-managed tenants use an ordered route of Azure, OpenAI and OpenRouter profiles for root inference requests. `ProviderProfile.route_priority` is the sole routing source (`NULL` = inactive); the former `Tenant.active_profile_id` is migrated to priority 1 and removed. The root model catalog exposes exactly `Tenant.custom_model_id` when the route has a valid candidate. Every request must use that exact logical ID; each candidate explicitly maps it to its own catalog-validated `default_model`. No heuristic model-name normalization is used.

`/azure` stays pinned to the first routed Azure profile and never falls back to another provider. `/codex` remains separate and is not a database-route candidate; tenant mode continues to disallow the shared Codex login. Management, readiness and model discovery are not failover operations. `ENABLE_AZURE=false` excludes Azure candidates; explicit disabled provider paths fail locally.

Root failover tries each valid profile once in priority order on a provable connection-establishment failure, HTTP 408/429/5xx, or a recognized structured early SSE error with one of those statuses. Other 4xx and ambiguous read failures are final. Routed Azure attempts do not use Azure's internal unlimited 429 retry loop. This is deliberately best effort: an upstream may have processed a failed request, so a switch can cause additional processing or costs. It is not an exactly-once guarantee.

Adapters inspect at most 64 KiB / 32 initial SSE events before committing the client response, with a hard 5-second inspection deadline and a 30-second socket read timeout (10-second connect timeout). One pending read is handed back to the same stream when the inspection deadline expires; this does not reopen or replay the upstream request. Once output is released, there is no replay, including later SSE errors or stream interruptions. Failed connections are closed, and exhausted routes return the last error's HTTP status with sanitized error details.

Single mode continues to use `SERVICE_API_KEY`; database tenants authenticate using their own Cursor key. Route edits affect new request snapshots, never an already-running request.

## Consequences

Azure users keep the existing root URL and environment variables. Codex users get the ported request adaptation, response adaptation, auth-state handling, and token refresh behavior under the same Flask server stack.

Codex is not a fallback for Azure. It is the explicit provider path for users who want to use a ChatGPT monthly subscription from Cursor.
