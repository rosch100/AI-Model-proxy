# ADR 0003: Database-backed tenant admin UI

## Status

Accepted

## Context

Environment `TENANTS` JSON cannot store OpenAI/OpenRouter profiles, admin logins, or cost history. Operators need a tenant-scoped HTML UI on the same Flask process as the Cursor proxy.

## Decision

- Persist tenants, admin accounts, sessions, provider profiles, and costs in PostgreSQL.
- Serve `/admin` from the existing Flask app with opaque session cookies (`Path=/admin`) and CSRF confined to that blueprint.
- Keep Cursor authentication on Bearer API keys. HTML `GET /` redirects to `/admin`; JSON proxy traffic is unchanged.
- Activate exactly one provider profile per tenant. `/models` exposes the stable custom model id.

## Consequences

- `TENANT_CONFIG_SOURCE=database` requires `DATABASE_URL`, `PROVIDER_ENCRYPTION_KEY`, and `ADMIN_SESSION_SECRET`.
- Codex remains unavailable in tenant mode.
- Billing scopes stay exclusive and operator-bound; the UI cannot retarget another tenant’s Azure resource group.
