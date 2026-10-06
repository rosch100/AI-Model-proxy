# Cursor Azure and Codex Model Proxy

A Flask proxy that lets [Cursor](https://cursor.com) use **Azure OpenAI** or a **Codex / ChatGPT subscription** as OpenAI-compatible providers. It translates Cursor requests to the selected upstream Responses API, streams results back in the format Cursor expects, and needs no Cursor forks or client patches.

> **You still need a paid Cursor plan.** This project only redirects where model traffic goes.

> [!IMPORTANT]
> **Using GPT-5.5 in Cursor?** Cursor may fail to route direct `gpt-5.5` traffic through custom base URLs and return `User API Key Rate limit exceeded`. See [GPT-5.5 Cursor routing](#gpt-55-cursor-routing). Forum thread: <https://forum.cursor.com/t/not-able-to-use-azure-api-key/149185/34>

This fork is maintained at [rosch100/Cursor-Azure-GPT-5](https://github.com/rosch100/Cursor-Azure-GPT-5). It is based on the community Cursor Azure/Codex proxy line ([jbwmc/Cursor-Azure-GPT-5](https://github.com/jbwmc/Cursor-Azure-GPT-5) as the configured `upstream` remote; original project history at [gabrii/Cursor-Azure-GPT-5](https://github.com/gabrii/Cursor-Azure-GPT-5)). See [Changes in this fork](#changes-in-this-fork) for what differs from that base.

---

## Contents

- [What it does](#what-it-does)
- [Changes in this fork](#changes-in-this-fork)
- [Authentication modes](#authentication-modes)
- [Provider paths](#provider-paths)
- [Features](#features)
- [Supported Azure models](#supported-azure-models)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running](#running)
- [Endpoints](#endpoints)
- [Smoke tests](#smoke-tests)
- [Architecture](#architecture)
- [Development](#development)
- [Traffic recording](#traffic-recording)
- [Troubleshooting](#troubleshooting)

---

## What it does

Cursor speaks OpenAI-shaped APIs. Azure and Codex use Responses-style APIs with different auth, headers, and SSE event shapes. This proxy sits in the middle:

| Path | Upstream | Typical use |
| --- | --- | --- |
| `/` or `/azure` | Azure OpenAI | Azure-hosted deployments, per-deployment routing, Azure prompt caching |
| `/codex` | Codex / ChatGPT | Drive Cursor from an existing ChatGPT monthly subscription (`codex login`) |

Azure owns the root URL for backward compatibility. Codex is **never** a silent fallback: it is selected only via `/codex`. Disabling a provider returns a local configuration error instead of routing to the other provider.

---

## Changes in this fork

Relative to the fork base (`upstream/main` / this repo’s `main` merge base), this fork adds multi-tenant Azure hosting, newer model coverage, reliability fixes, and CI/security hardening.

### Multi-tenant Azure auth

| Base behavior | This fork |
| --- | --- |
| One shared `SERVICE_API_KEY` for all Cursor clients | `AUTH_MODE=single` (compatible default) **or** `AUTH_MODE=tenant` |
| One global Azure URL, API key, and deployment map | Per-tenant Azure URL, API key (via env), and required `azure_model_deployments` |
| Codex available whenever enabled | Codex remains single-login only; `AUTH_MODE=tenant` refuses `ENABLE_CODEX=true` |
| Auth failures as Cursor configuration errors (HTTP 400) | Generic HTTP **401** for missing/invalid Bearer tokens |
| Conversation cache keys shared by conversation id only | Tenant-prefixed cache/session keys (`{tenant_id}:{conversation_id}`) |

See [Authentication modes](#authentication-modes) for configuration details.

### Azure models and deployments

- Cursor-facing support for GPT-5.6 (`luna` / `sol` / `terra`) and GPT-6 (`astra` / `luna` / `sol`) ids.
- Explicit deployment scoping: a model may be on the proxy allowlist yet rejected if it is not mapped for the active Azure resource or tenant.
- Example Instanz2 deployment names documented in `.env.example` and [Supported Azure models](#supported-azure-models).

### Reliability and streaming

- Retries Azure `rate_limit_exceeded` HTTP 429s and SSE failures before visible output; after output starts, it keeps streaming incrementally and does not restart a failed attempt.
- Honors Azure `retry-after-ms` / `Retry-After` with waits capped at 60 seconds per retry. Retry attempts are unbounded; the proxy does not monitor downstream disconnects during synchronous pre-output upstream attempts or retry waits, so it may continue retrying after the client disconnects, potentially indefinitely if Azure keeps rate-limiting the request.
- Avoids empty reasoning blocks that confused Cursor’s thinking UI.

### Security and defaults

- Startup refuses missing, empty, or placeholder `SERVICE_API_KEY` values in single mode.
- Sensitive request/completion logging defaults to off (`LOG_CONTEXT`, `LOG_COMPLETION`); Compose `flask` defaults `LOG_LEVEL` to `warning`.
- Upstream SSE fixtures are recorded once at stream end (same as downstream), keeping only complete SSE events and dropping an incomplete trailing UTF-8 sequence before anonymize.
- Constant-time Bearer comparison using UTF-8 bytes (avoids `hmac.compare_digest` TypeError on non-ASCII tokens).
- Dependency updates for known vulnerable packages.

### CI and supply chain

- OSV scanning for Python requirements.
- Additional GitHub Actions for security review tooling, actionlint, zizmor, and dependency review.
- Dependabot cooldown configuration and CodeRabbit workflow support.

### Documentation

- README restructured around auth modes, provider paths, and complete configuration.
- Operator docs for tenant setup, smoke tests, and fork-specific troubleshooting.

Upstream-compatible single-tenant Azure + Codex setups continue to work with `AUTH_MODE=single` (the default).

---

## Authentication modes

The proxy authenticates every protected request with Cursor’s **OpenAI API Key** (Bearer token). Choose how that key is interpreted with `AUTH_MODE`.

| Mode | Cursor API key | Azure credentials | Codex |
| --- | --- | --- | --- |
| `single` (default) | Shared `SERVICE_API_KEY` | Global `AZURE_BASE_URL` / `AZURE_API_KEY` / `AZURE_MODEL_DEPLOYMENTS` | Allowed when `ENABLE_CODEX=true` |
| `tenant` | One high-entropy key per tenant | Per-tenant URL, API key, and model→deployment map | **Not available** (shared `auth.json` is not tenant-safe) |

### Single mode (`AUTH_MODE=single`)

- Set `SERVICE_API_KEY` to a unique secret (required at startup).
- Put the same value in Cursor → Settings → Models → OpenAI API Key.
- Optional Codex and Azure can run on the same host; both clients share that key.

### Tenant mode (`AUTH_MODE=tenant`)

Use this when multiple teams or Azure resources share one proxy host.

- Configure a non-empty `TENANTS` JSON list.
- Store **only SHA-256 digests** of tenant API keys in config; never commit cleartext keys.
- Inject each tenant’s Azure API key through the environment variable named by `azure_api_key_env`.
- Every tenant must declare its own non-empty `azure_model_deployments` map. Global `AZURE_MODEL_DEPLOYMENTS` is ignored.
- `/v1/models` and request routing use only the authenticated tenant’s map.
- Cache / session keys are prefixed with the tenant id so tenants never share Azure cache partitions.
- `SERVICE_API_KEY` is **not** accepted as a fallback.
- Unknown or invalid keys return the same generic **HTTP 401**.
- `ENABLE_CODEX` must be `false`; the proxy refuses to start otherwise.

### Tenant database mode (`TENANT_CONFIG_SOURCE=database`)

Instead of `TENANTS` in the environment, persist tenants in PostgreSQL and manage
them through `/admin`. Required settings include `DATABASE_URL`,
`PROVIDER_ENCRYPTION_KEY`, `ADMIN_SESSION_SECRET`, and `WEBAUTHN_*` (RP ID, name,
origins). After the first password bootstrap, admins must enroll a passkey;
password login is then rejected. See [DEPLOYMENT.md](DEPLOYMENT.md#tenant-admin-ui-database-mode)
for enrollment and operator recovery if the last passkey is lost.

In **Verbindung**, add provider accounts, refresh their catalogs and select a
catalog-backed standard model per account. Activate multiple accounts and use
**Nach oben / Nach unten** to set the root inference failover order. Cursor keeps
one stable tenant model ID; it uses only activated accounts in the configured
failover order. The model list also exposes native IDs from every configured,
ready account; selecting one routes directly to an account advertising that
model, even if that account is not in the default failover order. To select a
specific account with a unique name, use `Account Name/model-id`, for example
`Research Team/deepseek-flash`. If account names are duplicated, use
`provider:Account Name/model-id`, such as
`deepseek:Research Team/deepseek-flash`; spaces remain readable, and reserved
characters in account names are percent-encoded. Provider-qualified IDs use the
provider key (`deepseek`, `openrouter`, `openai`, or `azure`), then the exact
account display name and native model ID. An explicit account/model ID is pinned
to that account and never falls back to a different account. Unknown model IDs
are rejected rather than forwarded unchanged.

Root requests using the tenant model ID try the next account on HTTP 402/408/429/5xx,
safe connection failures, or recognized early SSE errors, but never after stream
output begins. Other 4xx and ambiguous read failures are final. After the full route
is tried, transient HTTP 402/429 responses and explicitly transient provider limits
with a `Retry-After` of at most 30 seconds (or no header) get one cooperative retry
per profile, using a 1-second default delay. Hard quota exhaustion is not retried
on the same profile, though the next account in the route is still tried. Switching
is best effort and may incur additional processing/costs. `/azure` stays scoped to
active Azure accounts; `/codex` and single/environment-tenant mode keep their
existing behavior.
Run `flask db upgrade` before deploying the new route schema. See the
[routing decision](docs/adr/0001-azure-rooted-multi-provider-proxy.md).

Generate a tenant cleartext key and hash:

```bash
python3 -c 'import hashlib,secrets; k=secrets.token_urlsafe(32); print(k); print(hashlib.sha256(k.encode()).hexdigest())'
```

Example:

```env
AUTH_MODE=tenant
ENABLE_AZURE=true
ENABLE_CODEX=false
TENANTS=[{"id":"acme","api_key_hash":"<sha256-hex>","azure_base_url":"https://acme.openai.azure.com","azure_api_key_env":"TENANT_ACME_AZURE_API_KEY","azure_model_deployments":{"gpt-5.6-sol":"acme-sol","gpt-5.5":"acme-gpt55"}},{"id":"beta","api_key_hash":"<sha256-hex>","azure_base_url":"https://beta.openai.azure.com","azure_api_key_env":"TENANT_BETA_AZURE_API_KEY","azure_model_deployments":{"gpt-5.6-luna":"beta-luna"}}]
TENANT_ACME_AZURE_API_KEY=your-acme-azure-api-key
TENANT_BETA_AZURE_API_KEY=your-beta-azure-api-key
```

In Cursor, set OpenAI API Key to the **cleartext** tenant key (not the hash).

---

## Provider paths

| Cursor Override Base URL | Provider | Model list |
| --- | --- | --- |
| `https://your-public-proxy-url` | Azure | Azure models for the authenticated principal |
| `https://your-public-proxy-url/azure` | Azure | Same as root |
| `https://your-public-proxy-url/codex` | Codex | `CODEX_SUPPORTED_MODELS` (`AUTH_MODE=single` only) |

In `AUTH_MODE=single`, switch providers by changing only the path (`/azure` ↔ `/codex`) while keeping the same `SERVICE_API_KEY`.

### GPT-5.5 Cursor routing

Until Cursor routes native `gpt-5.5` through custom base URLs correctly:

**Codex workaround** — rewrite upstream model only:

```env
CODEX_MODEL_REWRITES=gpt-5.4:gpt-5.5
```

Select `gpt-5.4` in Cursor; the proxy rewrites the upstream Codex model field to `gpt-5.5`.

**Azure workaround** — map a Cursor-facing id to your GPT-5.5 deployment:

```env
AZURE_MODEL_DEPLOYMENTS={"gpt-5.4":"your-gpt-5.5-deployment-name"}
```

These are stopgaps. Cursor still builds prompts for the source model id, so behavior may differ from native `gpt-5.5` routing. Remove them when Cursor can send `gpt-5.5` directly.

---

## Features

### Native model IDs and reasoning

The proxy exposes Cursor-native model ids (for example `gpt-5.5`, `gpt-5.4`, `gpt-6-luna`). Cursor therefore sends model-specific system prompts instead of generic fallbacks. Incoming `reasoning.effort` is preserved and forwarded when the upstream model supports it.

Legacy aliases (`gpt-high`, `gpt-medium`, …) are intentionally unsupported.

### Azure prompt caching

For Azure, the proxy uses `metadata.cursorConversationId` as the cache routing key and sets:

- `prompt_cache_key`, `session_id`, and `x-client-request-id` to that conversation id
  (in tenant mode: `{tenant_id}:{conversation_id}`)
- `store: true` for Azure server-side storage
- `parallel_tool_calls: true`

Do **not** use Cursor’s `user` field for cache routing: it is a per-user hash shared across conversations and would mix unrelated cache partitions.

### Streaming and tool bridging

Provider Responses SSE is converted to OpenAI Chat Completions chunks when Cursor used Chat Completions. Azure event coverage includes reasoning, text, tool-call deltas, native Azure tool types (wrapped as standard function calls), errors, refusals, and more. Unknown events are logged rather than silently dropped.

Dual input formats are accepted:

- Chat Completions (`messages`) → converted to Responses
- Responses API (`input` + `instructions`) → passed through

### Reasoning display mode

Set `REASONING_DISPLAY_MODE`:

| Value | Behavior |
| --- | --- |
| `none` | Keep native reasoning metadata only; Cursor may show no thinking UI on BYOK |
| `mdthinkblocks` | Mirror reasoning into Markdown `<details>` blocks (default; experimental) |
| `thinkblocks` | Mirror into legacy `<think>...</think>` chat content |

### Usage logging

When usage is available, the proxy logs lines such as:

```text
USAGE: input=45230 (cached=38400, 85%) output=1205 (reasoning=890) total=46435
```

Analyze Docker or process logs with:

```bash
python scripts/analyze_token_usage.py --hours 48
```

---

## Supported Azure models

Root and `/azure` expose the Azure model list for the authenticated principal (global map in single mode; tenant map in tenant mode). `/codex` exposes `CODEX_SUPPORTED_MODELS`.

| Model | Status |
| --- | --- |
| `gpt-6-astra` | Azure Responses enabled; proxy E2E not yet verified |
| `gpt-6-luna` | Azure Responses enabled; proxy E2E not yet verified |
| `gpt-6-sol` | Available only on tenant `altanis-ai`; proxy E2E verified |
| `gpt-6.1-sol` | Available on both tenants; proxy E2E verified |
| `gpt-5.6-luna` | Not deployed on the configured tenant accounts |
| `gpt-5.6-sol` | Not deployed on the configured tenant accounts |
| `gpt-5.6-terra` | Not deployed on the configured tenant accounts |
| `gpt-5.5` | Verified |
| `gpt-5.4` | Verified |
| `gpt-5.4-mini` | Verified |
| `gpt-5.4-nano` | Verified |
| `gpt-5.3-codex` | Verified |
| `gpt-5.2` | Expected (same Responses API) |
| `gpt-5.2-codex` | Verified |
| `gpt-5.1` | Expected (same Responses API) |
| `gpt-5.1-codex` | Expected (same Responses API) |
| `gpt-5.1-codex-max` | Expected (same Responses API) |
| `gpt-5.1-codex-mini` | Verified |
| `gpt-5` | Expected (same Responses API) |
| `gpt-5-mini` | Verified |
| `gpt-5-codex` | Expected (same Responses API) |

Default account mapping in `.env.example` (`AzureOpenAI-Instanz2`, verified 2026-10-01):

| Cursor model ID | Azure deployment name |
| --- | --- |
| `gpt-6-astra` | `gpt-6-astra-api` |
| `gpt-6-luna` | `gpt-6-luna-api` |
| `gpt-6.1-sol` | `gpt-6.1-sol-api` |

The `altanis-ai` tenant additionally exposes `gpt-6-sol` and maps the other
three models to deployments with matching names. Both tenants were verified
with successful Azure Responses requests on 2026-10-01.

If Cursor requests a model that is not in the active deployment map (unknown
ID or known but not provided for that resource), the proxy falls back to the
cheapest configured preference model, typically `gpt-6-luna`. When none of the
preference models are mapped, the request still fails with a configuration
error.

Default Codex model list (override with `CODEX_SUPPORTED_MODELS`):

```env
CODEX_SUPPORTED_MODELS=gpt-5.5,gpt-5.4,gpt-5.4-mini,gpt-5.3-codex,gpt-5.3-codex-spark
```

---

## Quick start

### 1. Configure

```bash
cp .env.example .env
```

**Single-mode Azure example:**

```env
AUTH_MODE=single
SERVICE_API_KEY=          # generate: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
ENABLE_AZURE=true
ENABLE_CODEX=false
AZURE_BASE_URL=https://your-resource.openai.azure.com
AZURE_API_KEY=your-azure-api-key
```

`AZURE_BASE_URL` is the resource root only. Do **not** append `/openai/v1` or `/openai/responses` — the proxy builds the Responses URL.

**Single-mode Codex example:** run `codex login` first, then:

```env
AUTH_MODE=single
SERVICE_API_KEY=your-proxy-secret
ENABLE_AZURE=false
ENABLE_CODEX=true
CODEX_AUTH_PATH=~/.codex/auth.json
CODEX_SUPPORTED_MODELS=gpt-5.5,gpt-5.4,gpt-5.4-mini,gpt-5.3-codex,gpt-5.3-codex-spark
```

**Tenant-mode Azure:** see [Authentication modes](#authentication-modes).

### 2. Start

```bash
./start.sh 8082
```

### 3. Expose publicly

Cursor’s servers must reach the proxy (reverse proxy, tunnel, or public host):

```bash
cloudflared tunnel --url http://localhost:8082
```

For a self-hosted server where multiple DNS names share one IP and need
automatic HTTPS certificates, see the [Caddy Docker Compose setup](DEPLOYMENT.md#self-hosted-mehrere-dns-namen-auf-einer-ip).

### 4. Configure Cursor

| Setting | Value |
| --- | --- |
| OpenAI API Key | `SERVICE_API_KEY` (single) or tenant cleartext key (tenant) |
| Override Base URL | `https://your-public-proxy-url` (Azure) or `…/codex` (Codex) |

Then select models in Cursor’s model picker as usual.

### Both providers on one host (single mode only)

```env
AUTH_MODE=single
ENABLE_AZURE=true
ENABLE_CODEX=true
SERVICE_API_KEY=your-shared-secret
```

Use `/azure` for Azure clients and `/codex` for Codex clients. Both share `SERVICE_API_KEY`.

---

## Configuration

### Auth and providers

| Variable | Default | Required when | Description |
| --- | --- | --- | --- |
| `AUTH_MODE` | `single` | — | `single` or `tenant` |
| `SERVICE_API_KEY` | — | `AUTH_MODE=single` | Shared Cursor Bearer secret |
| `TENANTS` | empty | `AUTH_MODE=tenant` | JSON list of tenant objects |
| `ENABLE_AZURE` | `true` | — | Enable root and `/azure` |
| `ENABLE_CODEX` | `false` | Codex use | Enable `/codex` (`AUTH_MODE=single` only) |

### Azure (single mode globals)

| Variable | Default | Description |
| --- | --- | --- |
| `AZURE_BASE_URL` | — | Azure OpenAI resource root |
| `AZURE_API_KEY` | — | Azure API key |
| `AZURE_MODEL_DEPLOYMENTS` | identity map of supported ids | JSON Cursor model id → Azure deployment name |
| `AZURE_SUMMARY_LEVEL` | `detailed` | `auto`, `detailed`, or `concise` |
| `AZURE_VERBOSITY_LEVEL` | `medium` | `low`, `medium`, or `high` |
| `AZURE_TRUNCATION` | `disabled` | `auto` or `disabled` |

If deployment names match Cursor model ids, leave `AZURE_MODEL_DEPLOYMENTS` empty. Otherwise:

```env
AZURE_MODEL_DEPLOYMENTS={"gpt-6-luna":"your-luna-deployment","gpt-5.5":"prod-gpt55"}
```

### Tenant object fields

Each entry in `TENANTS` must include:

| Field | Description |
| --- | --- |
| `id` | Unique tenant id (also used as cache-key prefix) |
| `api_key_hash` | SHA-256 hex digest of the Cursor API key |
| `azure_base_url` | That tenant’s Azure resource root |
| `azure_api_key_env` | Name of the env var holding the Azure API key |
| `azure_model_deployments` | Non-empty JSON object: Cursor model id → Azure deployment |

`AUTH_MODE=single` must not define `TENANTS`. `AUTH_MODE=tenant` must not set `ENABLE_CODEX=true`.

### Codex

| Variable | Default | Description |
| --- | --- | --- |
| `CODEX_AUTH_PATH` | `~/.codex/auth.json` | ChatGPT auth file from `codex login` |
| `CODEX_RESPONSES_URL` | ChatGPT Codex backend | Upstream Responses URL |
| `CODEX_SUPPORTED_MODELS` | see above | Models listed at `/codex/v1/models` |
| `CODEX_MODEL_REWRITES` | empty | `source:target` pairs (for example `gpt-5.4:gpt-5.5`) |
| `CODEX_ORIGINATOR` | `codex_cli_rs` | Upstream originator header |
| `CODEX_USER_AGENT` | Codex proxy UA | Upstream user-agent |
| `CODEX_DISCOVERY_MODE` | `false` | Relax Cursor marker checks for some clients |
| `CODEX_TOKEN_REFRESH_SKEW_SECONDS` | `300` | Refresh access tokens before expiry |
| `CODEX_REQUEST_TIMEOUT_SECONDS` | `600` | Upstream read timeout |

### Logging and recording

| Variable | Default | Description |
| --- | --- | --- |
| `REASONING_DISPLAY_MODE` | `mdthinkblocks` | `none`, `mdthinkblocks`, or `thinkblocks` |
| `LOG_LEVEL` | `warning` (Compose `flask`); `debug` in `.env.example` | Gunicorn/app log level (`debug`/`info`/`warning`/`error`). Supervisord’s own log stays at `warn`. |
| `RECORD_TRAFFIC` | `off` | Write redacted fixtures under `recordings/` (keep off in production; large streams bloat disk) |
| `LOG_CONTEXT` | `off` | Log incoming request details (can be huge) |
| `LOG_COMPLETION` | `off` | Log streamed completion content |
| `LOG_REDACT` | `true` | Redact secrets in logs |

---

## Running

### Local development

```bash
./start.sh 8082
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
FLASK_APP=autoapp.py flask run -p 8082
```

### Docker (production)

```bash
docker compose up flask
```

Gunicorn listens on `127.0.0.1:5000` with health checks. When Codex is enabled, `docker-compose.yml` mounts `${HOME}/.codex` into the container so ChatGPT auth state is available:

- Run `codex login` on the Docker host.
- Keep `CODEX_AUTH_PATH=~/.codex/auth.json`.
- Do not mount the directory read-only (token refresh rewrites `auth.json`).
- Do not run multiple independent proxies or copied `auth.json` files against the same login — refresh tokens are single-use.

`/codex/ready` checks that the auth file is readable and its parent directory is writable.

### Docker (development)

```bash
docker compose --profile dev up flask-dev
```

Flask reload server on port `8082`.

---

## Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | No | Liveness |
| `GET` | `/v1/models`, `/azure/v1/models` | Bearer | OpenAI-compatible model list for the principal |
| `GET` | `/v1/providers` | Bearer | Database-tenant provider profiles with routable models and available pricing |
| `GET` | `/v1/providers/<provider>/models?profile=<name>` | Bearer | Routable models and pricing for one named database-tenant profile |
| `GET` | `/codex/v1/models` | Bearer | Codex model list |
| `GET` | `/codex/ready` | Bearer | Codex auth readiness |
| `POST` | `/v1/chat/completions`, `/azure/…` | Bearer | Chat completions → Azure |
| `POST` | `/codex/v1/chat/completions` | Bearer | Chat completions → Codex |
| `POST` | `/v1/responses`, `/azure/…` | Bearer | Responses → Azure |
| `POST` | `/codex/v1/responses` | Bearer | Responses → Codex |
| `*` | `/*` | Bearer | Catch-all proxy |

Provider discovery is available only with `AUTH_MODE=tenant` and
`TENANT_CONFIG_SOURCE=database`. `/v1/models` remains OpenAI-compatible and lists
models available to the authenticated tenant. `/v1/providers` lists each ready
profile as `{provider, name, models}`; `name` is the configured display name
(or `null` for an unnamed profile). Models expose the native `id` and, for
named profiles, a provider-qualified `qualified_id` using
`provider:display-name/native-id`; reserved characters in the display name are
percent-encoded in this identifier. The `profile` query parameter must also be
URL-encoded. Select a profile with
`/v1/providers/<provider>/models?profile=<URL-encoded display
name>`; slashes and other reserved characters in the profile name must be
URL-encoded. Unnamed profiles appear in the list but cannot be requested
individually. Requests outside database tenant mode receive HTTP 400, missing
authentication receives 401, and unknown providers/profiles receive 404. Both
routes read only the authenticated request's `available_profiles` snapshot, do
not query global configuration or other tenants, and expose no internal profile
or tenant IDs, secrets, Azure deployment names, or raw settings.

When a complete published price is available, each model includes `pricing`
with `input_per_1m_tokens`, `output_per_1m_tokens`, `cache_per_1m_tokens`,
`currency`, `source`, and `updated_at`. Prices are decimal strings per million
tokens; cache means cache-read pricing. The pricing object is omitted when any
required rate is unavailable. For example, OpenRouter can publish all three
rates, whereas an OpenAI or Azure catalog item has no `pricing` field:

```json
{
  "object": "list",
  "data": [
    {
      "provider": "openrouter",
      "name": "Research/Team",
      "models": [
        {
          "id": "anthropic/claude-3.7",
          "qualified_id": "openrouter:Research%2FTeam/anthropic/claude-3.7",
          "pricing": {
            "input_per_1m_tokens": "1.25",
            "output_per_1m_tokens": "2.5",
            "cache_per_1m_tokens": "0.25",
            "currency": "USD",
            "source": "https://openrouter.ai/api/v1/models",
            "updated_at": "2026-10-05T00:00:00+00:00"
          }
        }
      ]
    }
  ]
}
```

Without complete rates, the model item contains no `pricing` property:

```json
{
  "provider": "openai",
  "name": null,
  "models": [{"id": "gpt-4.1"}]
}
```

---

## Smoke tests

```bash
curl http://127.0.0.1:8082/health

# Single mode
curl -H "Authorization: Bearer $SERVICE_API_KEY" \
  http://127.0.0.1:8082/v1/models

# Tenant mode — use a tenant cleartext key
curl -H "Authorization: Bearer $TENANT_CLEARTEXT_KEY" \
  http://127.0.0.1:8082/v1/models

# Database tenant mode — provider profiles and a named profile's models
curl -H "Authorization: Bearer $TENANT_CLEARTEXT_KEY" \
  http://127.0.0.1:8082/v1/providers
curl -G -H "Authorization: Bearer $TENANT_CLEARTEXT_KEY" \
  --data-urlencode "profile=Research/Team" \
  http://127.0.0.1:8082/v1/providers/openrouter/models

# Codex (AUTH_MODE=single, ENABLE_CODEX=true)
curl -H "Authorization: Bearer $SERVICE_API_KEY" \
  http://127.0.0.1:8082/codex/ready
curl -H "Authorization: Bearer $SERVICE_API_KEY" \
  http://127.0.0.1:8082/codex/v1/models
```

---

## Architecture

```text
Cursor ──► Proxy ─┬─► Azure OpenAI     / and /azure
                  └─► Codex/ChatGPT    /codex  (AUTH_MODE=single only)

app/
  blueprint.py              Routing, provider gates, model discovery
  auth.py                   Bearer auth → tenant principal or single key
  tenants.py                AUTH_MODE, TENANTS parse/validate, key digests
  settings.py               Environment configuration
  models.py                 Supported Azure model ids + deployment parsing
  azure/
    adapter.py              Azure orchestrator
    request_adapter.py      Cursor → Azure Responses (+ tenant isolation)
    response_adapter.py     Azure SSE → Chat Completions SSE
  codex/
    adapter.py              Codex provider
    auth_state.py           ChatGPT auth refresh
    request_adapter.py      Cursor → Codex Responses
    response_adapter.py     Codex SSE → Chat Completions SSE
    upstream.py             Codex headers and HTTP
  common/                   SSE, logging, recording, usage report
```

**Request flow**

1. Authenticate Bearer token (`single` or `tenant`).
2. Select Azure (root/`/azure`) or Codex (`/codex`).
3. Adapt the request to the provider Responses API.
4. Forward upstream; stream and adapt the response back to Cursor.

---

## Development

```bash
flask test              # pytest with coverage
flask test -k "tenant"  # filter by keyword
flask lint              # black + isort + flake8
flask lint --check      # check only
```

CI runs lint and tests on Python 3.13 and uploads coverage to Codecov.

---

## Traffic recording

Set `RECORD_TRAFFIC=on` to write redacted request/response pairs under `recordings/`:

```text
recordings/
  1/
    downstream_request.json
    upstream_request.json
    upstream_response.sse
    downstream_response.sse
```

Sensitive fields are anonymized for debugging and fixtures.

---

## Troubleshooting

**`401` from the proxy**
Bearer token missing or wrong. In single mode it must match `SERVICE_API_KEY`. In tenant mode it must match a configured tenant key (cleartext); `SERVICE_API_KEY` is not a fallback. Invalid keys share one generic 401 body.

**`401` / `403` from Azure**
Check the Azure API key, resource, and deployment for that principal (global or tenant).

**`404` from Azure**
`AZURE_BASE_URL` / tenant `azure_base_url` must be the resource root. Deployment names in the active model map must exist in Azure.

**`rate_limit_exceeded` from Azure**
The proxy retries HTTP 429 responses and `error` / `response.failed` rate-limit SSE events returned with HTTP 200 while no response content or tool-call data has been sent to Cursor. Once output starts, upstream chunks are forwarded incrementally to preserve low time-to-first-token; a later failure is reported in-stream and is not retried, avoiding duplicated text or repeated tool calls. Empty SSE `error` events before output are treated as rate-limit precursors and retried; after output they are reported as an incomplete response without restarting the request. The proxy honors `retry-after-ms` / `Retry-After` from HTTP headers or SSE errors, floors short hints at 15 seconds, and caps each wait at 60 seconds; without a hint it uses 15–60 seconds of exponential backoff. Concurrent requests handled by the same worker process share the cooldown, including when a failure arrives after output has begun; Gunicorn workers do not share this in-memory cooldown with one another. Pre-output retries can keep a Cursor request open indefinitely while Azure continues to rate-limit it; a disconnected client may not stop the retry loop promptly. Quota errors such as `insufficient_quota` are returned immediately. Persistent token-limit errors still indicate that the Azure deployment TPM is too low for the parallel Cursor load.

**Cursor cannot connect**
The override base URL must be reachable from Cursor’s servers. `http://localhost:8082` is only for local checks.

**Tenant config will not start**
Ensure `AUTH_MODE=tenant`, non-empty `TENANTS`, unique ids/hashes, each `azure_api_key_env` set to a real key, each `azure_model_deployments` non-empty, `ENABLE_CODEX=false`, and no `TENANTS` when using `AUTH_MODE=single`.

**Docker + Codex `not_ready`**
Mount `${HOME}/.codex` writable into the container; confirm `CODEX_AUTH_PATH`.

**Codex auth / refresh failures**
Run `codex login` on the proxy host. Avoid shared or copied `auth.json` across processes. Refresh tokens are single-use; the proxy serializes refresh with a file lock for local workers only.

**Codex curl: `Missing Cursor Request Marker`**
Provide a session identity (for example JSON `user`, `metadata.cursorConversationId`, or `x-client-request-id`), or set `CODEX_DISCOVERY_MODE=true` for looser integration testing.

**Codex model missing**
Check `CODEX_SUPPORTED_MODELS`. `/codex/v1/models` is separate from `/v1/models`.
