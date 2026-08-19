# Deployment: Cursor → Azure OpenAI proxy

How this checkout is deployed and configured for Josh's setup. The upstream
[README](README.md) documents the proxy itself; this file documents *our*
deployment on Azure Container Apps and the local patches on top of upstream.

Set up on 2026-08-18 (see Cursor chat history for the full walkthrough).

## Why this exists

Cursor's BYOK ("Override OpenAI Base URL") sends plain OpenAI Chat Completions
requests from Cursor's servers — so the endpoint must be publicly reachable,
and Azure OpenAI's Responses API isn't directly compatible. This proxy sits in
between and translates both directions, streaming included.

```
Cursor (BYOK) ──▶ Azure Container App (this proxy) ──▶ Azure OpenAI (gpt-5.6-sol)
```

## Live deployment

| Thing | Value |
|---|---|
| Public URL | `https://cursor-azure-proxy.kindsmoke-cdf27eec.eastus2.azurecontainerapps.io` |
| Container App | `cursor-azure-proxy` (resource group `RG1`, eastus2) |
| Environment | `cursor-proxy-env` |
| Registry | `riphqcursorproxy.azurecr.io`, repo `cursor-azure-proxy` (currently tag `v5`) |
| Sizing | 2 vCPU / 4 GiB, min 1 replica (always warm), max 3 |
| Azure OpenAI backend | `wymetyme-5999-resource` (eastus2), deployment `gpt-5.6-sol` |

Secrets (`SERVICE_API_KEY`, `AZURE_API_KEY`) are stored as Container Apps
secrets, referenced by env vars. The rest of the config is plain env vars on
the app: `AZURE_BASE_URL`, `ENABLE_AZURE=true`, `ENABLE_CODEX=false`,
`LOG_REDACT=true`, `AZURE_MODEL_DEPLOYMENTS={"gpt-5.5":"gpt-5.6-sol"}`,
`GUNICORN_WORKERS=4`, `LOG_LEVEL=info`.

The local `.env` in this directory holds the same values and is the source of
truth when (re)creating the app. It is excluded from the image by
`.dockerignore` — never remove that entry, the Dockerfile does `COPY . .`.

## Cursor settings

In Cursor → Settings → Models → OpenAI API Key:

- **Override OpenAI Base URL:** the public URL above + `/v1`
- **API key:** the `SERVICE_API_KEY` value from `.env` (our own shared secret
  that the proxy checks — not an OpenAI or Azure key)
- **Custom model names:** `gpt-5.6-sol` (defaults to medium reasoning effort),
  plus optionally `gpt-5.6-sol-low` / `gpt-5.6-sol-high` to pick effort

### Known Cursor BYOK bugs we work around

1. **Cursor drops `reasoning.effort` over BYOK** for GPT-5.6 models
   ([forum thread](https://forum.cursor.com/t/byok-does-not-send-the-reasoning-effort-parameter-for-gpt-5-6-models/165529)).
   The effort shown in Cursor's model picker never reaches the proxy. Until
   Cursor fixes it, effort is controlled by the model-name suffix; an explicit
   `reasoning.effort` in the request wins if Cursor ever starts sending it.
2. **Cursor tool-call IDs can exceed 64 chars**, Azure's Responses API limit
   for `call_id` (Azure returns 400 `string_above_max_length` on any turn
   after a tool call). The proxy shortens long IDs deterministically.

## Local patches vs upstream

This checkout diverges from `gabrii/Cursor-Azure-GPT-5` in four places:

- `app/models.py` — added `gpt-5.6-sol` to `SUPPORTED_MODELS`.
- `app/azure/request_adapter.py` —
  - `_resolve_model_and_reasoning`: accept effort-suffixed model names
    (`-minimal/-low/-medium/-high`) and default to `medium` instead of raising
    when Cursor omits `reasoning.effort` (bug 1 above).
  - `_safe_call_id`: hash-shorten tool-call IDs over 64 chars (bug 2 above).
- `requirements/prod.txt` — added `packaging` (gunicorn's gevent worker
  imports it but doesn't declare it; the container crash-loops without it).
- `app/blueprint.py` + `app/common/logging.py` + `app/azure/response_adapter.py`
  + `app/azure/adapter.py` — request/response content (tool descriptions, error
  messages) was passed to `rich` unescaped; bracketed text like `[/igp]` in
  code raised `MarkupError` and 500'd the request, which Cursor surfaced as
  "User API Key Rate limit exceeded". Dynamic strings are now escaped and the
  logging guard catches all exceptions.
- `.dockerignore` — created (upstream has none) to keep `.env`, `.venv`, etc.
  out of the image.

Check `git diff` before pulling upstream updates; these need to survive.

## Deploying a change

Builds happen in the cloud (`az acr build`), no local Docker needed. From this
directory, bump the tag and roll it out:

```bash
az acr build -r riphqcursorproxy -t cursor-azure-proxy:v5 \
  --build-arg INSTALL_PYTHON_VERSION=3.13 --target production .
az containerapp update -g RG1 -n cursor-azure-proxy \
  --image riphqcursorproxy.azurecr.io/cursor-azure-proxy:v5
```

Smoke test:

```bash
source <(rg '^SERVICE_API_KEY=' .env)
BASE=https://cursor-azure-proxy.kindsmoke-cdf27eec.eastus2.azurecontainerapps.io
curl -sS "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $SERVICE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.6-sol","messages":[{"role":"user","content":"Reply with exactly: ok"}],"stream":true}'
```

## Operations

```bash
# Live logs
az containerapp logs show -g RG1 -n cursor-azure-proxy --type console --follow

# Replica health
az containerapp replica list -g RG1 -n cursor-azure-proxy -o table

# Update a secret (e.g. rotated Azure key), then restart by updating revision
az containerapp secret set -g RG1 -n cursor-azure-proxy --secrets azure-api-key=NEW_KEY
az containerapp revision restart -g RG1 -n cursor-azure-proxy \
  --revision $(az containerapp revision list -g RG1 -n cursor-azure-proxy --query '[0].name' -o tsv)

# Cost dial-down (proxy is pure I/O; this behaves identically at ~1/4 the cost)
az containerapp update -g RG1 -n cursor-azure-proxy --cpu 0.5 --memory 1Gi
```

Always-on at 2 vCPU / 4 GiB runs roughly $150/month before credits; 0.5 vCPU /
1 GiB is roughly $40.

## Retired local setup

Before the cloud deployment, the proxy ran locally behind a Cloudflare quick
tunnel, kept alive by a LaunchAgent. Both are retired:

- LaunchAgent plist parked at
  `~/Library/LaunchAgents/com.joshwymer.cursor-azure-proxy.plist.disabled`
  (rename back and `launchctl bootstrap gui/$UID <plist>` to resurrect a local
  fallback on port 8082).
- The tunnel URL rotated on every restart, which is why it was replaced.
