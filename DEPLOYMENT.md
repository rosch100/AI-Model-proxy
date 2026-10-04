# Deployment: Cursor → Azure OpenAI proxy

The upstream [README](README.md) documents the proxy itself. This file covers
the existing Azure Container Apps deployment, its local patches, and the
self-hosted Docker Compose option with Caddy for multiple DNS names on one IP.

Set up on 2026-08-18 (see Cursor chat history for the full walkthrough).

## Why this exists

Cursor's BYOK ("Override OpenAI Base URL") sends plain OpenAI Chat Completions
requests from Cursor's servers — so the endpoint must be publicly reachable,
and Azure OpenAI's Responses API isn't directly compatible. This proxy sits in
between and translates both directions, streaming included.

```
Cursor (BYOK) ──▶ Azure Container App (this proxy) ──▶ Azure OpenAI (gpt-5.6-sol)
```

## Self-hosted: mehrere DNS-Namen auf einer IP

Der Docker-Compose-Stack nutzt Caddy als öffentlichen Reverse-Proxy. Alle in
`PUBLIC_HOSTNAMES` aufgeführten DNS-Namen können auf dieselbe statische
öffentliche IPv4-Adresse zeigen. Caddy unterscheidet sie über TLS-SNI und den
HTTP-Hostnamen, bezieht automatisch separate ACME-Zertifikate und erneuert sie
automatisch. Dafür sind weder eine IP pro Domain noch Wildcard-Zertifikate
notwendig.

### Voraussetzungen

- Für jeden Namen einen öffentlichen `A`-Record auf dieselbe IPv4-Adresse
  setzen, zum Beispiel `proxy.altanis.de` und `proxy.iffm-gmbh.de`.
- `AAAA`-Records nur setzen, wenn der Host und die Firewall IPv6 tatsächlich
  bis zum Container routen; ein falscher `AAAA`-Record kann Clients vom
  funktionierenden IPv4-Pfad ablenken.
- TCP-Ports 80 und 443 am Host bzw. Router zur Maschine mit Docker weiterleiten
  und in der Firewall freigeben. Caddy braucht Port 80 für HTTP-01 und
  HTTPS-Weiterleitungen sowie Port 443 für TLS-ALPN und HTTPS. UDP 443 ist
  optional und ermöglicht HTTP/3.
- Auf dem Host Docker Compose installieren und sicherstellen, dass kein anderer
  Dienst diese Ports belegt.

### Start

1. `.env.example` als `.env` kopieren und die erforderlichen Proxy-/Provider-
   Secrets setzen. `PUBLIC_HOSTNAMES` enthält ausschließlich kommagetrennte
   Hostnamen ohne Schema oder Pfad, zum Beispiel:

   ```dotenv
   PUBLIC_HOSTNAMES=proxy.altanis.de, proxy.iffm-gmbh.de
   ```

2. Stack bauen und starten:

   ```bash
   docker compose up -d --build
   docker compose logs -f caddy
   ```

3. Zertifikatsausstellung und HTTPS prüfen:

   ```bash
   curl --fail https://proxy.altanis.de/health
   curl --fail https://proxy.iffm-gmbh.de/health
   ```

Caddy speichert Zertifikate und ACME-Zustand im persistenten Volume
`caddy_data`; `caddy_config` bewahrt die aktive Konfiguration. Diese Volumes
bei Updates nicht entfernen und in die Host-Backup-Strategie aufnehmen. Bei
DNS- oder ACME-Problemen zuerst öffentliche DNS-Auflösung, Portfreigaben,
Firewall und Caddy-Logs prüfen.

Flask hat im Produktions-Compose-Stack keinen veröffentlichten Host-Port und
ist nur für Caddy über das private Compose-Netz erreichbar. Die Liste aus
`PUBLIC_HOSTNAMES` wird zugleich als Flask-Host-Allowlist verwendet. Werkzeug
`ProxyFix` vertraut genau einem Reverse-Proxy-Hop für Client-IP, Scheme und
Host; deshalb darf Flask nicht zusätzlich direkt öffentlich erreichbar sein.

SNI und `Host` bestimmen ausschließlich TLS-Zertifikat und HTTP-Routing. Sie
wählen keinen Tenant aus und umgehen keine Authentifizierung: Tenant-Zuordnung
bleibt an den Bearer-API-Key gebunden. Ein und derselbe Key muss über beide
DNS-Namen dieselbe Tenant-Konfiguration erreichen.

## Tenant Admin UI (database mode)

With `AUTH_MODE=tenant` and `TENANT_CONFIG_SOURCE=database`, the proxy serves a
browser admin UI under `/admin`. Authentication uses mandatory WebAuthn
passkeys after bootstrap:

1. Set `ADMIN_SESSION_SECRET` (≥32 random bytes), `WEBAUTHN_RP_ID`,
   `WEBAUTHN_RP_NAME`, and `WEBAUTHN_ORIGINS` (comma-separated exact origins for
   every public hostname that serves `/admin`).
2. Run migrations (`flask db upgrade`) so `admin_passkeys` /
   `admin_webauthn_challenges` exist.
3. Bootstrap: sign in once with username/password. The UI forces passkey
   enrollment before any other admin page is reachable.
4. Afterwards password login is rejected; use „Mit Passkey anmelden“.

### Provider failover order

Run `flask db upgrade` with the migration role before starting the updated app.
The migration preserves the previous primary profile at priority 1, leaves other
profiles inactive and removes `Tenant.active_profile_id`. A downgrade is refused
while any tenant has more than one active profile; deactivate extra candidates
explicitly before rolling back. Do not run old and new application versions
concurrently across this schema change.

Under **Verbindung**, load each account's catalog, choose its standard model and
activate it. Activation appends to the route. **Nach oben / Nach unten** changes
priority; removal compacts the remaining order. These operations are tenant-bound,
CSRF-protected and serialized by a tenant row lock. Catalog loss of a selected
model or an Azure endpoint change removes that account from routing. Scope
bindings and billing data remain profile-owned and do not change with priority.

The stable Cursor tenant model ID maps to each account's selected standard model;
there is no automatic matching of similar provider model names. Check that every
selected model supports the tools and capabilities your Cursor workload needs.
Root inference switches on HTTP 408/429/5xx and recognized early SSE errors before
output, not on other 4xx or after partial output. Such failover is best effort and
can cause additional processing or costs. `/azure` remains pinned to the first
active Azure profile without cross-provider failover; Codex is not a candidate.

### Operator recovery (lost last passkey)

There is no self-service or CLI recovery. On the host, delete the account's
passkey rows so password bootstrap works again, then enroll a new passkey:

```sql
DELETE FROM admin_passkeys
WHERE account_id = (SELECT id FROM admin_accounts WHERE username = 'admin');
```

When zero passkeys remain, password login is allowed once and redirects to
forced enrollment.

## Intranet-only Kerberos for `/admin`

The public proxy API stays on the existing Nginx catch-all and retains its
Bearer-token authentication. Only `/admin` and its subpaths are intercepted by
the separate SPNEGO location. Nginx first restricts those requests to explicitly
configured intranet/VPN client CIDRs, then requires Kerberos; it removes the
incoming `Authorization` header before proxying to Flask. The Flask admin's
existing account and passkey authentication remains enabled as a second layer.

### Prerequisites

- The production proxy host is `proxy.altanis.de`, with Nginx serving the site
  `/etc/nginx/sites-available/proxy.altanis.de` and Flask listening only on
  `127.0.0.1:5000`.
- The admin application is already configured in tenant database mode, including
  its database migrations, `ADMIN_SESSION_SECRET`, `WEBAUTHN_RP_ID`,
  `WEBAUTHN_RP_NAME` and `WEBAUTHN_ORIGINS`. Add the exact origin
  `https://proxy.altanis.de` to `WEBAUTHN_ORIGINS` if it is not already present.
- An AD administrator provisions the unique SPN
  `HTTP/proxy.altanis.de@ALTANIS.DE` and exports a matching keytab. Distribute
  it using the approved secret-transfer mechanism; never commit it or print its
  contents. On the proxy host, install it at `/etc/nginx/proxy-admin.keytab`
  with owner `root`, group `www-data` and mode `0640`.
- The host can reach AD KDC `192.168.253.5` on TCP and UDP port 88 through its
  private IONOS interface. The generated persistent host route uses router
  `192.168.20.31`; the router must permit and SNAT only this proxy-to-KDC
  Kerberos traffic over its AD-facing WireGuard path.
- Select actual client source CIDRs for the private network and VPN. Do not
  infer them from DNS or allow public CIDRs. Verify the addresses Nginx actually
  sees, particularly where client traffic is NATed.

### Review and deploy

1. On a trusted development machine, render the router's least-privilege nftables
   rules and review them against the router's existing table and chain
   definitions:

   ```bash
   nft -a list chain inet wireguard_filter forward
   nft -a list chain ip wireguard_nat4 postrouting
   python3 scripts/deploy_proxy_admin_iwa.py --render-router-rules \
     --router-kerberos-address <VERIFIED_ROUTER_ADDRESS> \
     --forward-chain-handle <REVIEWED_FORWARD_RULE_HANDLE> \
     --nat-chain-handle <REVIEWED_NAT_RULE_HANDLE>
   ```

   Choose existing rule handles that place the generated forward rules before
   any terminal drop/reject and the SNAT rules before broader source-NAT rules.
   The named chains must already exist, and established/related conntrack must
   be enabled for Kerberos replies. The generated `insert rule ... handle`
   commands are for controlled one-time application; persist the equivalent
   rule expressions in the router's native configuration at the same positions.
   Review the handle targets and do not repeat the commands blindly,
   since nftables rule insertion is not idempotent. Validate and apply through
   the router's normal controlled process with a rollback path; do not flush or
   reload an unrelated ruleset as a shortcut.

2. Copy this repository and the provisioned keytab to `proxy.altanis.de`, then
   run the installer with only the verified intranet/VPN CIDRs. Repeat
   `--trusted-network` for each allowed range:

   ```bash
   sudo python3 scripts/deploy_proxy_admin_iwa.py \
     --trusted-network 192.168.20.0/24 \
     --trusted-network 10.66.1.0/24 \
     --install-spnego-package
   ```

   Replace the example ranges with the client CIDRs verified for this
   environment. The installer validates the host identity, SPN/keytab ownership
   and contents, adds the persistent host route, configures the Kerberos realm,
   tests Nginx, and reloads it. Without
   `--install-spnego-package`, required Debian packages must already be
   installed. Packages installed with that flag remain installed if a later
   validation fails; on failure the installer attempts to restore every changed
   configuration file and any route it added, and reports incomplete rollback.

3. From an allowed intranet/VPN client with a valid AD Kerberos ticket, open
   `https://proxy.altanis.de/admin` and complete the existing passkey/session
   login. From a public network, `/admin` must return 404 without a SPNEGO
   challenge. Verify `/health` and a Bearer-authenticated `/v1/models` request
   from outside the intranet to confirm public proxy behavior is unchanged.

The installer intentionally does not alter the router or create/rotate AD
accounts, SPNs or keytabs. Router changes, SPN uniqueness and client CIDRs must
be reviewed by their operators before this procedure is run.

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
- **Custom model names:** `gpt-5.6-sol` (defaults to high reasoning effort),
  plus optionally `gpt-5.6-sol-low` / `gpt-5.6-sol-medium` to pick effort

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
    (`-minimal/-low/-medium/-high`) and default to `high` instead of raising
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
