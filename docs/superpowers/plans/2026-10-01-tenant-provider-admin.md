# Tenant- und Provider-Admin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Auth overlay:** Mandatory passkeys after bootstrap —
[`2026-10-02-admin-passkey.md`](2026-10-02-admin-passkey.md) /
[`../specs/2026-10-02-admin-passkey-design.md`](../specs/2026-10-02-admin-passkey-design.md).

**Goal:** Tenant-Administratoren verwalten per sicherem, responsivem Jinja/HTMX-Adminbereich den Provider und das Default-Upstream-Modell, während Cursor URL, API-Key und stabile Custom-Model-ID behält. Die Installation ist auf wenige Tenants mit wenigen aktiv gepflegten Modellen je Tenant ausgelegt; starre numerische Höchstgrenzen oder SaaS-Scale-out-Anforderungen werden nicht eingeführt.

**Architecture:** Flask bleibt der einzige HTTP-Prozess und rendert Jinja-Templates; die bestehende Proxy-API bleibt unangetastet im Single-Modus und wird im verwalteten Tenant-Modus aus einem konsistenten PostgreSQL-Profil-Snapshot gespeist. SQLAlchemy 2.1 + Alembic verwalten Persistenz und Migrationen, psycopg 3 ist der PostgreSQL-Treiber, Provider-Adapter verwenden feste Upstream-Hosts. Admin-Zugang ist ein separates lokales Konto-/Session-System mit CSRF, persistenter Rate-Limitierung und tenantgebundener Autorisierung. HTMX wird lokal eingebunden und nur für klar abgegrenzte Server-Partial-Updates verwendet; normale Form-POSTs bleiben vollständig funktionsfähig. Für öffentliche Installationen wird ein vorgeschalteter Reverse-Proxy zur TLS-Terminierung und Unterstützung mehrerer DNS-Hostnamen empfohlen. Hostnamen wie `proxy.altanis.de` und `proxy.iffm-gmbh.de` dürfen auf dieselbe Flask-Instanz zeigen; der Host-Header dient weder zur Tenant-Auswahl noch als Autorisierung.

**Tech Stack:** Python 3.13, bestehendes Flask 3.1, Jinja2, SQLAlchemy 2.1, Alembic, PostgreSQL/psycopg 3, Flask-WTF CSRF, Werkzeug Passwort-Hashing, Azure Identity SDK für workloadbasierte Azure-Kostenabfragen, pytest/WebTest, HTMX aus der jeweils aktuellen stabilen npm-`latest`-Linie (lokal ausgeliefert), Vanilla-JavaScript, HTML/CSS. Exakte aktuelle stabile Dependency-Versionen werden zum Implementierungszeitpunkt verifiziert statt vorab veraltet festgeschrieben.

## Global Constraints

- Zielgröße ist eine kleine Installation mit wenigen Tenants und wenigen ausgewählten Modellen je Tenant. Es gibt keine künstliche feste Anzahlgrenze; Datenmodell, Admin-UI und Abfragen bleiben für diesen kleinen Betriebsumfang einfach und vermeiden unnötige Bulk-/Scale-out-Infrastruktur.
- Kein FastAPI-Umbau, zweiter Server, SPA-Framework, clientseitiges Routing oder unnötige Frontend-Abhängigkeit.
- Die bestehende Flask-Proxy-API, `AUTH_MODE=single`, Azure- und Codex-Legacy-Verträge müssen regressionssicher bleiben.
- `TENANT_CONFIG_SOURCE=environment` erhält den bestehenden statischen Tenantbetrieb vollständig. `TENANT_CONFIG_SOURCE=database` lädt Tenantprofile ausschließlich aus PostgreSQL; `TENANTS` ist beim DB-Web-Startup verboten und wird nur vom expliziten Import-CLI gelesen.
- Kein stiller Fallback, keine Beispieldaten, keine erfundenen Notifications und kein impliziter Providerwechsel.
- Tenant-Zugehörigkeit stammt ausschließlich aus authentifizierter Identität; Cursor-API-Keys geben keinen Zugriff auf Admin-Seiten.
- Provider-Secrets sind nie in Leseresponses/HTML/Logs sichtbar und ruhen verschlüsselt mit einem Schlüssel außerhalb der Datenbank.
- Upstream-Hosts sind providergebunden/allowlisted; freie URL-Eingaben sind ausgeschlossen.
- Alle zustandsändernden browserseitigen Aktionen verlangen CSRF-Schutz; Admin-Sessions sind kurzlebig, serverseitig widerrufbar und sicher als Cookie transportiert.
- Markdown wird UTF-8 ohne BOM gespeichert; `git diff --check` muss sauber sein.
- Testläufe im Worktree nutzen die bereits vorhandene Projekt-venv aus dem ursprünglichen Checkout; keine neue Abhängigkeit ohne fachlichen Bedarf.

---

## Datei- und Schichtkarte

- `app/admin/`: Admin-Blueprint, Session-/Auth-Services, Formvalidierung, Template-View-Modelle. Kein Proxy-Adapter-Code.
- `app/persistence/`: SQLAlchemy-Metadaten, DB-Session/Transaktionen, Tenant-/Profil-/Credential-/Session-/Audit-Repositories und CLI-Import/-Bootstrap. Nur persistenzbezogene Logik.
- `app/providers/`: Provider-Profiltypen, feste Endpoint-/Headerauflösung, Katalog-Clients und Cursor-kompatible Adapter; keine HTML- oder Flask-View-Logik.
- `migrations/`: Alembic Environment und versionierte Migrationen; keine Produktdaten/Secrets.
- `app/templates/admin/` und `app/static/admin/`: gemeinsame Base, fokussierte Partials, lokales HTMX, CSS-Variablen, minimales Vanilla-JS.
- `tests/`: getrennte Persistenz-, Security-, View-/Route-, Adapter-, Routing- und Regressionstests.
- `app/app.py`, `app/blueprint.py`, `app/auth.py`, `app/azure/request_adapter.py`: schmale Integrationsänderungen, bestehende Proxy-Verträge erhalten.
- `docker-compose.yml`, `Caddyfile`, `.env.example`, `README.md`, `DEPLOYMENT.md`, `docs/adr/0001-azure-rooted-multi-provider-proxy.md`: notwendige Betriebs- und Architekturaktualisierung einschließlich des festgelegten Caddy-Reverse-Proxys.
- Caddy terminiert öffentlich HTTPS für die konfigurierten DNS-Namen mit automatischer ACME-Ausstellung/-Erneuerung. Caddy veröffentlicht TCP 80/443 und optional UDP 443; Flask veröffentlicht keinen Host-Port und ist ausschließlich im gemeinsamen privaten Compose-Netz erreichbar. Caddys Zertifikats-/ACME-Zustand bleibt in persistenten Volumes erhalten.

## Task 1: PostgreSQL-Persistenz, Schema und Operator-CLI

**Files:**
- Create: `app/persistence/__init__.py`, `app/persistence/database.py`, `app/persistence/models.py`, `app/persistence/repositories.py`
- Create: `migrations/env.py`, `migrations/versions/<initial-schema>.py`, `alembic.ini`
- Modify: `app/settings.py`, `app/app.py`, `app/commands.py`, `requirements/prod.txt`, `requirements/dev.txt`, `docker-compose.yml`, `.env.example`
- Test: `tests/test_persistence.py`, `tests/test_tenant_import.py`, `tests/test_commands.py`

**Interfaces:**
- Produces `Database.from_config(config)`, `TenantRepository.get_by_api_key(token)`, `TenantRepository.get_admin_snapshot(tenant_id)`, `ProviderRepository.save_profile(tenant_id, provider, values)`, `ProviderRepository.activate(tenant_id, profile_id)`, `import_tenants(raw_tenants, session)`, `flask db upgrade`, `flask tenants import`, `flask tenants create` und `flask tenants bind-billing-scope`. `AUTH_MODE=tenant` requires exactly one of `TENANT_CONFIG_SOURCE=environment` (existing path) or `TENANT_CONFIG_SOURCE=database`; the import CLI reads source environment using a standalone settings loader without the DB-backed web app factory. Binding fragt Tenant, Provider und Scope interaktiv mit expliziter Exklusivitätsbestätigung ab. OpenAI bindet dedizierte Organization-ID als Parent und Project-ID als Leaf; die Organization-Admin-Credential wird nur für diese tenantexklusive Organization akzeptiert, Project wird bei jeder Antwort exakt gefiltert/geprüft. OpenRouter bindet einen exklusiven Account. Opaque Provider-IDs bleiben case-sensitiv. Azure fragt Resource-Group-Scope plus Cognitive-Services-Resource-ID ab; beide IDs müssen kanonisch gültige ARM-IDs und Parent/Child sein. Azure-ID-Segmente werden case-insensitiv kleingeschrieben und Subscription-GUIDs kanonisiert; URL, Query, Fragment, leere/redundante Segmente und unbekannte Resource-Types werden abgewiesen. Nur die exklusive Parent-Resource-Group wird global gebunden.
- Persistiert Tenant, gehashte Cursor-API-Keys, stabile tenantweite Cursor-Custom-Model-ID, Admin-Konten, Providerprofile mit verschlüsselten Inference-Credentials sowie separaten verschlüsselten OpenAI-/OpenRouter-Billing-Credentials (Azure Billing verwendet Operator-konfigurierte Workload-Identity, kein Tenant-Client-Secret), vom Operator gesetzte immutable Scope-Bindungen (Unique tenant/provider/purpose, global unique Leaf- und Parent-Scopes sowie same-tenant Composite-FKs für Azure-/OpenAI-Child-Scopes), Modellkatalog-Snapshots, zustandsbehaftete CostRefreshJobs und append-only RefreshEvents/CostUsageRecords, aktive Profilreferenz, Login-Rate-Limit-Zustand und Audit-Ereignisse mit FK/Unique-/Check-Constraints. Scope-Bindung erfolgt nur durch `flask tenants bind-billing-scope`; Tenant-Admins können nur Credentials für den gebundenen Scope setzen. OpenAI/OpenRouter haben je eine Billing-Bindung, Azure eine RG-Billing-Bindung plus darunterliegende Cognitive-Resource-Usage-Bindung; parallele/überlappende Azure-Bindungen anderer Tenants sind ausgeschlossen.
- DB-Quelle validiert `DATABASE_URL` und `PROVIDER_ENCRYPTION_KEY`; Environment-Quelle behält den bestehenden `TENANTS`-Startupvertrag. Import-CLI liest Environment-Konfiguration eigenständig, öffnet die DB separat und ruft nicht `create_app()` im Tenant-DB-Modus auf. Keine automatische DB-Erzeugung oder Migration beim App-Start.

- [ ] RED-Tests für `DATABASE_URL`/`PROVIDER_ENCRYPTION_KEY` im DB-Modus, bestehende `TENANTS`-Kompatibilität im Env-Modus, Ablehnung von `TENANTS` beim DB-Webstart, Import-CLI ohne Web-App-Factory, Bootstrap und `bind-billing-scope` CLI (Exklusivitätsbestätigung, Provider-spezifische Validierung/Kanonisierung, genau eine Bindung, OpenAI Organization-/Project- und Azure Resource-Group-/Resource-Parent-/Child-Scope-Isolation, Audit), Schema-Constraints, Tenant-Key-Hashing, verschlüsselte Inference-/Billing-Secrets, unique tenant-bound billing scope, typisierte Snapshot-Status-/Feld-/Nullsemantik samt Estimate-Provenienz und Append-only-Constraints, unveränderliche Custom-ID sowie atomaren Import und Duplicate-/Partial-Rollback schreiben.
- [ ] Tests ausführen und bestätigen, dass sie fachlich wegen fehlendem Persistenzmodell fehlschlagen.
- [ ] SQLAlchemy/Alembic/psycopg mit Installationsbefehl in die Projekt-venv holen und die aktuellsten stabilen Abhängigkeiten anhand der Requirement-Policy dokumentieren.
- [ ] Tabellen mit SQLAlchemy 2.x typed mappings, FKs, Eindeutigkeits- und Integritätsbedingungen ergänzen; kanonische Provider-Scope-IDs unique machen und Azure Child-Usage-Scope unter tenant-exklusive RG-Billing-Bindung validieren. Feld-/Nullmatrix durch DB-Constraints erzwingen: RefreshJob hat immer Provider/Scope/Zeitraum/Status/Quell-API/created_at; `retry_wait` verlangt `retry_at`; Azure-Kostenabfragen nutzen synchrone, idempotente Cost Management Query-Requests ohne providerseitige Job-ID oder Submit-Unsicherheit. Ein eindeutiger Operationsschlüssel verhindert doppelte Verarbeitung desselben Refresh-Versuchs; ein späterer expliziter Refresh erhält einen neuen Schlüssel und einen eigenen Snapshot auch für denselben Zeitraum. Ein persistent angelegter `running`-Versuch blockiert parallele Refreshes für bis zu 60 s (45-s maximaler Provideraufruf plus Sicherheitsabstand); erst danach darf ein expliziter neuer Refresh den verwaisten Versuch samt Event atomar als `failed` markieren. Jüngere Jobs dürfen nicht als verwaist angenommen werden. CostUsageRecord nur für `success`, immer Scope/Metrik/Wert/Einheit/Bucket/Quelle/Granularität; `actual`/`estimate` erfordern ISO-4217-Währung, `usage` verbietet Währung; Estimate erfordert Preisquelle/-version, usage source/bucket, Formelparameter und Modell-/Region-/Deployment-Dimension. Check-Constraints und Tests decken jede Status-/Typkombination ab. RefreshEvents/UsageRecords sind append-only; DB-Trigger verweigern UPDATE/DELETE, Tests weisen dies direkt auf Datenbankebene nach. Keine Klartext-Cursor- oder Provider-Secrets persistieren.
- [ ] PostgreSQL Secret-Envelope-Verschlüsselung mit AES-GCM (Nonce und Authentizitäts-Tag pro Wert) implementieren; außerhalb der DB gelagerten Schlüssel validieren und falsche Schlüssel-/Tagfehler klar propagieren.
- [ ] Alembic-Konfiguration und explizite Initialmigration ergänzen; Migrationen testen, ohne Schema-Create-on-startup.
- [ ] Atomaren/idempotenten `TENANTS`-Import implementieren; widersprüchlichen bestehenden Zielzustand/duplizierte IDs/fehlende Keys/Deployments mit verständlichem Fehler und Rollback ablehnen.
- [ ] `flask tenants create` implementieren: Tenant/Admin-Benutzername wird explizit abgefragt, initiales Passwort mit `click.prompt(..., hide_input=True, confirmation_prompt=True)`, Cursor-Key und stabile Custom-Model-ID kryptografisch zufällig erzeugt; nur der Cursor-Key wird nach erfolgreichem Commit einmalig ausgegeben, nie das Passwort.
- [ ] `flask db upgrade`/`flask tenants import`/`flask tenants bind-billing-scope` CLI-Routen ergänzen; Binding ist operator-only, bestätigt Exklusivität, erzwingt Unique Scope und schreibt atomar ein Audit-Event. DB Service in Compose nur für den Dev-/Vollstack-Betrieb dokumentiert starten.
- [ ] Gezielte Tests ausführen; Baseline-Tenant-Tests und `git diff --check` erneut ausführen.

## Task 2: Admin-Identität, CSRF, Sessions und Login-Schutz

**Files:**
- Create: `app/admin/auth.py`, `app/admin/security.py`, `app/admin/forms.py`, `app/persistence/admin_auth.py`
- Modify: `app/app.py`, `app/settings.py`, `requirements/prod.txt`
- Test: `tests/test_admin_auth.py`, `tests/test_admin_security.py`

**Interfaces:**
- `admin_login(username, password, remote_addr) -> AdminPrincipal | None` verifiziert scrypt/Werkzeug-Passworthash, generische Fehler und DB-Rate-Limit.
- `create_admin_session(principal) -> opaque_token` erstellt zufällige opake Token; DB speichert nur den Session-Token-Hash, Ablauf und Revocation-Status. Flask-WTF signiert den CSRF-Token mit `ADMIN_SESSION_SECRET` und speichert ihn in der Flask-Session; Login rotiert Flask-Session und opakes Admin-Cookie.
- `load_admin_principal(session_token) -> AdminPrincipal | None`, `revoke_admin_session(session_token) -> None`, `require_admin` und `require_csrf` schützen serverseitig tenantgebundene Admin-Aufrufe.
- Cursor API Bearer-Auth bleibt unabhängig und kann Admin-Route nicht autorisieren.

- [ ] RED-Tests für generische Login-Fehler, gültige/ungültige Passwörter, persistent geteilte Login-Sperre, DB-session expiry/revocation, session-token rotation, CSRF im Login und an allen Schreibmethoden schreiben.
- [ ] Tests gegen die noch fehlenden Auth-Komponenten ausführen und erwartetes fachliches Scheitern prüfen.
- [ ] `ADMIN_SESSION_SECRET` als Pflichtkonfiguration (32 zufällige Bytes, URL-safe codiert) beim App-Start validieren und als Flask-`SECRET_KEY` setzen; Cookie-Konfiguration `HttpOnly`, `SameSite=Lax`, `Secure` in Produktion und `Path=/admin` definieren. `CSRFProtect` mit `WTF_CSRF_CHECK_DEFAULT=False` registrieren und `csrf.protect()` ausschließlich im Admin-Blueprint aufrufen. Alle Formulare einschließlich Login rendern `{{ csrf_token() }}` als hidden input, HTMX sendet den Token per `X-CSRFToken`; keine Admin-Schreibroute ausnehmen.
- [ ] Werkzeug `generate_password_hash`/`check_password_hash` verwenden und Passwort-/Username-Fehler mit gleichem Body/Status/ähnlicher Prüfungsarbeit beantworten.
- [ ] Opaque Session-ID mit `secrets`, Hash-only DB-Lookup, kurze Idle-/Absolute-Expiry, serverseitigen Widerruf, Secure-if-HTTPS/production Cookie, `HttpOnly`, `SameSite=Lax`, enge Path-Angabe und Session-Rotation bauen.
- [ ] Workerübergreifendes Login-Rate-Limit persistieren; Login-Drossel nicht nur im Prozessspeicher führen.
- [ ] Security-Tests und bestehende Proxy-Auth-Tests ausführen; keine Session-/CSRF- oder Login-Fehlerdetails loggen.

## Task 3: Servergerenderte Admin-UI und Formularverträge

**Files:**
- Create: `app/admin/__init__.py`, `app/admin/views.py`, `app/admin/view_models.py`
- Create: `app/templates/admin/base.html`, `login.html`, `dashboard.html`, `settings/general.html`, `settings/connection.html`, `settings/costs.html`, `account.html`, `partials/flash.html`, `partials/provider_form.html`, `partials/model_catalog.html`
- Create: `app/static/admin/admin.css`, `app/static/admin/admin.js`, `app/static/admin/htmx.min.js`
- Modify: `app/app.py`, `Dockerfile`
- Test: `tests/test_admin_views.py`, `tests/test_admin_forms.py`

**Interfaces:**
- Flask `admin_blueprint` registriert `/admin/login`, `/admin`, `/admin/settings/general`, `/admin/settings/connection`, `/admin/settings/costs`, `/admin/account`, `POST /admin/logout`, mutierende Form-POSTs und HTMX-Partials vor dem Proxy-Catch-all. `GET /` leitet nur Browser-Anfragen mit `Accept: text/html` zu `/admin` um; Proxy-Methoden/-Inhalte bleiben unverändert.
- Jinja View-Models enthalten nur tenantberechtigte, maskierte Felder und tatsächliche Statuswerte; niemals rohe Secrets.
- Änderungen nutzen PRG (POST/Redirect/GET). HTMX-Anfragen geben ausschließlich den Zielpartial zurück; Fehler liefern formgebundene Meldungen und erhalten Eingaben.

- [ ] RED-Route-/Template-Tests: Login GET/POST, generischer Auth-Fehler, redirect für anonyme `/admin`, getrennte Proxy-Key-Auth, CSRF-Fehler, erfolgreiche und fehlerhafte General-/Connection-POSTs, foreign-tenant ID-manipulation, Logout und Partial-Response schreiben.
- [ ] Tests ausführen, erwartete fehlende Admin-Routen nachweisen.
- [ ] `admin_blueprint` mit Routen und klaren Helpergrenzen erstellen; exakt getrennte Session und Tenant-Principal authorisieren.
- [ ] OpenAI-/OpenRouter-Billing-Credential-Formular ergänzen; Azure-Workload-Identity-Status und operatorgebundenen Scope ausschließlich lesend anzeigen. Refresh-Formular enthält nur Zeitraum und niemals Scope-ID, Projekt oder Azure-Resource-URL.
- [ ] Login-/Dashboard-/Allgemein-/Verbindung-/Kosten-/Konto-Templates mit sichtbaren Labels, IDs, Help-/Error-Texten, richtigen `autocomplete`, `aria-describedby`, `aria-invalid`, Fokusziel und ohne künstliche Daten bauen; `GET /` Browser-Alias, nicht-HTML-Proxy-GET und POST mit HTML-Accept separat regressionsprüfen.
- [ ] Dashboard-Status aus realen DB-Werten ableiten: Gespeichert, Nicht konfiguriert, Prüfung erforderlich. Notifications-Seite bewusst nicht hinzufügen.
- [ ] Providerprofile nach Azure/OpenAI/OpenRouter und fachlichen Feldern gruppieren; keine beliebigen Endpoint-Felder; Secrets nur in leeres Schreibfeld und nie vorbefüllen; Azure Deployment-Zuordnung fachlich verständlich gestalten.
- [ ] POST/Redirect/GET sowie HTMX-Antwortvarianten für Speichern, Modellkatalogrefresh und Statusfeedback implementieren; Toast-/Statusbereich mit `role=status` und `aria-live` aktualisieren.
- [ ] Stylesheet mit zentralen Variablen für Farben, Text-/Borderfarben, 4/8/12/16/24/32px Skala, Control-/Buttonhöhen, Radius, Focus-Ring und Breakpoints erstellen; ruhige kompakte responsive Gestaltung umsetzen.
- [ ] Lokale Vanilla-JS-Verbesserung nur für Submit-Loading/Fokus ergänzen, die native Formulare ohne JS weiterlaufen lässt. HTMX lokal aus aktueller stabiler Distribution übernehmen; keine CDN-Laufzeitabhängigkeit.
- [ ] Views, Fehlerzustände, Accessibility-Assertions und Responsive-Browserlauf für 1440/1024/768/390 verifizieren.

## Task 4: Providerprofile, Modellkatalog und dynamischer Proxy-Vertrag

**Files:**
- Create: `app/providers/__init__.py`, `app/providers/models.py`, `app/providers/catalog.py`, `app/providers/openai.py`, `app/providers/openrouter.py`
- Create: `app/providers/costs/__init__.py`, `app/providers/costs/openai.py`, `app/providers/costs/openrouter.py`, `app/providers/costs/azure.py`, `app/providers/costs/models.py`
- Modify: `app/blueprint.py`, `app/auth.py`, `app/azure/request_adapter.py`, `app/azure/adapter.py`, `app/tenants.py`, `app/app.py`
- Test: `tests/test_provider_routing.py`, `tests/test_provider_adapters.py`, `tests/test_model_catalog.py`, `tests/test_tenant_auth.py`, `tests/test_models.py`, `tests/test_cost_adapters.py`

**Interfaces:**
- `TenantSnapshot` bündelt bei genau einem DB-read konsistent Tenant-ID, Cursor-key principal, Custom-Model-ID und aktives `ProviderProfile` (provider, credentials, default model, provider-specific map).
- `ProviderAdapter.forward(request, profile) -> flask.Response`; adapters select immutable provider base URL; Azure reuses Azure transformations after injection of runtime profile rather than changing legacy settings.
- `ProviderCatalog.refresh(profile) -> CatalogRefreshResult` records timestamp and explicit last error; only API-derived OpenAI/OpenRouter model IDs enter catalog; Azure mappings remain explicit admin-entered model/deployment tuples.
- `ProviderCostAdapter.fetch(binding, period) -> CostUsageResult` accepts only a persisted, validated tenant billing binding (never a request-selected scope) and returns typed buckets with kind (`actual`/`usage`/`estimate`), metric, Decimal value, unit, optional currency, bucket interval, verified dimensions, source endpoint/granularity and delay metadata. Provider adapters must verify response scope against the binding. Unsupported/unverifiable scope yields `unavailable`; transport/provider errors yield `failed`; neither may return fabricated rows. Azure auth uses `ManagedIdentityCredential` on Azure and operator-configured workload identity federation/certificate off Azure, with Azure RBAC only at the bound resource; no tenant-configured Azure client secret. All HTTP calls have finite timeouts; only safe/idempotent query/status operations retry boundedly with exponential backoff, honoring provider `Retry-After`/rate-limit headers.
- OpenAI adapter separately queries Costs and Usage, restricts scope to bound `project_id`, and maps model cost only for documented identifying `line_item`; OpenRouter reads `/api/v1/analytics/meta` then queries only supported metric/dimensions under dedicated account binding; Azure separates Azure Monitor usage (Cognitive resource scope) from Cost Management actuals, fetched synchronously through Query API (`2026-06-01`) at the tenant-exclusive Resource Group scope with an exact Cognitive-Resource `ResourceId` filter and grouping. Cost Details is not used because its REST API does not support Resource-Group scope. All provider cost queries are idempotent synchronous calls; there are no Azure provider-job IDs or ambiguous-submit recovery states. Connect/read timeouts configurable within hard maxima 3 s/15 s; max two attempts; total handler budget 45 s and internal retry-delay budget 5 s; honor longest Azure `Retry-After`/`x-ms-ratelimit-*-retry-after`, persist `retry_at` rather than wait past budget. No unbounded waiting or unscoped report retrieval.

- [ ] RED-Tests für tenant-`/models` genau eine stabile Custom-ID, beliebiges eingehendes Modell deterministisch auf aktives Defaultmodell gemappt, alle Worker lesen Profilwechsel beim nächsten Request, missing/invalid config rejects with no fallback, explicit `/azure`-/`/codex` semantics, legacy single unchanged schreiben.
- [ ] Adapter tests for fixed endpoints/Auth header filtering/model override/timeout/stream SSE/error response/credential secrecy and SSRF input validation write prior to implementation.
- [ ] RED-Tests je Billingprovider für feste Endpunkte, minimale Credential-Scope-Weitergabe, exakte Scope-Antwortvalidierung, supported/unsupported Dimensions, typed actual/usage values, fehlende Rechte und secretfreie Providerfehler ergänzen; Azure prüft synchrone Cost-Management-Query-Requests auf Resource-Group-Scope, `ActualCost`, exakten `ResourceId`-Filter und Gruppierung, jede Ergebniszeile gegen die Bindung, `PreTaxCost`/Währung/Tages-Bucket sowie `Retry-After` und `retry_at` ohne providerseitige Job-ID.
- [ ] Persisted TenantSnapshot query uses one transaction/read-consistent join; ensure proxy key only resolves tenant via digest and database data replaces environment config only under documented AUTH_MODE.
- [ ] `AUTH_MODE=single` remains same. Define clear boot validation and explicit `AUTH_MODE=tenant` database source; legacy `TENANTS` value accepted solely by import command then rejected as conflicting runtime source.
- [ ] Azure adapter takes profile-specific base URL/API key/deployment and maintains established Responses adaptation contract without mutating shared global adapter state.
- [ ] OpenAI Responses and OpenRouter provider adapters use fixed HTTPS endpoints, provider auth, model override, finite timeout and the current Cursor-compatible stream/response contract; implement SSE failure/termination paths.
- [ ] explicit `/azure` uses tenant's Azure profile, returns clear configuration error if absent; `/codex` stays denied for tenant mode; single mode paths preserve old behavior.
- [ ] `/models` returns exactly tenant's stable Cursor ID in database mode, while single/legacy paths retain their existing model data.
- [ ] Catalog refresh uses bounded HTTP timeouts, handles provider errors explicitly, persists successful catalog/time and query errors, does not echo credential or provider response secrets.
- [ ] Kostenadapter je Provider nach obigem Vertrag umsetzen: OpenAI Costs und Usage getrennt (nur gebundenes Projekt); OpenRouter erst bestätigte `/analytics/meta`-Dimensionen abfragen; Azure Monitor Usage und synchrone Cost Management Query Actuals auf Resource-Group-Scope trennen. Azure sendet `ActualCost` mit exaktem Cognitive-Resource-`ResourceId`-Filter und Gruppierung; jede Ergebniszeile muss den gebundenen Scope bestätigen. Cost Details, providerseitige Job-IDs und Polling entfallen. Scopeabweichungen -> `unavailable`, APIfehler -> `failed`, begrenztes `Retry-After` -> `retry_wait` mit `retry_at`; keine Modellkosten aus nicht dokumentierten Line Items oder Aggregaten ableiten.
- [ ] Run provider/tenant/model tests and all legacy routing regression tests.

## Task 5: Profilaktivierung, Account-Aktionen und Audit

**Files:**
- Modify: `app/admin/views.py`, `app/admin/forms.py`, `app/persistence/repositories.py`, `app/persistence/models.py`, `app/templates/admin/settings/connection.html`, `app/templates/admin/settings/costs.html`, `app/templates/admin/account.html`
- Test: `tests/test_admin_actions.py`, `tests/test_provider_activation.py`, `tests/test_cost_reporting.py`

**Interfaces:**
- `activate_profile(tenant_id, profile_id) -> ActiveProfile` validates owned profile, decrypted credential readiness and catalog membership/allowed Azure mapping then transactionally updates one tenant's active reference and audit row.
- `change_password(principal, current_password, new_password) -> None` revalidates current password, stores modern password hash, revokes other sessions and issues current rotated session.
- `rotate_cursor_api_key(principal) -> one-time cleartext key` stores digest only and audits actor/target (without secret).
- `refresh_costs(principal, provider, period) -> CostRefreshJob` invokes only the selected provider's official cost/usage adapter using stored OpenAI/OpenRouter billing credentials or operator-configured Azure workload identity and stored scope bindings, stores typed job/events/records and rejects unscoped data. Azure Query API requests are synchronous and idempotent; `retry_wait` blocks another explicit refresh until `retry_at`, then a new query may be issued. There is no provider-job resume or ambiguous-submit retry action.

- [ ] RED-test cross-tenant profile activation rejected, unconfigured/invalid/missing-secret profile doesn't activate, transaction rollback preserves old profile, successful activation persists once/audit records.
- [ ] RED-test password verification, current session rotation/revocation, one-time API-key rotation and secret nonappearance in subsequent views/logs.
- [ ] RED-test successful, unavailable, failed and `retry_wait` cost/usage refresh, billing-scope isolation, unsupported model dimensions, append-only events/records, job state/field nullability matrix, estimate provenance constraints and secret-free audit events; verify refresh is rejected before `retry_at` and can issue a new synchronous Azure Query afterward.
- [ ] RED-test DB-Transaktionsgrenze für Providerprofil-/Billing-Credential-Speichern, Aktivierung, Passwortwechsel und Key-Rotation (Change+Audit atomar) sowie Katalog- und Kostenrefresh (Provideraufruf außerhalb; Ergebnisstatus+Audit atomar; Fehler ebenfalls); Rollback je Persistenzfehler sichern.
- [ ] Implement DB transactions for every persistent admin mutation: profile/billing-credential settings + audit, activation + audit, password/key rotation + audit; use row locks/ownership predicates for activation and DB uniqueness or compare-and-set for one active profile. For external catalog/cost calls perform I/O outside DB transaction, then persist result/status and audit atomically; represent provider failure explicitly.
- [ ] Implement CSRF-protected confirm actions separated visually from Save; destructive/rotation buttons clearly labeled, not adjacent to primary Save; display API key once after creation/rotation and redirect to no-secret page.
- [ ] Implement `/admin/settings/costs/refresh` as a synchronous adapter invocation outside the DB transaction; persist `CostRefreshJob` transitions/events and append-only typed records atomically through official APIs. Refresh-Scope exclusively from DB bindings and verify returned provider dimensions. Store `unavailable`/`failed`/`retry_wait` states explicitly; reject refresh before `retry_at`. Estimates only with complete spec price/usage match and versioned provenance, never from usage alone.
- [ ] Test transaction failure, idempotent activation, audit event and auth invalidation at the HTTP boundary; assert cost refresh scope and audit behavior through the registered HTTP route.

## Task 6: Compose, migration workflow, ADR und Betriebshinweise

**Files:**
- Modify: `docs/adr/0001-azure-rooted-multi-provider-proxy.md`, `README.md`, `DEPLOYMENT.md`, `docker-compose.yml`, `.env.example`, `requirements/prod.txt`, `.github/workflows/lint.yml`
- Create: `Caddyfile`, `infra/azure-cost-access.bicep`, `tests/test_database_migrations.py`

**Interfaces:**
- Operator sequence `flask db upgrade` → `flask tenants import` or `flask tenants create` → optional operator-confirmed `flask tenants bind-billing-scope` per kostenberechtigtem Provider → enter one-time key into Cursor/admin login → `/admin` profile and separate billing-credential setup → explicit activation.
- Docker Compose defines PostgreSQL health/dependency for dev profile and persistent volume; application migrations remain an explicit operator command, not silent automatic startup. For public production deployments, Caddy is the selected reverse proxy: its `Caddyfile` serves the explicitly configured hostnames (for example `proxy.altanis.de` and `proxy.iffm-gmbh.de`) through one Flask instance and automatically obtains/renews ACME certificates. Both DNS names may resolve to the same public IP; TCP 80/443 reach Caddy, UDP 443 is optional for HTTP/3, and Caddy certificate state uses persistent volumes. Flask has no published host port and is reachable only on the private Compose network shared with Caddy. `PUBLIC_HOSTNAMES` is the single hostname source for Caddy and Flask's `TRUSTED_HOSTS`; Flask applies Werkzeug `ProxyFix` trusting exactly one configured proxy hop for client IP, scheme, and host. Reject unknown hosts and do not derive tenant identity, authorization, or billing scope from `Host`, SNI, or forwarded host headers. Both names use the same tenant-authenticated proxy/admin contracts unless a separate host-to-tenant policy is explicitly specified later. Optional `infra/azure-cost-access.bicep` assigns built-in Cost Management Reader at the tenant-exclusive Resource Group and Monitoring Reader only at the bound Cognitive Services resource to the supplied user-assigned managed identity principalId; it does not create identities or broaden scope.

- [ ] Update ADR in place: `AUTH_MODE=single` remains path-/Azure-rooted; DB-backed tenant root follows active profile; explicit paths, models and Codex constraints are precise.
- [ ] Document env/DB URL, outside-DB encryption key creation/storage/rotation/backup, database backup and restore, alembic upgrade, atomic import, CLI bootstrap, exclusive billing-scope binding, Azure host workload identity and least-privilege resource RBAC, initial-admin credentials handling, Caddy-based HTTPS/ACME and trusted-host/session-cookie assumptions. Document same-IP DNS records, required TCP 80/443 and optional UDP 443, persistent Caddy state, explicitly configured hostnames, Flask's private Compose-only reachability, one trusted proxy hop, rejection of unknown hosts, and that hostnames never select or authorize tenants.
- [ ] Compose adds a database service only where appropriate, health checks/volume and dev wiring; production database credentials come from runtime secret injection, not compose literals.
- [ ] Add CI PostgreSQL service or explicit integration test job for migration, transaction, locking and multiple-connection visibility. Do not claim DB concurrency based solely on SQLite tests.
- [ ] Build/lint `infra/azure-cost-access.bicep`; verify principal ID, Cost Management Reader role ID `72fafb9e-0641-4937-9268-a91bfd8191a3`, Monitoring Reader role ID `43d0d8ad-25c7-4714-9337-8ba259a9fe05`, exact RG/resource assignment scopes, and absence of subscription/account-wide assignment. Document operator `what-if`/deployment plus post-deployment role-assignment smoke check. Without an Azure subscription, leave runtime capability in `open_gaps` rather than claiming it verified.
- [ ] Verify no secret is committed, Compose configuration resolves, Caddy accepts the expanded multi-host Caddyfile, Flask rejects untrusted hosts and trusts only the configured proxy hop, both public aliases preserve API-key-based tenant identity, `flask db upgrade` applies and migration can be rolled back/reapplied in test database.

## Task 7: Full verification and UI evidence

**Files:**
- Test: all new tests; `tests/test_tenant_auth.py`, `tests/test_provider_routing.py`, existing full suite
- UI: `app/templates/admin/**`, `app/static/admin/**`

- [ ] Run targeted TDD tests after each task, then `source .venv/bin/activate && pytest -k ""` from the feature worktree (the project venv is in the original checkout).
- [ ] Run `source .venv/bin/activate && flask lint` from a temporary env with `PYTHONPATH` and `FLASK_APP` aimed at the feature worktree; confirm it formats/checks only the worktree files.
- [ ] Run alembic migration against PostgreSQL and migration/transaction integration suite on PostgreSQL, not only SQLite.
- [ ] Verify `GET /admin/login`, login success/failure, settings saves, activation, catalog errors and logout in a real browser at widths 1440, 1024, 768, 390; validate no horizontal overflow, labels/focus/errors and accessible navigation.
- [ ] Validate CSS variables, local HTMX asset, correct `aria` links, keyboard operation, no external frontend runtime calls, and `git diff --check`; verify allowed-host handling for configured aliases and prove that changing `Host`/forwarded-host headers never changes the authenticated tenant.
- [ ] Re-run the entire existing suite to protect the legacy single-mode Cursor protocol.
- [ ] Summarize delivered routes, architecture decision, test/lint/browser evidence, migration/bootstrap operational needs, and any explicitly uncovered deployment verification.
