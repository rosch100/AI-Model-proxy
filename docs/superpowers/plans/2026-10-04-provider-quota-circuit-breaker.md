# Provider-Quota-Circuit-Breaker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eindeutig erschöpfte Providerbudgets werden persistent und workerübergreifend pausiert, während andere Kaskadenprofile weiter bedient werden und genau eine echte Nutzeranfrage als fällige Probe dient.

**Architecture:** Eine providerneutrale Klassifikation extrahiert sichere Kategorie/Scope/Retry-After-Daten aus strukturierten Fehlern. Eine PostgreSQL-Tabelle speichert HMAC-identifizierte Breaker und Probe-Leases; Routing prüft alle Kandidatenscopes vor dem Attempt-Logging, Provideradapter melden HTTP- und SSE-Ergebnisse zurück, und das Dashboard rendert den persistenten Zustand.

**Tech Stack:** Python 3.13, Flask, SQLAlchemy, Alembic/PostgreSQL, requests/SSE, pytest.

## Global Constraints

- Unbekannte oder mehrdeutige Fehler werden niemals als Quota-Erschöpfung geraten.
- Fehlerklassifikation verwendet ausschließlich dokumentierte strukturierte Codes/Felder, niemals Freitext.
- Profile-Default ist der Scope; gemeinsame Sperren sind nur mit explizit übereinstimmender Organisations-/Projekt-ID erlaubt.
- Der Scope-Schlüssel ist HMAC-SHA256 über Tenant, Provider, Scope-Typ und Scope-ID mit domain-separated Key aus `SecretCipher`.
- Erste Pause 1 Stunde, danach 2, 4, 8, 16 und maximal 24 Stunden; ein valider Retry-After verlängert, verkürzt aber nie die Pause.
- Keine synthetischen Hintergrund-Proben. Fällige Prüfung erfolgt durch genau eine echte Anfrage mit atomarer, ablaufender DB-Lease.
- DB-Sperr-/Leasefehler dürfen nicht als geschlossener Breaker oder Erlaubnis für einen ungeschützten Upstreamaufruf behandelt werden.
- HTTP-Aufrufe finden außerhalb kurzer DB-Transaktionen statt; übersprungene Kandidaten erzeugen kein ProviderAttemptEvent.
- Keine Credentials, Rohfehlertexte, Prompts oder Antworten in Breakerzustand, Clientfehler oder Dashboard.
- Vorhandene kurze Transient-Failover-Policy, Attempt-Lifecycle und Single-/Environment-Auth bleiben erhalten.

---

### Task 1: Providerneutrale Fehlerklassifikation

**Files:**
- Create: `app/providers/error_classification.py`
- Create: `tests/test_provider_error_classification.py`
- Modify: `app/providers/failover_upstream.py`

**Interfaces:** `classify_upstream_error(provider, status, error, headers, settings, *, now=None)` liefert Kategorie, gegebenenfalls gemeldeten Scope-Typ und Retry-After-Dauer; `UpstreamError` trägt optional dieses Ergebnis durch Preflight/Fallback.

- [x] Teste OpenAI-Quota-Codes `insufficient_quota`, `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`, `organization_usage_limit_exceeded`; bestätige, dass nur explizit zuordenbare Org-/Projektfehler einen Shared Scope liefern.
- [x] Teste OpenAI `rate_limit_exceeded`, Azure 429/`rate_limit_exceeded`, OpenRouter `openrouter_in_flight_budget`, OpenRouter `openrouter_key_limit` sowie mehrdeutigen 402-Fehler.
- [ ] Teste Retry-After als Sekunden und HTTP-Datum, ungültige/negative/überlange Werte, Zeitobergrenze und dass Retry-After den Backoff nicht verkürzt.
- [ ] Führe `pytest tests/test_provider_error_classification.py -q` aus und bestätige erwartete RED-Fehler.
- [ ] Implementiere die Klassifikation nur anhand Status, strukturiertem Code/Metadata, Headern und konfigurierten Scope-IDs; verwende keine Fehlernachrichten.
- [ ] Führe die gezielte Suite erneut aus und bestätige GREEN.

### Task 2: Persistenter Zustand, Schlüsselableitung und Migration

**Files:**
- Modify: `app/persistence/secrets.py`
- Modify: `app/persistence/models.py`
- Create: `app/persistence/provider_circuit_breaker.py`
- Create: `migrations/versions/20261008_provider_circuit_breaker.py`
- Modify: `tests/test_persistence.py`
- Create: `tests/test_provider_circuit_breaker.py`
- Modify: `tests/test_postgres_provider_schema.py`

**Interfaces:** Store-Operationen berechnen Scope-Fingerprints, prüfen/beanspruchen alle fälligen Scopes atomar, öffnen/verlängern Quota-Zustände und schließen oder geben Leases tokengebunden idempotent frei. Leasezeit ist begrenzt; Persistenzfelder enthalten keine Scope-ID im Klartext.

- [ ] Schreibe Tests für Profil-/Org-/Projekt-Fingerprint-Isolation, HMAC-Domaintrennung und fehlende Roh-ID/Secret-Spalten.
- [ ] Schreibe Tests für initiale 1h-Pause, persistierte Backoffstufen bis 24h, Retry-After-Verlängerung, erfolgreiche Rücksetzung, Lease-Ablauf, Exklusivität und veraltete Token.
- [ ] Schreibe PostgreSQL-Migrations-/DDL-Tests für Unique Constraint, Checks und Lookup-Index.
- [ ] Führe die betroffenen Persistence-Tests aus und bestätige erwartete RED-Fehler.
- [ ] Implementiere kurzlebige Sessiontransaktionen und DB-seitige atomare Compare-and-Set-/Row-Lock-Operationen. Der DB-Zustand ist Fail-Closed; Storefehler werden explizit signalisiert.
- [ ] Führe gezielte Store-, Schema- und Migrationstests aus und bestätige GREEN.

### Task 3: Kaskaden-Gating, Probe und kontrollierter 503

**Files:**
- Modify: `app/providers/routing.py`
- Modify: `app/providers/failover_upstream.py`
- Modify: `tests/test_provider_failover.py`
- Modify: `tests/test_provider_route_persistence.py`

**Interfaces:** Vor `start_provider_attempt` prüft Routing Profil-, konfigurierte Org- und Projekt-Scope-Identitäten. Eine Permit enthält tokengebundene Probe-Leases; abgelehnte Kandidaten werden übersprungen. Bei ausschließlich blockierten/geleasten Kandidaten kommt `503` mit `Retry-After` bis zum frühesten sinnvollen erneuten Prüftermin.

- [ ] Teste einen offenen ersten Provider mit erfolgreichem späteren Kandidaten; der gesperrte Upstream wird nicht kontaktiert und erhält kein Attempt-Event.
- [ ] Teste alle Profile blockiert, aktive Probe-Lease sowie Storefehler: secretfreier 503, gültiger Retry-After, keine Upstreamaufrufe.
- [ ] Teste fällige Probe-Lease exklusiv, Probe durch echten Request, und voneinander unabhängige Profile/Scopes.
- [ ] Führe betroffene Routingtests aus und bestätige RED.
- [ ] Implementiere das DB-Gating vor Attempt-Logging. Verfügbarkeit anderer Profile bleibt bei blockiertem Kandidaten erhalten; bei Storefehlern wird der jeweilige Scope übersprungen.
- [ ] Führe Routing- und Persistenz-Routingtests aus und bestätige GREEN.

### Task 4: HTTP-, Preflight- und SSE-Lifecycle integrieren

**Files:**
- Modify: `app/providers/failover_upstream.py`
- Modify: `app/providers/openai_compat.py`
- Modify: `app/azure/adapter.py`
- Modify: `app/azure/response_adapter.py`
- Modify: `tests/test_provider_failover.py`
- Modify: `tests/test_openai_compat.py`
- Modify: `tests/test_response_adapter.py`

- [ ] Teste strukturierte HTTP-Quota-Fehler, frühe SSE-Quota-Fehler und späte SSE-Quota-Fehler für unterstützte Provider; Transientfehler dürfen den langlebigen Breaker nicht öffnen.
- [ ] Teste Probe-Erfolg beim erfolgreichen Preflight sowie Nicht-Quota-Probeabbruch (Breaker aufgehoben, normale Failover-Regel gilt).
- [ ] Teste, dass ein später SSE-Quota-Fehler die nächste Anfrage sperrt, aber sichtbare Ausgabe nicht wiederholt oder zurücknimmt.
- [ ] Führe die betroffenen Adapter-/Failovertests aus und bestätige RED.
- [ ] Reiche providerneutrale Klassifikation, sicher extrahierte Fehlerfelder, Header/Retry-After und konfigurierte Scope-Settings durch Adapter weiter. Lies HTTP-Fehlerkörper begrenzt; schließe Verbindungen in allen Pfaden.
- [ ] Schließe Probe-Leases beim bestandenen Preflight, öffne Quota-Zustände vor Failover und bereinige Lease-Tokens idempotent; behalte den bestehenden Attempt-Lifecycle.
- [ ] Führe Adapter-, Stream- und Failovertests aus und bestätige GREEN.

### Task 5: Adminstatus und HTMX-Refresh

**Files:**
- Modify: `app/admin/view_models.py`
- Modify: `app/admin/views.py`
- Modify: `app/templates/admin/_provider_status.html`
- Modify: `tests/test_admin_activity.py`
- Modify: `tests/test_admin_dashboard.py`

- [ ] Teste Dashboardzustand und Probezeit für Profil- und Shared-Scope-Breaker sowie secretfreie/barrierefreie Darstellung.
- [ ] Führe die gezielten Admin-Tests aus und bestätige RED.
- [ ] Lade nur Breaker-Summarydaten des angemeldeten Tenants, projiziere Shared-Scope-Zustand konsistent auf betroffene Profile und ergänze den deutschsprachigen Pausenstatus in der bestehenden 5-Sekunden-HTMX-Ansicht.
- [ ] Führe Admin-Tests aus und bestätige GREEN.

### Task 6: Abschlussverifikation

**Files:** keine zusätzlichen.

- [x] Provider-Attempt-Telemetrie mit 24 Stunden opportunistischer, indexierter Batch-Retention begrenzen.
- [ ] Prüfe Migration-Head, `git diff --check`, Scope/Fehlerpfade und Secretfreiheit über den gesamten Diff.
- [ ] Führe `source .venv/bin/activate && flask lint` aus.
- [ ] Führe `source .venv/bin/activate && pytest -k ""` aus.
- [ ] Behebe Fehler ausschließlich mit passenden regressionssichernden Tests und wiederhole beide Prüfungen.
