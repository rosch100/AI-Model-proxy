# Provider Accounts and Usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support multiple named Azure/OpenAI/OpenRouter accounts per tenant with one selected active account, account-scoped provider costs, deletable account history, and privacy-preserving per-request token usage statistics.

**Architecture:** Keep each account as a `ProviderProfile`; lift provider-wide uniqueness, preserve the single `Tenant.active_profile_id`, and thread immutable profile attribution through request snapshots. First ship profile administration and correctly scoped billing data; then persist normalized inference usage from adapter terminal events and add SQL-backed UTC reports.

**Tech Stack:** Flask/Jinja2/WTForms, SQLAlchemy, Alembic, PostgreSQL, `requests`, pytest, existing Azure Responses and OpenAI-compatible SSE adapters.

**Source Spec:** [`../specs/2026-10-02-provider-accounts-usage-design.md`](../specs/2026-10-02-provider-accounts-usage-design.md) is authoritative for product behavior and data semantics.

## Global Constraints

- PostgreSQL is authoritative in `TENANT_CONFIG_SOURCE=database`; migrations are explicit, never hidden startup side effects.
- Provider secrets are encrypted at rest and never returned in HTML/API responses or logs.
- Each request uses one consistent profile snapshot; activation affects new requests only.
- Provider scopes are immutable and globally exclusive; account-specific billing cannot share a scope with another profile.
- No prompt, completion, tool-argument, or raw provider-response contents enter the new usage history.
- Usage values are recorded only when provider-reported; unavailable values remain `NULL`, never inferred from text length.
- Every admin mutation is tenant-authorized and CSRF-protected; no provider fallback is allowed.
- Preserve `AUTH_MODE=single`, environment-based tenants, existing proxy routes, and the stable Cursor-facing model ID.

---

## File Structure

- `app/persistence/models.py` — profile names/deletion-generation metadata, profile-scoped cost binding constraints, inference usage row.
- `migrations/versions/20261002_provider_accounts_usage.py` — PostgreSQL schema transition (`revision="20261002_provider_accounts_usage"`, `down_revision="20261002_passkeys"`) and tightly scoped history purge function.
- `tests/postgres_test_utils.py`, `tests/test_postgres_provider_schema.py` — isolated PostgreSQL integration tests using required `TEST_DATABASE_ADMIN_URL` (migration/schema owner) and `TEST_DATABASE_RUNTIME_URL` (application role); no silent SQLite substitution for migrations, grants, constraints, or locking.
- `app/persistence/admin_ops.py` — profile-ID create/update/activate/delete operations and audit.
- `app/persistence/repositories.py`, `app/tenants.py` — atomic snapshot carries profile ID/name/history generation.
- `app/persistence/usage.py` — idempotent request start/finalization and report queries.
- `app/providers/cost_jobs.py`, `app/providers/costs.py` — profile-ID cost collection and provider-verified cost semantics.
- `app/admin/forms.py`, `app/admin/views.py`, `app/admin/view_models.py` — account-scoped forms, routes, actions, cost and usage views.
- `app/templates/admin/settings/connection.html`, `app/templates/admin/settings/costs.html`, `app/templates/admin/settings/usage.html`, and a connection form partial if needed — account lists, create/edit forms, deletion confirmations, usage views.
- `app/azure/response_adapter.py`, `app/providers/openai_compat.py`, `app/blueprint.py` — usage capture/finalization for streamed inference.
- `app/tenant_commands.py` — explicit-profile scope binding CLI updates.
- `tests/test_admin_providers.py`, new `tests/test_admin_costs.py`, new `tests/test_usage_persistence.py`, new `tests/test_usage_reporting.py`, new `tests/test_usage_capture.py`, new `tests/test_openai_compat_usage.py`, `tests/test_response_adapter.py`, `tests/test_provider_routing.py`, new PostgreSQL migration/race tests — regression and new behavior.
- `docs/superpowers/specs/2026-10-02-provider-accounts-usage-design.md`, deployment/operator docs — resulting operational contracts.

## Delivery 1: Multi-Account Administration and Costs

### Task 1: PostgreSQL migration for multiple profiles and profile-scoped bindings

**Files:**
- Modify: `migrations/versions/20261002_provider_accounts_usage.py` (`down_revision="20261002_passkeys"`)
- Modify: `app/persistence/models.py`
- Test: `tests/test_admin_providers.py`, `tests/test_admin_costs.py`

**Interfaces:**
- Existing profile IDs and tenant active-profile FK remain stable.
- Migration preflight reads the current `alembic_version` and depends on the existing head `20261002_passkeys`.
- New columns: `ProviderProfile.display_name`, `ProviderProfile.deleted_at`, `ProviderProfile.history_generation`.
- New DB uniqueness: case-insensitive active profile names per `(tenant_id, provider)`; scope node cannot be bound to multiple profiles.
- `CostRefreshJob` and `CostUsageRecord` retain their binding FKs. The binding's profile FK becomes the normalized account attribution path, so report queries join through `binding_id` and filter `ProviderScopeBinding.profile_id`.
- A preflight query rejects duplicate node bindings before the node uniqueness constraint is created.

- [ ] Add PostgreSQL migration tests asserting multiple same-provider profiles and duplicate active names rejected. Run them only when both test URLs are configured; otherwise mark the PostgreSQL integration job skipped (never fall back to SQLite).
- [ ] Add migration preflight test with the same `node_id` referenced more than once; migration must stop with collision details and leave schema/data unmodified.
- [ ] Add migration test preserving existing active profile, encrypted credentials, catalog rows, billing bindings, refresh jobs and cost records.
- [ ] Add an environment-import regression: an existing tenant with multiple same-provider profiles and an identifiable active matching Azure profile imports idempotently without mutating other accounts; ambiguous active profile resolution fails closed.
- [ ] Implement migration. Remove `uq_profile_tenant_provider` and `uq_binding_tenant_purpose`; retain composite tenant/provider FKs and add case-insensitive name and unique-node binding constraints. Add tombstone/generation columns. Costs remain linked by immutable binding ID; queries resolve profile via `ProviderScopeBinding.profile_id`. Existing profiles get stable labels (`Azure`, `OpenAI`, `OpenRouter`); do not derive labels from secrets or arbitrary settings.
- [ ] Run focused DB tests against PostgreSQL (SQLite is not evidence for PostgreSQL triggers/locking): `pytest tests/test_admin_providers.py tests/test_admin_costs.py -q`.
- [ ] Run migration downgrade/upgrade in an isolated test database and verify exact round trip for pre-existing rows.

### Task 2: Carry the active profile snapshot to all request paths

**Files:**
- Modify: `app/tenants.py`
- Modify: `app/persistence/repositories.py`
- Modify: `app/blueprint.py`
- Modify: `app/azure/adapter.py`
- Modify: `app/providers/openai_compat.py`
- Test: `tests/test_provider_routing.py`, `tests/test_tenant_auth.py`

**Interfaces:**
- `DatabaseTenantSnapshot` gains `profile_id: str | None`, `profile_name: str | None`, and `history_generation: int | None`.
- `TenantRepository.get_proxy_snapshot_by_api_key(api_key, cipher)` reads tenant plus `active_profile_id` and profile metadata in the same query.
- `catch_all` creates one `request_id = uuid4().hex` for the authenticated request and passes that stable ID through streaming.

- [ ] Write tests proving snapshots identify exactly the active profile, and are unaffected if `Tenant.active_profile_id` changes after snapshot creation.
- [ ] Add tests for no active profile and deleted tombstone: both reject forwarding with explicit configuration errors; neither selects another profile.
- [ ] Update `DatabaseTenantSnapshot` and repository construction sites, including no-profile and environment-configured paths. For idempotent environment imports, resolve the existing active Azure profile; if multiple Azure profiles exist without a unique active Azure target, fail with a clear ambiguity error rather than calling scalar on a non-unique provider query or overwriting another profile.
- [ ] Thread the snapshot through Azure and OpenAI-compatible adapters without querying active profile again during streaming.
- [ ] Run `pytest tests/test_provider_routing.py tests/test_tenant_auth.py -q`; expect snapshot metadata populated for the active DB profile and existing environment-mode tests unchanged.

### Task 3: Profile-ID persistence operations and admin account CRUD

**Files:**
- Modify: `app/persistence/admin_ops.py`
- Modify: `app/admin/forms.py`
- Modify: `app/admin/views.py`
- Modify: `app/admin/view_models.py`
- Modify: `app/templates/admin/settings/connection.html`
- Add: `app/templates/admin/settings/profile_form.html` (or a partial if current structure supports it)
- Test: `tests/test_admin_providers.py`

**Interfaces:**
- `create_provider_profile(session, cipher, tenant_id, provider, display_name, settings, default_model, inference_secret, actor_id) -> ProviderProfile`
- `update_provider_profile(session, cipher, tenant_id, profile_id, display_name, settings, default_model, inference_secret, actor_id) -> ProviderProfile`
- `activate_provider_profile(session, tenant, profile_id, actor_id) -> ProviderProfile`
- The UI should use provider-specific fields for new/edit workflows, but provider identity is immutable after creation.
- `display_name` is required for visible profiles and is set to `NULL` only after `deleted_at` by the transactional account tombstone operation.

- [ ] Add service tests for create, update, duplicate case-folded name, missing/foreign profile, secret masking/preservation, and provider immutability. Start with `test_create_provider_profile_is_inactive_and_tenant_scoped`: create two accounts under `acme`, assert their IDs and names differ and `Tenant.active_profile_id` is unchanged, then assert a query constrained to another tenant returns no profile.
- [ ] Add route tests that foreign tenant profile IDs never reveal profile existence or mutate data; all POST routes require CSRF.
- [ ] Implement account listing grouped by provider with name, model/catalog/configuration state and active marker.
- [ ] Implement “Neuer Account” provider selection followed by provider-specific fields; save creates inactive account and does not alter active profile.
- [ ] Implement edit and explicit activate-by-profile-ID actions. A single `Tenant.active_profile_id` is the only active reference; optional deactivate sets it to `NULL` and explains proxy unavailability.
- [ ] Replace legacy provider-only save/activate/catalog handlers with profile-ID handlers; preserve validation and PRG behavior.
- [ ] Run `pytest tests/test_admin_providers.py -q`.

### Task 4: Make catalog and cost operations profile-specific

**Files:**
- Modify: `app/persistence/admin_ops.py`
- Modify: `app/admin/views.py`
- Modify: `app/admin/forms.py`
- Modify: `app/providers/cost_jobs.py`
- Modify: `app/providers/costs.py`
- Modify: `app/admin/view_models.py`
- Modify: `app/templates/admin/settings/costs.html`
- Modify: `app/templates/admin/settings/connection.html`
- Modify: operator binding CLI in `app/tenant_commands.py`
- Test: `tests/test_admin_providers.py`, new `tests/test_admin_costs.py`, provider collector tests

**Interfaces:**
- `load_billing_binding(session, tenant_id, profile_id) -> tuple[ProviderProfile, ProviderScopeBinding, ProviderScopeNode]`
- `collect_provider_costs(cipher, profile, canonical_scope_id, start, end) -> list[CostBucket]` has no implicit inference-secret fallback.
- Refresh jobs and cost records resolve account identity through their immutable profile-scoped `binding_id`; report queries filter by tenant and joined `ProviderScopeBinding.profile_id`.

- [ ] Add this regression test before changing the collector; first assert it fails because profile IDs are not accepted yet, then change `load_billing_binding` and assert only the requested account is returned:

```python
from uuid import uuid4

from app.persistence.models import ProviderProfile, ProviderScopeBinding, ProviderScopeNode
from app.providers.cost_jobs import load_billing_binding
from tests.admin_app import build_admin_database, seed_admin


def test_load_billing_binding_selects_requested_profile():
    database = build_admin_database()
    seed_admin(database)
    profile_ids = [str(uuid4()), str(uuid4())]
    with database.sessions.begin() as session:
        for suffix, profile_id in zip(("a", "b"), profile_ids):
            profile = ProviderProfile(
                id=profile_id, tenant_id="acme", provider="openrouter",
                settings={}, default_model=f"model-{suffix}",
            )
            session.add(profile)
            session.flush()
            node = ProviderScopeNode(
                id=str(uuid4()), tenant_id="acme", provider="openrouter",
                scope_type="account", canonical_scope_id=f"account-{suffix}",
            )
            session.add(node)
            session.flush()
            session.add(ProviderScopeBinding(
                id=str(uuid4()), tenant_id="acme", provider="openrouter",
                profile_id=profile_id, purpose="billing", node_id=node.id,
            ))
    with database.sessions() as session:
        profile, binding, node = load_billing_binding(session, "acme", profile_ids[1])
        assert profile.id == profile_ids[1]
        assert binding.profile_id == profile_ids[1]
        assert node.canonical_scope_id == "account-b"
    database.engine.dispose()
```

- [ ] Test the OpenRouter `/credits` result is represented as lifetime, never relabeled as a chosen window; test unsupported period queries do not produce fake daily costs.
- [ ] Test response scope/dimensions and currency are verified before cost records are written; Azure resource-group aggregates without exact Cognitive resource attribution remain unavailable.
- [ ] Update cost collector and views to use explicit profile ID; resolve profile through the immutable binding and show account name/scope/source/freshness/granularity.
- [ ] Include `profile_id` and `binding_id` in the refresh `operation_key` so concurrent same-provider account refreshes cannot collide.
- [ ] Remove OpenAI inference-key fallback for billing; missing billing credentials/privileges show a profile-scoped unavailable status without affecting inference.
- [ ] Change binding CLI to require explicit profile ID and reject scope reuse across profiles; no automatic scope assignment on account creation.
- [ ] Test OpenAI/OpenRouter billing credential masking and no cross-profile secret overwrites.
- [ ] Run `pytest tests/test_admin_costs.py tests/test_admin_providers.py -q`.

### Task 5: Tombstone account deletion and audited history purge

**Files:**
- Modify: `migrations/versions/20261002_provider_accounts_usage.py`
- Modify: `app/persistence/admin_ops.py`
- Modify: `app/persistence/models.py`
- Modify: `app/admin/views.py`
- Modify: `app/templates/admin/settings/connection.html`
- Modify: `app/templates/admin/settings/costs.html`
- Test: `tests/test_admin_providers.py`, `tests/test_admin_costs.py`

**Interfaces:**
- `purge_profile_history(session, tenant_id, profile_id, actor_id, *, delete_profile: bool) -> None` invokes one fixed DB function; it cannot receive table names or scopes.
- Account deletion requires inactive profile, preserves profile ID/tombstone and scope nodes/bindings, NULLs secrets, clears settings/name/model, sets `deleted_at`, increments generation, and purges cost and inference history atomically.
- Historical cost rows keep their `binding_id`; the profile tombstone and binding remain, so account attribution remains stable after account deletion.

- [ ] Add Postgres tests proving the runtime role cannot directly update/delete append-only records; `PUBLIC` cannot execute purge; the app role can execute only the fixed function, owned by the migration-created non-login least-privilege role. If deployment cannot provide distinct `TEST_DATABASE_ADMIN_URL`/`TEST_DATABASE_RUNTIME_URL` roles, fail closed: do not claim DB-level isolation; keep purge unavailable until role provisioning is supplied.
- [ ] Add tests that purge function authorizes profile/tenant tuple, writes secret-free audit in the same transaction, preserves scope/audit records, and rolls back fully on injected FK/trigger error. Session/CSRF/tenant authorization happens in app before calling the function.
- [ ] Add race tests using two PostgreSQL transactions: finalization vs purge, cost refresh vs purge, activation vs account removal. Acquire locks in the documented order (tenant then profile) and verify no post-purge stale inserts.
- [ ] Add separate “Verlauf löschen” and inactive-only “Account entfernen” forms with explicit confirmation and impact summary.
- [ ] Implement tombstoning and history-generation compare under the profile row lock; all list, activation, edit, catalog and billing queries exclude tombstones.
- [ ] Run `pytest tests/test_admin_providers.py tests/test_admin_costs.py -q` on PostgreSQL.

## Delivery 2: Per-Request Token Usage and Time Statistics

### Task 6: Persist one privacy-preserving usage row per request

**Files:**
- Modify: `app/persistence/models.py`
- Modify: `migrations/versions/20261002_provider_accounts_usage.py`
- Add: `app/persistence/usage.py`
- Test: new `tests/test_usage_persistence.py`

**Interfaces:**
- `UsageResult` is a frozen value with nullable nonnegative integer fields `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens`, `reasoning_tokens`, plus a fixed source enum and optional provider request ID.
- `UsageRecorder.start(*, request_id, snapshot, model, started_at) -> None` inserts one `started` row in a short transaction.
- `UsageRecorder.finalize(request_id, *, finished_at, status, usage, source, provider_request_id, expected_generation) -> bool` locks tenant/profile, checks generation/tombstone, and performs the sole terminal transition.
- `UsageRecorder.fail(request_id, *, status, finished_at) -> bool` finalizes a started request without inventing token values.
- `request_id` is generated once in `blueprint.catch_all` (`uuid4().hex`), globally unique from first insert; `started -> terminal` is the only update transition; terminal entries are immutable.
- `InferenceUsageRequest.status` values are `started`, `succeeded_with_usage`, `succeeded_without_usage`, `failed`, and `interrupted`; only the transition out of `started` is permitted.
- Each recorder call opens and commits its own short DB session; `catch_all` attempts `start` before forwarding for persisted profiles, but metrics errors are logged and do not alter proxy behavior. Every pre-stream/upstream error finalizes `failed` once; adapter terminal events finalize success once.

- [ ] Add `test_usage_start_and_terminal_transition_are_unique` with this contract; run it red, implement the repository/trigger, rerun green:

```python
from datetime import datetime, timezone
from sqlalchemy import select

from app.persistence.models import InferenceUsageRequest


def test_usage_start_and_terminal_transition_are_unique(database, usage_recorder, active_snapshot):
    request_id = "req-usage-001"
    now = datetime.now(timezone.utc)
    usage_recorder.start(
        request_id=request_id,
        snapshot=active_snapshot,
        model="gpt-5.4",
        started_at=now,
    )
    usage_recorder.finalize(
        request_id,
        finished_at=now,
        status="succeeded_without_usage",
        usage=None,
        source=None,
        provider_request_id=None,
        expected_generation=active_snapshot.history_generation,
    )
    assert usage_recorder.finalize(
        request_id,
        finished_at=now,
        status="succeeded_without_usage",
        usage=None,
        source=None,
        provider_request_id=None,
        expected_generation=active_snapshot.history_generation,
    ) is False
    with database.sessions() as session:
        rows = list(session.scalars(select(InferenceUsageRequest).where(
            InferenceUsageRequest.request_id == request_id
        )))
    assert len(rows) == 1
    assert rows[0].status == "succeeded_without_usage"
    assert rows[0].input_tokens is None
```

- [ ] Test generation mismatch/deleted tombstone prevents stale finalize after clear/delete; duplicate finalize is idempotent and cannot double-count.
- [ ] Add database check constraints, one global unique index on `(request_id)` from initial `started` insert, tenant/profile/time/status reporting index, bounded nullable model/provider request IDs, and a trigger that permits exactly the one `started -> terminal` transition then rejects updates/deletes.
- [ ] Implement `UsageRecorder.start`, `finalize`, and `fail` using a consistent lock order on tenant then profile; the start insert is attempted before upstream forwarding, every pre-stream/upstream error path finalizes `failed` once, and each method opens/commits its own session. A failed metrics write is isolated from the response path.
- [ ] Add a CLI command `flask usage-sweep --older-than-seconds 900 --limit 500` that changes stale `started` rows to `interrupted` in bounded batches; it writes no token values.
- [ ] Run `pytest tests/test_usage_persistence.py -q` against PostgreSQL.

### Task 7: Capture Azure usage from the terminal Responses event

**Files:**
- Modify: `app/azure/response_adapter.py`
- Modify: `app/azure/adapter.py`
- Modify: `app/blueprint.py`
- Test: `tests/test_response_adapter.py`, new `tests/test_usage_capture.py`

**Interfaces:**
- `AzureAdapter.forward(req, snapshot, request_id, usage_recorder)` uses the already-resolved immutable snapshot; `ResponseAdapter` invokes `usage_recorder.finalize(...)` exactly once after a terminal event.
- Azure response adapter extracts normalized counts from `response.completed`; `response.failed` or premature EOF finalizes with `failed`/`interrupted` and no invented counts.
- `UsageResult` contains normalized counts and a fixed source enum only; it never retains raw response bodies.

- [ ] Add `test_completed_usage_is_persisted_even_without_client_usage_request` with a recorded `response.completed` event; assert the recorder receives normalized token fields once and the downstream stream contract remains unchanged, then run red and green.
- [ ] Move existing `response.completed` usage extraction into a normalized, validated result; keep Cursor terminal usage chunk behavior unchanged.
- [ ] Connect callback to persistence using immutable snapshot profile ID/generation and request ID.
- [ ] Ensure adapter callbacks execute in Flask request context or use explicit immutable callback dependencies; do not query a new active profile.
- [ ] Run `pytest tests/test_response_adapter.py tests/test_usage_capture.py -q`.

### Task 8: Parse OpenAI/OpenRouter SSE usage while streaming through unchanged

**Files:**
- Modify: `app/providers/openai_compat.py`
- Add: focused SSE parser module under `app/providers/` if required by a distinct responsibility
- Test: new `tests/test_openai_compat_usage.py`

**Interfaces:**
- A bounded incremental parser accepts byte chunks and emits normalized terminal `UsageResult | None`; forwarding yields the exact original bytes to the client.

- [ ] Add failing tests where JSON/SSE frame boundaries split across arbitrary chunks, include `data:` and `[DONE]`, and the final usage frame arrives before termination.
- [ ] Test request option `stream_options.include_usage` is enabled upstream independent of the incoming Cursor option, while downstream output only exposes the extra terminal usage chunk when requested.
- [ ] Implement a bounded incremental SSE tee. When the client requests usage, forward the upstream usage-only frame unchanged; otherwise consume/drop only that terminal usage-only frame. Forward every other original byte unchanged, including `[DONE]`. Malformed/oversized frames mark usage unavailable without corrupting or indefinitely buffering the response.
- [ ] Test OpenAI-compatible upstream errors and empty streams; on disconnect or missing terminal marker persist `interrupted` or unavailable usage exactly once, then close upstream. Duplicate terminal frames cannot finalize twice.
- [ ] Persist normalized usage once at terminal completion and ensure instrumentation failures do not alter streamed bytes/status.
- [ ] Run `pytest tests/test_openai_compat_usage.py -q`.

### Task 9: SQL aggregates and per-account statistics UI

**Files:**
- Modify: `app/persistence/usage.py`
- Modify: `app/admin/views.py`
- Modify: `app/admin/view_models.py`
- Add: `app/templates/admin/settings/usage.html`
- Modify: `app/templates/admin/settings/costs.html` or connection account partials for account links
- Test: new `tests/test_usage_reporting.py`, admin route tests

**Interfaces:**
- `query_usage_summary(session, tenant_id, profile_id, start, end, bucket_seconds) -> list[UsageBucket]`
- `list_usage_requests(session, tenant_id, profile_id, start, end, *, cursor, limit) -> UsagePage`

- [ ] Add `test_query_usage_summary_scopes_profile_and_keeps_unknown_totals`: seed `profile-a` with one terminal row of 120 input tokens and one row with all token fields NULL, then seed `profile-b` with one terminal row. Call `query_usage_summary(session, "acme", "profile-a", start, end, 300)` and assert request count `2`, input sum `120`, missing-usage count `1`, and no contribution from `profile-b`; run red, implement SQL filtering/bucketing, rerun green.

- [ ] Test hourly and daily UTC boundaries, 5-minute/hour/day bucket widths, tenant/profile filters, NULL usage counts, request pagination and 30-day maximum range.
- [ ] Implement SQL aggregation keyed by tenant/profile/UTC time bucket; aggregate count, missing-usage count and available input/output/total/cache/reasoning tokens separately.
- [ ] Add choices for last hour, last 24 hours, last 7 days and last 30 days; default last 24 hours; render UTC labels explicitly.
- [ ] Add paginated request table newest-first with account/model/status/known token fields and clear “Usage nicht geliefert” / “Anfrage unterbrochen” states.
- [ ] Separate proxy token usage from provider billing snapshots; never divide account costs into request costs or combine currencies.
- [ ] Add a profile selector/list link, history-clear action, and tenant-scoped profile validation.
- [ ] Run `pytest tests/test_usage_reporting.py tests/test_admin_providers.py tests/test_admin_costs.py -q`.

### Task 10: Regression, documentation, and release verification

**Files:**
- Modify: `README.md`, `DEPLOYMENT.md`, `.env.example`, `app/migrations.py`, `migrations/env.py`, relevant operator docs and `docs/superpowers/specs/2026-10-01-tenant-provider-admin-design.md`
- Test: full repository suite and migration integration suite

- [ ] Document explicit migration/upgrade, provider binding by profile ID, scope uniqueness, the active-account switch, billing-key requirements, provider usage availability, and account/history deletion semantics.
- [ ] Verify specs do not claim unsupported provider metrics: OpenRouter credits remain lifetime only; OpenAI/Azure values require verified scope/credential; absent usage stays unknown.
- [ ] Test `AUTH_MODE=single`, environment tenant import, `/models`, Azure/OpenAI/OpenRouter routes, and a mid-stream active-profile switch regression.
- [ ] Run `source .venv/bin/activate && flask lint` and `source .venv/bin/activate && pytest -k ""` from the repository root.
- [ ] Run PostgreSQL migration upgrade from the currently deployed schema and test rollback in a disposable database; do not use SQLite for PostgreSQL RLS/trigger/locking evidence.
- [ ] Review the final implementation diff and address all medium-or-higher findings before release.

## Coverage Map

- Account lists/create/provider-specific fields/update/activate: Tasks 1–3.
- Multiple accounts and one active snapshot across workers: Tasks 1–3 and 10.
- Profile-specific costs, cost APIs, keys, scope constraints and cost UI: Tasks 1, 4, 5, 9, 10.
- Account removal, history deletion, append-only exception and races: Tasks 1, 5, 6, 9.
- Per-request metadata, Azure/OpenAI/OpenRouter token capture and missing usage: Tasks 6–8.
- Individual request list, hourly/daily aggregates and UTC windows: Tasks 6, 9.
- Privacy, tenant isolation, regressions and docs: Tasks 2–10.
