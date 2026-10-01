# Azure Deployed Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose all six Responses-capable deployments in `AzureOpenAI-Instanz2` through the Cursor-facing model catalog.

**Architecture:** Keep `app.models.SUPPORTED_MODELS` as the explicit allowlist and `AZURE_MODEL_DEPLOYMENTS` as the account-specific mapping to Azure deployment names. Preserve the existing Azure Responses request adapter and verify the model catalog, deployment routing, and environment example through the existing pytest suite.

**Tech Stack:** Python 3.13, Flask, pytest, environs, Azure Responses API.

## Global Constraints

- Do not allow arbitrary deployment IDs; every Cursor model ID remains explicitly allowlisted.
- Use the exact Azure deployment names from the read-only inventory of `AzureOpenAI-Instanz2`.
- Do not add Azure SDK dependencies or change the request protocol.
- Keep live Azure inference outside the local test run; Azure metadata confirms the deployments expose Responses API.

---

### Task 1: Expose the configured Azure deployments

**Files:**
- Create: `docs/superpowers/specs/2026-09-28-azure-deployed-models-design.md`
- Create: `docs/superpowers/plans/2026-09-28-azure-deployed-models.md`
- Modify: `tests/test_request_adapter.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_config.py`
- Modify: `app/models.py`
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/test_request_adapter.py`, `tests/test_models.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `SUPPORTED_MODELS`, `AZURE_MODEL_DEPLOYMENTS`, and `RequestAdapter.adapt`.
- Produces: six Cursor model IDs routed to their account deployment names:

| Cursor model ID | Azure deployment name |
|---|---|
| `gpt-5.6-luna` | `gpt-5-6-luna-api` |
| `gpt-5.6-sol` | `gpt-5-6-sol-api` |
| `gpt-5.6-terra` | `gpt-5-6-terra-api` |
| `gpt-6-astra` | `gpt-6-astra-api` |
| `gpt-6-luna` | `gpt-6-luna-api` |
| `gpt-6-sol` | `gpt-6-sol-api` |

- [x] **Step 1: Add failing contract tests**

Add this explicit catalog assertion to `tests/test_models.py`:

```python
assert {
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gpt-6-luna",
    "gpt-6-sol",
}.issubset(returned_models)
```

In `tests/test_request_adapter.py`, add the six `(model_name, deployment_name)` pairs from the table as `DEPLOYED_MODEL_ROUTES`, then add:

```python
@pytest.mark.parametrize(
    ("model_name", "deployment_name"),
    DEPLOYED_MODEL_ROUTES,
)
def test_request_adapter_routes_deployed_models_with_reasoning_suffix(
    app, model_name, deployment_name
):
    app.config["AZURE_MODEL_DEPLOYMENTS"][model_name] = deployment_name
    adapter = AzureAdapter().request_adapter
    request = app.test_request_context(
        "/chat/completions",
        method="POST",
        json={
            "model": f"{model_name}-high",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
            ],
            "stream": True,
        },
        headers={"Authorization": "Bearer test-service-api-key"},
    ).request

    request_kwargs = adapter.adapt(request)

    assert request_kwargs["json"]["model"] == deployment_name
    assert request_kwargs["json"]["reasoning"]["effort"] == "high"
```

In `tests/test_config.py`, assert each of the six `.env.example` entries against the deployment table with `settings.AZURE_MODEL_DEPLOYMENTS.get(model_name) == deployment_name`, so a missing mapping fails as an assertion.

- [x] **Step 2: Confirm RED**

Run:

```sh
PATH="$PWD/.venv/bin:$PATH" SERVICE_API_KEY=verification-test-service-key FLASK_APP=autoapp.py .venv/bin/pytest tests/test_request_adapter.py tests/test_models.py tests/test_config.py -q
```

Expected: the new model IDs are absent from the catalog and the new adapter cases fail with the existing unsupported-model error; the test run must not fail from syntax or fixture errors.

- [x] **Step 3: Add the four missing allowlist entries**

Add `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-6-astra`, and `gpt-6-sol` to `SUPPORTED_MODELS`. Retain the already supported `gpt-5.6-sol` and `gpt-6-luna`.

- [x] **Step 4: Configure and document exact deployment mappings**

Set `AZURE_MODEL_DEPLOYMENTS` in `.env.example` to the six mappings in the table above. Add the four missing IDs and the existing `gpt-5.6-sol` to the README model table. Add a separate README table for all six exact Azure deployment names; state that Azure reports Responses API capability but the proxy has not been verified end-to-end.

- [x] **Step 5: Confirm GREEN**

Run the same targeted pytest command from Step 2. Expected: all tests in the three selected files pass.

- [x] **Step 6: Run full verification**

Run:

```sh
PATH="$PWD/.venv/bin:$PATH" SERVICE_API_KEY=verification-test-service-key FLASK_APP=autoapp.py .venv/bin/flask test
.venv/bin/python -m compileall -q app tests
.venv/bin/isort --check app/models.py tests/test_request_adapter.py tests/test_models.py tests/test_config.py
.venv/bin/black --check app/models.py tests/test_request_adapter.py tests/test_models.py tests/test_config.py
.venv/bin/flake8 app/models.py tests/test_request_adapter.py tests/test_models.py tests/test_config.py
git diff --check
```

Expected: 0 test failures, successful compile and targeted style checks, and no whitespace errors. The diff adds only model identifiers and tests; no complexity-bearing production function changes, so the CRAP gate is not applicable.

- [x] **Step 7: Review and publish**

Review the current diff against the design, fix any concrete medium-or-higher findings, then re-review. Stage only the two design documents and six implementation/test files listed above, commit with `feat: expose configured Azure GPT-5.6 and GPT-6 deployments`, and push `feature/gpt-6-luna` to `origin`.
