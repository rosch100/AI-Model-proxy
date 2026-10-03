"""Tests for click commands defined in the application."""

import os
import re
from datetime import datetime

import click
import pytest
from click.testing import CliRunner
from flask import Flask
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.security import check_password_hash

import app.commands as commands
from app import create_app
from app.persistence.database import Database
from app.persistence.models import (
    AdminAccount,
    AuditEvent,
    Base,
    CostRefreshJob,
    ProviderProfile,
    ProviderScopeBinding,
    ProviderScopeNode,
    Tenant,
)
from app.persistence.secrets import SecretCipher
from app.tenant_commands import (
    _canonical_azure_cognitive_resource,
    _canonical_azure_resource_group,
    create_tenant_cli_app,
    tenant_commands,
)


def test_app_registers_explicit_database_migration_commands():
    """Database migrations are exposed as an explicit Flask CLI command group."""
    app = create_app("tests.settings")

    assert "db" in app.cli.commands
    assert "upgrade" in app.cli.commands["db"].commands


def test_tenants_create_generates_and_displays_only_the_cursor_key():
    """Tenant bootstrap hashes the Cursor key and never prints the password."""
    app = create_app("tests.settings")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )

    result = app.test_cli_runner().invoke(
        args=["tenants", "create", "acme"],
        input="admin@example.com\ninitial-password\ninitial-password\n",
    )

    assert result.exit_code == 0, result.output
    assert "initial-password" not in result.output
    match = re.search(r"Cursor API key: ([^\s]+)", result.output)
    assert match is not None
    with Session(engine) as session:
        tenant = session.get(Tenant, "acme")
        admin = session.scalar(
            select(AdminAccount).where(AdminAccount.tenant_id == "acme")
        )
        assert tenant is not None
        assert tenant.api_key_hash != match.group(1)
        assert tenant.custom_model_id.startswith("cursor-")
        assert len(tenant.custom_model_id) <= 128
        assert admin is not None
        assert check_password_hash(admin.password_hash, "initial-password")
    engine.dispose()


def test_tenants_import_uses_legacy_environment_without_web_app_factory(
    monkeypatch,
):
    """Legacy import is available from the dedicated standalone CLI factory."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cipher = SecretCipher.from_key("a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=")
    database = Database(engine, sessionmaker(bind=engine), cipher)
    monkeypatch.setattr(Database, "from_config", lambda config: database)
    monkeypatch.setenv(
        "TENANTS",
        '[{"id":"acme","api_key_hash":"'
        + "a" * 64
        + '","azure_base_url":"https://acme.openai.azure.com",'
        '"azure_api_key_env":"ACME_KEY","azure_model_deployments":{"gpt-5.4":"acme-gpt54"},'
        '"azure_default_model":"gpt-5.4"}]',
    )
    monkeypatch.setenv("ACME_KEY", "azure-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://db/proxy")
    monkeypatch.setenv(
        "PROVIDER_ENCRYPTION_KEY",
        "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=",
    )
    cli_app = create_tenant_cli_app()
    result = cli_app.test_cli_runner().invoke(args=["tenants", "import"])

    assert result.exit_code == 0, result.output
    assert "Imported 1 tenant(s)." in result.output
    with Session(engine) as session:
        assert session.get(Tenant, "acme") is not None
    engine.dispose()


@pytest.mark.parametrize(
    "usage_scope",
    [
        "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
        "resourceGroups/other-rg/providers/Microsoft.CognitiveServices/accounts/acme-ai",
        "/subscriptions/bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb/"
        "resourceGroups/acme-rg/providers/Microsoft.CognitiveServices/accounts/acme-ai",
    ],
)
def test_tenants_bind_scope_rejects_azure_resource_outside_billing_group(usage_scope):
    """Azure usage resource must be a child of its tenant's billing group."""
    app = Flask("tenant-scope-test")
    app.cli.add_command(tenant_commands)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )
    with Session(engine) as session:
        session.add_all(
            (
                Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="tenant-acme"),
                ProviderProfile(
                    id="profile-acme",
                    tenant_id="acme",
                    provider="azure",
                    display_name="Production",
                    settings={},
                ),
            )
        )
        session.commit()

    result = app.test_cli_runner().invoke(
        args=["tenants", "bind-billing-scope"],
        input=(
            "acme\nprofile-acme\n"
            "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
            "resourceGroups/acme-rg\n"
            f"{usage_scope}\n"
        ),
    )

    assert result.exit_code != 0
    assert "must belong to the bound Resource Group" in result.output
    with Session(engine) as session:
        assert session.scalars(select(ProviderScopeNode)).all() == []
        assert session.scalars(select(ProviderScopeBinding)).all() == []
    engine.dispose()


def test_tenants_create_rejects_empty_initial_password():
    """Bootstrap cannot create an administrator with a blank password."""
    app = create_app("tests.settings")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )

    result = app.test_cli_runner().invoke(
        args=["tenants", "create", "acme"], input="admin@example.com\n\n\n"
    )

    assert result.exit_code != 0
    assert "Aborted!" in result.output
    with Session(engine) as session:
        assert session.get(Tenant, "acme") is None
        assert session.scalar(select(AdminAccount)) is None
    engine.dispose()


@pytest.mark.parametrize(
    ("scope_id", "canonicalizer"),
    [
        (
            "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
            "resourceGroups/invalid%2Fgroup",
            _canonical_azure_resource_group,
        ),
        (
            "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
            "resourceGroups/acme-rg/providers/Microsoft.CognitiveServices/accounts/acme_ai",
            _canonical_azure_cognitive_resource,
        ),
        (
            "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
            "resourceGroups/acme-rg/providers/Microsoft.CognitiveServices/accounts/-acme",
            _canonical_azure_cognitive_resource,
        ),
        (
            "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
            "resourceGroups/acme-rg/providers/Microsoft.CognitiveServices/accounts/åcme",
            _canonical_azure_cognitive_resource,
        ),
        (
            "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
            "resourceGroups/acme²",
            _canonical_azure_resource_group,
        ),
    ],
)
def test_azure_scope_canonicalizers_reject_invalid_resource_names(
    scope_id, canonicalizer
):
    """Canonical Azure IDs reject resource names outside ARM naming rules."""
    with pytest.raises(click.ClickException, match="name is invalid"):
        canonicalizer(scope_id)


def test_tenants_create_rejects_blank_admin_username():
    """Bootstrap cannot create an administrator without a username."""
    app = create_app("tests.settings")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )

    result = app.test_cli_runner().invoke(
        args=["tenants", "create", "acme"], input="   \n"
    )

    assert result.exit_code != 0
    assert "Admin username cannot be empty." in result.output
    with Session(engine) as session:
        assert session.get(Tenant, "acme") is None
    engine.dispose()


@pytest.mark.parametrize("provider", ["azure", "openrouter"])
def test_tenants_bind_scope_creates_exclusive_bindings_and_audit(provider):
    """Provider scope binding validates hierarchy and writes one audit event."""
    app = Flask("tenant-scope-test")
    app.cli.add_command(tenant_commands)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )
    with Session(engine) as session:
        session.add_all(
            (
                Tenant(
                    id="acme",
                    api_key_hash="a" * 64,
                    custom_model_id="tenant-acme",
                ),
                ProviderProfile(
                    id="profile-acme",
                    tenant_id="acme",
                    provider=provider,
                    display_name="Production",
                    settings={},
                ),
                ProviderProfile(
                    id="profile-staging",
                    tenant_id="acme",
                    provider=provider,
                    display_name="Staging",
                    settings={},
                ),
            )
        )
        session.commit()

    result = app.test_cli_runner().invoke(
        args=["tenants", "bind-billing-scope"],
        input=(
            "acme\nprofile-acme\n"
            + (
                "/subscriptions/AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA/"
                "resourceGroups/Acme-RG\n"
                "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
                "resourceGroups/acme-rg/providers/Microsoft.CognitiveServices/"
                "accounts/acme-ai\n"
                if provider == "azure"
                else "550e8400-e29b-41d4-a716-446655440000\n"
            )
            + "yes\n"
        ),
    )

    assert result.exit_code == 0, result.output
    assert f"{provider.capitalize()} billing scope bound" in result.output
    with Session(engine) as session:
        bindings = session.scalars(select(ProviderScopeBinding)).all()
        nodes = session.scalars(select(ProviderScopeNode)).all()
        events = session.scalars(select(AuditEvent)).all()
        assert len(bindings) == (2 if provider == "azure" else 1)
        assert {binding.profile_id for binding in bindings} == {"profile-acme"}
        assert len(nodes) == (2 if provider == "azure" else 1)
        assert len(events) == 1
        if provider == "azure":
            assert {node.canonical_scope_id for node in nodes} == {
                "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/resourcegroups/acme-rg",
                "/subscriptions/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/"
                "resourcegroups/acme-rg/providers/microsoft.cognitiveservices/"
                "accounts/acme-ai",
            }
            billing_binding = next(
                binding for binding in bindings if binding.purpose == "billing"
            )
            usage_binding = next(
                binding for binding in bindings if binding.purpose == "usage"
            )
            assert usage_binding.parent_binding_id == billing_binding.id
        else:
            assert nodes[0].scope_type == "workspace"
        assert events[0].action == "billing_scope.bind"
        assert events[0].actor_id == f"uid:{os.getuid()}"
    engine.dispose()


def test_tenants_bind_openrouter_workspace_migrates_existing_account_binding():
    """Rebind an existing account scope to the required workspace UUID."""
    app = Flask("tenant-scope-test")
    app.cli.add_command(tenant_commands)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )
    with Session(engine) as session:
        tenant = Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="tenant-acme")
        profile = ProviderProfile(
            id="profile-acme", tenant_id="acme", provider="openrouter", settings={}
        )
        account = ProviderScopeNode(
            id="account-node",
            tenant_id="acme",
            provider="openrouter",
            scope_type="account",
            canonical_scope_id="legacy-account-id",
        )
        session.add_all((tenant, profile, account))
        session.flush()
        session.add(
            ProviderScopeBinding(
                id="billing-binding",
                tenant_id="acme",
                provider="openrouter",
                profile_id=profile.id,
                purpose="billing",
                node_id=account.id,
            )
        )
        session.commit()

    result = app.test_cli_runner().invoke(
        args=["tenants", "bind-billing-scope"],
        input="acme\nprofile-acme\n550e8400-e29b-41d4-a716-446655440000\nyes\n",
    )

    assert result.exit_code == 0, result.output
    assert "updated" in result.output.casefold()
    with Session(engine) as session:
        nodes = session.scalars(select(ProviderScopeNode)).all()
        bindings = session.scalars(select(ProviderScopeBinding)).all()
        events = session.scalars(select(AuditEvent)).all()
        assert len(nodes) == 1
        assert nodes[0].scope_type == "workspace"
        assert nodes[0].canonical_scope_id == "550e8400-e29b-41d4-a716-446655440000"
        assert len(bindings) == 1
        assert bindings[0].node_id == nodes[0].id
        assert events[0].action == "billing_scope.rebind"
        assert events[0].details == {
            "provider": "openrouter",
            "purpose": "billing",
            "previous_scope_type": "account",
            "previous_scope_id": "legacy-account-id",
            "new_scope_type": "workspace",
            "new_scope_id": "550e8400-e29b-41d4-a716-446655440000",
        }
    engine.dispose()


@pytest.mark.parametrize(
    ("job_status", "should_rebind"),
    [("failed", True), ("unavailable", True), ("running", False), ("success", False)],
)
def test_tenants_bind_openrouter_workspace_respects_refresh_history(
    job_status, should_rebind
):
    """Rebind only when no active or successful refresh exists."""
    app = Flask("tenant-scope-test")
    app.cli.add_command(tenant_commands)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )
    with Session(engine) as session:
        tenant = Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="tenant-acme")
        profile = ProviderProfile(
            id="profile-acme", tenant_id="acme", provider="openrouter", settings={}
        )
        account = ProviderScopeNode(
            id="account-node",
            tenant_id="acme",
            provider="openrouter",
            scope_type="account",
            canonical_scope_id="legacy-account-id",
        )
        session.add_all((tenant, profile, account))
        session.flush()
        binding = ProviderScopeBinding(
            id="billing-binding",
            tenant_id="acme",
            provider="openrouter",
            profile_id=profile.id,
            purpose="billing",
            node_id=account.id,
        )
        session.add(binding)
        session.flush()
        session.add(
            CostRefreshJob(
                id="historical-job",
                tenant_id="acme",
                provider="openrouter",
                binding_id=binding.id,
                period_start=datetime(2026, 9, 1),
                period_end=datetime(2026, 9, 2),
                source_api="openrouter",
                operation_key="openrouter:historical",
                status=job_status,
            )
        )
        session.commit()

    result = app.test_cli_runner().invoke(
        args=["tenants", "bind-billing-scope"],
        input="acme\nprofile-acme\n550e8400-e29b-41d4-a716-446655440000\nyes\n",
    )

    assert (result.exit_code == 0) is should_rebind, result.output
    with Session(engine) as session:
        node = session.get(ProviderScopeNode, "account-node")
        assert node.scope_type == ("workspace" if should_rebind else "account")
        assert node.canonical_scope_id == (
            "550e8400-e29b-41d4-a716-446655440000"
            if should_rebind
            else "legacy-account-id"
        )
    engine.dispose()


def test_tenants_bind_scope_rejects_a_scope_owned_by_another_tenant():
    """A global scope uniqueness conflict cannot transfer tenant ownership."""
    app = Flask("tenant-scope-test")
    app.cli.add_command(tenant_commands)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )
    with Session(engine) as session:
        session.add_all(
            (
                Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="tenant-acme"),
                Tenant(id="beta", api_key_hash="b" * 64, custom_model_id="tenant-beta"),
                ProviderProfile(
                    id="profile-acme",
                    tenant_id="acme",
                    provider="openrouter",
                    settings={},
                ),
                ProviderProfile(
                    id="profile-beta",
                    tenant_id="beta",
                    provider="openrouter",
                    settings={},
                ),
            )
        )
        session.commit()

    first = app.test_cli_runner().invoke(
        args=["tenants", "bind-billing-scope"],
        input=("acme\nprofile-acme\n" "550e8400-e29b-41d4-a716-446655440000\nyes\n"),
    )
    second = app.test_cli_runner().invoke(
        args=["tenants", "bind-billing-scope"],
        input=("beta\nprofile-beta\n" "550e8400-e29b-41d4-a716-446655440000\nyes\n"),
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert "provider scope is already bound" in second.output.casefold()
    with Session(engine) as session:
        bindings = session.scalars(select(ProviderScopeBinding)).all()
        assert len(bindings) == 1
    engine.dispose()


def test_tenants_bind_scope_rejects_invalid_openrouter_workspace_id():
    """Require an API-filterable workspace UUID for OpenRouter billing."""
    app = Flask("tenant-scope-test")
    app.cli.add_command(tenant_commands)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )
    with Session(engine) as session:
        session.add(
            Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="tenant-acme")
        )
        session.add(
            ProviderProfile(
                id="profile-acme", tenant_id="acme", provider="openrouter", settings={}
            )
        )
        session.commit()

    result = app.test_cli_runner().invoke(
        args=["tenants", "bind-billing-scope"],
        input="acme\nprofile-acme\nnot-a-workspace\n",
    )

    assert result.exit_code != 0
    assert "workspace ID must be a UUID" in result.output
    with Session(engine) as session:
        assert session.scalar(select(ProviderScopeBinding)) is None
    engine.dispose()


def test_tenants_bind_billing_scope_requires_exclusivity_confirmation():
    """Operator must explicitly confirm each provider scope is exclusive."""
    app = Flask("tenant-scope-test")
    app.cli.add_command(tenant_commands)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app.extensions["database"] = Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
        secret_cipher=SecretCipher.from_key(
            "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
        ),
    )
    with Session(engine) as session:
        session.add(
            Tenant(id="acme", api_key_hash="a" * 64, custom_model_id="tenant-acme")
        )
        session.add(
            ProviderProfile(
                id="profile-acme", tenant_id="acme", provider="openrouter", settings={}
            )
        )
        session.commit()

    result = app.test_cli_runner().invoke(
        args=["tenants", "bind-billing-scope"],
        input="acme\nprofile-acme\n550e8400-e29b-41d4-a716-446655440000\nno\n",
    )

    assert result.exit_code != 0
    assert "confirmation" in result.output.casefold()
    with Session(engine) as session:
        assert session.scalar(select(ProviderScopeBinding)) is None
    engine.dispose()


def test_test_command_calls_pytest_with_coverage_and_exits(mocker):
    """Invoke `test` command with defaults and ensure subprocess call args include coverage."""
    mock_call = mocker.patch("app.commands.call", return_value=0)

    runner = CliRunner()
    result = runner.invoke(commands.test)

    assert result.exit_code == 0
    mock_call.assert_called_once()
    cmdline = mock_call.call_args[0][0]
    assert cmdline == [
        "pytest",
        commands.TEST_PATH,
        "--verbose",
        "--cov=app",
        "--cov-branch",
        "--cov-report=xml",
        "--cov-report=html",
        "--cov-report=term",
    ]


def test_test_command_no_coverage_and_filter(mocker):
    """Invoke `test` with no coverage and a filter; ensure subprocess call args are correct."""
    mock_call = mocker.patch("app.commands.call", return_value=5)

    runner = CliRunner()
    result = runner.invoke(commands.test, ["-C", "-k", "unit and not e2e"])

    assert result.exit_code == 5
    mock_call.assert_called_once()
    cmdline = mock_call.call_args[0][0]
    assert cmdline == [
        "pytest",
        commands.TEST_PATH,
        "--verbose",
        "-k",
        "unit and not e2e",
    ]


def test_lint_command_invokes_tools_with_expected_order(mocker):
    """Invoke `lint` and ensure isort, black, flake8 are called in order."""
    mock_call = mocker.patch("app.commands.call", return_value=0)

    runner = CliRunner()
    result = runner.invoke(commands.lint)

    assert result.exit_code == 0
    # Expect three calls: isort, black, flake8
    assert mock_call.call_count == 3

    first_cmd = mock_call.call_args_list[0].args[0]
    second_cmd = mock_call.call_args_list[1].args[0]
    third_cmd = mock_call.call_args_list[2].args[0]

    assert first_cmd[0] == "isort"
    assert "--check" not in first_cmd

    assert second_cmd[0] == "black"
    assert "--check" not in second_cmd

    assert third_cmd[0] == "flake8"


def test_lint_command_check_mode_adds_check_flags(mocker):
    """Invoke `lint -c` and ensure --check is added to isort and black only."""
    mock_call = mocker.patch("app.commands.call", return_value=0)

    runner = CliRunner()
    result = runner.invoke(commands.lint, ["-c"])  # --check

    assert result.exit_code == 0
    assert mock_call.call_count == 3

    first_cmd = mock_call.call_args_list[0].args[0]
    second_cmd = mock_call.call_args_list[1].args[0]
    third_cmd = mock_call.call_args_list[2].args[0]

    # isort and black should receive --check
    assert first_cmd[0] == "isort"
    assert "--check" in first_cmd

    assert second_cmd[0] == "black"
    assert "--check" in second_cmd

    # flake8 should be called without --check
    assert third_cmd[0] == "flake8"
    assert "--check" not in third_cmd


def test_lint_command_exits_on_nonzero_return(mocker):
    """Ensure lint exits with the tool's non-zero code and stops after first call."""
    mock_call = mocker.patch("app.commands.call", return_value=2)

    runner = CliRunner()
    result = runner.invoke(commands.lint)

    assert result.exit_code == 2
    assert mock_call.call_count == 1
    first_cmdline = mock_call.call_args[0][0]
    assert first_cmdline[0] == "isort"
