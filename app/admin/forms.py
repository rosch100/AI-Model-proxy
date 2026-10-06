"""WTForms for the tenant administrator UI."""

from __future__ import annotations

import json

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    FloatField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    TextAreaField,
)
from wtforms.validators import (
    DataRequired,
    InputRequired,
    Length,
    NumberRange,
    Optional,
    Regexp,
    ValidationError,
)
from wtforms.widgets import PasswordInput

from app.providers.azure_url import validate_azure_base_url
from app.providers.routing_config import (
    parse_profile_routing_settings,
    parse_tenant_routing_settings,
)
from app.providers.scheduler_config import parse_scheduler_limits


class RevealablePasswordField(PasswordField):
    """Password field that keeps the submitted/stored value for masked display."""

    widget = PasswordInput(hide_value=False)


class LoginForm(FlaskForm):
    """Username and password for the administrator login form."""

    username = StringField("Benutzername *", validators=[DataRequired()])
    password = PasswordField("Passwort *", validators=[DataRequired()])


class PasswordChangeForm(FlaskForm):
    """Current and replacement password for the account page."""

    current_password = PasswordField(
        "Aktuelles Passwort *", validators=[DataRequired()]
    )
    new_password = PasswordField(
        "Neues Passwort *", validators=[DataRequired(), Length(min=12)]
    )
    confirm_password = PasswordField(
        "Neues Passwort bestätigen *", validators=[DataRequired(), Length(min=12)]
    )

    def validate_confirm_password(self, field: PasswordField) -> None:
        """Reject a confirmation that does not match the new password."""
        if field.data != self.new_password.data:
            raise ValidationError("Die Passwörter stimmen nicht überein.")


class AzureConnectionForm(FlaskForm):
    """Azure inference connection settings for one tenant profile."""

    base_url = StringField("Azure-Adresse *", validators=[DataRequired()])
    api_key = RevealablePasswordField(
        "API-Schlüssel für Anfragen", validators=[Optional()]
    )
    default_model = SelectField(
        "Standardmodell *",
        validators=[DataRequired(message="Standardmodell ist erforderlich.")],
        choices=[],
        validate_choice=True,
        render_kw={"aria-describedby": "azure-model-hint"},
    )


class OpenAIConnectionForm(FlaskForm):
    """OpenAI inference connection settings for one tenant profile."""

    api_key = RevealablePasswordField(
        "API-Schlüssel für Anfragen", validators=[Optional()]
    )
    default_model = SelectField(
        "Standardmodell *",
        validators=[DataRequired(message="Standardmodell ist erforderlich.")],
        choices=[],
        validate_choice=True,
        render_kw={"aria-describedby": "openai-model-hint"},
    )
    organization = StringField("Organisation (optional)", validators=[Optional()])
    project = StringField("Projekt (optional)", validators=[Optional()])


class OpenRouterConnectionForm(FlaskForm):
    """OpenRouter inference connection settings for one tenant profile."""

    api_key = RevealablePasswordField(
        "API-Schlüssel für Anfragen", validators=[Optional()]
    )
    default_model = SelectField(
        "Standardmodell *",
        validators=[DataRequired(message="Standardmodell ist erforderlich.")],
        choices=[],
        validate_choice=True,
        render_kw={"aria-describedby": "openrouter-model-hint"},
    )


class TenantRoutingForm(FlaskForm):
    """Tenant-wide route selection policy with centralized bounds validation."""

    strategy = SelectField(
        "Routingstrategie",
        choices=[
            ("prioritized", "Priorisierte Kaskade"),
            ("load_balanced", "Load-Balancing gleicher Prioritäten"),
        ],
        validators=[DataRequired()],
    )
    load_balancing_method = SelectField(
        "Load-Balancing-Verfahren",
        choices=[("weighted_least_loaded", "Gewichtete geringste Auslastung")],
        validators=[DataRequired()],
    )
    cost_policy = SelectField(
        "Kostenpolitik",
        choices=[
            ("ignore", "Kosten ignorieren"),
            ("prefer_lower_cost", "Niedrigere vergleichbare Katalogkosten bevorzugen"),
            ("cost_tiers", "Manuelle Kostenstufen verwenden"),
        ],
        validators=[DataRequired()],
    )
    headroom_weight = FloatField(
        "Headroom-Gewichtung", validators=[InputRequired(), NumberRange(min=0, max=1)]
    )
    max_retry_wait_seconds = IntegerField(
        "Maximale Wartezeit (Sekunden)",
        validators=[DataRequired(), NumberRange(min=1, max=300)],
    )
    tie_breaker = SelectField(
        "Gleichstandsregel",
        choices=[("profile_id", "Profilkennung")],
        validators=[DataRequired()],
    )

    def validate(self, extra_validators=None) -> bool:
        """Use the shared routing parser for the complete tenant policy."""
        if not super().validate(extra_validators):
            return False
        try:
            parse_tenant_routing_settings(
                strategy=self.strategy.data,
                load_balancing_method=self.load_balancing_method.data,
                cost_policy=self.cost_policy.data,
                headroom_weight=self.headroom_weight.data,
                max_retry_wait_seconds=self.max_retry_wait_seconds.data,
                tie_breaker=self.tie_breaker.data,
            )
        except ValueError as exc:
            self.strategy.errors.append(str(exc))
            return False
        return True


class ProviderProfileForm(FlaskForm):
    """Provider-specific settings for creating or editing one named account."""

    provider = SelectField(
        "Anbieter *",
        choices=[
            ("azure", "Azure"),
            ("openai", "OpenAI"),
            ("openrouter", "OpenRouter"),
            ("deepseek", "DeepSeek"),
        ],
        validators=[DataRequired()],
    )
    display_name = StringField(
        "Name des Kontos *", validators=[DataRequired(), Length(max=128)]
    )
    base_url = StringField("Azure-Adresse *")
    resume_streams = BooleanField("Azure-Streams bei Verbindungsabbruch fortsetzen")
    default_model = SelectField(
        "Standardmodell",
        validators=[Optional(), Length(max=256)],
        choices=[("", "Nach dem Abruf der Modellliste auswählen")],
        validate_choice=True,
        render_kw={"size": 8},
    )
    api_key = RevealablePasswordField(
        "API-Schlüssel für Anfragen", validators=[Optional()]
    )
    organization = StringField("Organisation (optional)", validators=[Optional()])
    project = StringField("Projekt (optional)", validators=[Optional()])
    routing = TextAreaField("Routing und Kapazität (optionales JSON-Objekt)")
    scheduler_limits = TextAreaField(
        "Scheduler-Limits (optionales JSON-Array)",
    )
    token_reservation_estimate = IntegerField(
        "Token-Reservierungsschätzung (optional)",
        validators=[Optional(), NumberRange(min=1)],
    )

    def validate_routing(self, field: TextAreaField) -> None:
        """Validate and normalize per-profile scheduling parameters."""
        try:
            settings = json.loads(field.data or "{}")
            parse_profile_routing_settings({"routing": settings})
        except (ValueError, TypeError) as exc:
            raise ValidationError(
                "Routing/Kapazität muss ein gültiges JSON-Objekt mit positiven Grenzen sein."
            ) from exc
        field.data = json.dumps(settings, ensure_ascii=False, separators=(",", ":"))

    def validate_scheduler_limits(self, field: TextAreaField) -> None:
        """Validate and normalize scheduler tuples before persistence."""
        try:
            policies = parse_scheduler_limits(field.data or "[]")
        except ValueError as exc:
            raise ValidationError(
                "Scheduler-Limits müssen ein gültiges JSON-Array positiver Limits sein."
            ) from exc
        field.data = json.dumps(policies, ensure_ascii=False, separators=(",", ":"))

    def validate_base_url(self, field: StringField) -> None:
        """Validate Azure endpoints only when Azure is the selected provider."""
        if self.provider.data != "azure":
            return
        try:
            field.data = validate_azure_base_url(field.data)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc


class ActivateProviderForm(FlaskForm):
    """Activate one saved provider profile for proxy traffic."""

    profile_id = StringField(validators=[DataRequired(), Length(max=36)])


class DeactivateProviderForm(FlaskForm):
    """CSRF-protected request to explicitly clear the provider route."""


class SetProviderPriorityForm(ActivateProviderForm):
    """Assign a provider profile to a positive tenant-local priority group."""

    priority = IntegerField(
        "Prioritätsgruppe",
        validators=[DataRequired(), NumberRange(min=1)],
    )


class ReorderProviderForm(ActivateProviderForm):
    """Tenant-owned profile and accessible one-step priority change."""

    direction = SelectField(
        choices=[("up", "Nach oben"), ("down", "Nach unten")],
        validators=[DataRequired()],
    )


class DeleteProviderForm(FlaskForm):
    """CSRF-protected request to remove one provider profile."""

    profile_id = StringField(validators=[DataRequired(), Length(max=36)])


class BillingCredentialsForm(FlaskForm):
    """Profile-scoped optional billing credentials for cost refresh."""

    profile_id = StringField(validators=[DataRequired(), Length(max=36)])
    billing_secret = RevealablePasswordField(
        "Schlüssel für die Abrechnung *", validators=[Optional()]
    )


class OpenAIScopeForm(FlaskForm):
    """Tenant-admin configuration for one OpenAI organization and project."""

    profile_id = StringField(validators=[DataRequired(), Length(max=36)])
    organization_id = StringField(
        "OpenAI-Organisationskennung *",
        validators=[DataRequired(), Length(max=128), Regexp(r"^org-[A-Za-z0-9_-]+$")],
    )
    project_id = StringField(
        "OpenAI-Projektkennung *",
        validators=[DataRequired(), Length(max=128), Regexp(r"^proj_[A-Za-z0-9_-]+$")],
    )
    exclusive_scope_confirmation = BooleanField(
        "Organisation und Projekt sind ausschließlich diesem Konto zugeordnet.",
        validators=[DataRequired(message="Bestätige die Zuordnung dieser Ressource.")],
    )


class OpenAIProjectLookupForm(FlaskForm):
    """Request the active projects visible to one saved OpenAI Admin key."""

    profile_id = StringField(validators=[DataRequired(), Length(max=36)])
    organization_id = StringField(
        "OpenAI-Organisationskennung *",
        validators=[DataRequired(), Length(max=128), Regexp(r"^org-[A-Za-z0-9_-]+$")],
    )


class OpenRouterScopeForm(FlaskForm):
    """Tenant-admin configuration for one OpenRouter workspace."""

    profile_id = StringField(validators=[DataRequired(), Length(max=36)])
    workspace_id = StringField(
        "OpenRouter-Workspace-Kennung *", validators=[DataRequired(), Length(max=36)]
    )
    exclusive_scope_confirmation = BooleanField(
        "Der Workspace ist ausschließlich diesem Konto zugeordnet.",
        validators=[DataRequired(message="Bestätige die Zuordnung dieser Ressource.")],
    )


class AzureScopeForm(FlaskForm):
    """Tenant-confirmed Azure scopes for one provider profile."""

    profile_id = StringField(validators=[DataRequired(), Length(max=36)])
    subscription_id = StringField(
        "Azure-Abonnementkennung *", validators=[DataRequired(), Length(max=36)]
    )
    resource_group_arm_id = StringField(
        "Kennung der Ressourcengruppe (für Kosten) *",
        validators=[DataRequired(), Length(max=1024)],
    )
    cognitive_resource_arm_id = StringField(
        "Kennung des Cognitive-Services-Kontos (für Verbrauchsdaten) *",
        validators=[DataRequired(), Length(max=1024)],
    )
    exclusive_scope_confirmation = BooleanField(
        "Ich bestätige, dass beide Azure-Ressourcen nur diesem Konto zugeordnet sind.",
        validators=[
            DataRequired(
                message=(
                    "Bestätige, dass beide Azure-Ressourcen nur diesem Konto "
                    "zugeordnet sind."
                )
            )
        ],
    )
