"""WTForms for the tenant administrator UI."""

from __future__ import annotations

import json

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    TextAreaField,
)
from wtforms.validators import (
    DataRequired,
    Length,
    NumberRange,
    Optional,
    Regexp,
    ValidationError,
)
from wtforms.widgets import PasswordInput

from app.providers.azure_url import validate_azure_base_url
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
    scheduler_limits = TextAreaField(
        "Scheduler-Limits (optionales JSON-Array)",
    )
    token_reservation_estimate = IntegerField(
        "Token-Reservierungsschätzung (optional)",
        validators=[Optional(), NumberRange(min=1)],
    )

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
