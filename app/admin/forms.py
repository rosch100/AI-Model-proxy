"""WTForms for the tenant administrator UI."""

from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import PasswordField, SelectField, StringField
from wtforms.validators import DataRequired, Length, Optional, ValidationError
from wtforms.widgets import PasswordInput


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

    base_url = StringField("Azure Base-URL *", validators=[DataRequired()])
    api_key = RevealablePasswordField("Azure API-Schlüssel", validators=[Optional()])
    default_model = SelectField(
        "Standardmodell *",
        validators=[DataRequired(message="Standardmodell ist erforderlich.")],
        choices=[],
        validate_choice=True,
        render_kw={"aria-describedby": "azure-model-hint"},
    )


class OpenAIConnectionForm(FlaskForm):
    """OpenAI inference connection settings for one tenant profile."""

    api_key = RevealablePasswordField("OpenAI API-Schlüssel", validators=[Optional()])
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
        "OpenRouter API-Schlüssel", validators=[Optional()]
    )
    default_model = SelectField(
        "Standardmodell *",
        validators=[DataRequired(message="Standardmodell ist erforderlich.")],
        choices=[],
        validate_choice=True,
        render_kw={"aria-describedby": "openrouter-model-hint"},
    )


class ActivateProviderForm(FlaskForm):
    """Activate one saved provider profile for proxy traffic."""

    provider = SelectField(
        "Anbieter *",
        choices=[
            ("azure", "Azure"),
            ("openai", "OpenAI"),
            ("openrouter", "OpenRouter"),
        ],
        validators=[DataRequired()],
    )


class BillingCredentialsForm(FlaskForm):
    """Optional billing credentials for OpenAI or OpenRouter cost refresh."""

    provider = SelectField(
        "Anbieter *",
        choices=[("openai", "OpenAI"), ("openrouter", "OpenRouter")],
        validators=[DataRequired()],
    )
    billing_secret = PasswordField(
        "Provider-Schlüssel mit Billing-Rechten *", validators=[Optional()]
    )
