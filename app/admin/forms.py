"""WTForms for the tenant administrator UI."""

from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import PasswordField, SelectField, StringField, TextAreaField
from wtforms.validators import DataRequired, Length, Optional, ValidationError


class LoginForm(FlaskForm):
    """Username and password for the administrator login page."""

    username = StringField("Benutzername", validators=[DataRequired()])
    password = PasswordField("Passwort", validators=[DataRequired()])


class PasswordChangeForm(FlaskForm):
    """Current and replacement password for the account page."""

    current_password = PasswordField("Aktuelles Passwort", validators=[DataRequired()])
    new_password = PasswordField(
        "Neues Passwort", validators=[DataRequired(), Length(min=12)]
    )
    confirm_password = PasswordField(
        "Neues Passwort bestätigen", validators=[DataRequired(), Length(min=12)]
    )

    def validate_confirm_password(self, field: PasswordField) -> None:
        """Reject a confirmation that does not match the new password."""
        if field.data != self.new_password.data:
            raise ValidationError("Die Passwörter stimmen nicht überein.")


class AzureConnectionForm(FlaskForm):
    """Azure inference connection settings for one tenant profile."""

    base_url = StringField("Azure Base-URL", validators=[DataRequired()])
    api_key = PasswordField("Azure API-Schlüssel", validators=[Optional()])
    default_model = StringField("Standardmodell", validators=[DataRequired()])
    model_deployments = TextAreaField(
        "Modell-Deployments (JSON)", validators=[DataRequired()]
    )


class OpenAIConnectionForm(FlaskForm):
    """OpenAI inference connection settings for one tenant profile."""

    api_key = PasswordField("OpenAI API-Schlüssel", validators=[Optional()])
    default_model = StringField("Standardmodell", validators=[DataRequired()])
    organization = StringField("Organisation", validators=[Optional()])
    project = StringField("Projekt", validators=[Optional()])


class OpenRouterConnectionForm(FlaskForm):
    """OpenRouter inference connection settings for one tenant profile."""

    api_key = PasswordField("OpenRouter API-Schlüssel", validators=[Optional()])
    default_model = StringField("Standardmodell", validators=[DataRequired()])


class ActivateProviderForm(FlaskForm):
    """Activate one saved provider profile for proxy traffic."""

    provider = SelectField(
        "Anbieter",
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
        "Anbieter",
        choices=[("openai", "OpenAI"), ("openrouter", "OpenRouter")],
        validators=[DataRequired()],
    )
    billing_secret = PasswordField("Billing-Schlüssel", validators=[Optional()])
