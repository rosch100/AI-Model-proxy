"""Validation for Azure OpenAI and Cognitive Services endpoint URLs."""

from __future__ import annotations

from urllib.parse import urlsplit

AZURE_HOST_SUFFIXES = (
    ".openai.azure.com",
    ".openai.azure.us",
    ".openai.azure.cn",
    ".cognitiveservices.azure.com",
    ".cognitiveservices.azure.us",
    ".cognitiveservices.azure.cn",
)


def validate_azure_base_url(value: object) -> str:
    """Normalize an HTTPS Azure endpoint and reject unsafe or unrelated URLs."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Azure Base-URL ist erforderlich.")

    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError("Azure Base-URL ist ungültig.") from exc

    if (
        parsed.scheme != "https"
        or hostname is None
        or not any(hostname.endswith(suffix) for suffix in AZURE_HOST_SUFFIXES)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or "?" in normalized
        or "#" in normalized
    ):
        raise ValueError(
            "Azure Base-URL muss eine HTTPS-Adresse eines Azure OpenAI- oder "
            "Cognitive-Services-Endpunkts ohne Zugangsdaten, Query oder Fragment sein."
        )
    return normalized
