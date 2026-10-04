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
        raise ValueError("Die Azure-Adresse fehlt.")

    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError("Die Azure-Adresse ist ungültig.") from exc

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
            "Die Azure-Adresse muss mit HTTPS beginnen und zu einer Azure OpenAI- "
            "oder Cognitive-Services-Ressource gehören. Ergänze keine Zugangsdaten "
            "oder weiteren URL-Angaben."
        )
    return normalized
