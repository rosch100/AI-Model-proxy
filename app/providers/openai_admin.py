"""Read-only OpenAI Administration API operations used by tenant settings."""

from __future__ import annotations

import json
import re
from time import monotonic

import requests
from gevent import Timeout

OPENAI_PROJECTS_URL = "https://api.openai.com/v1/organization/projects"
OPENAI_PROJECT_PAGE_SIZE = 100
OPENAI_PROJECT_MAX_PAGES = 5
OPENAI_PROJECT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
OPENAI_PROJECT_REQUEST_TIMEOUT = (3, 15)
OPENAI_PROJECT_REQUEST_BUDGET_SECONDS = 30


class OpenAIProjectLookupError(Exception):
    """Raised when the admin key cannot safely return a complete project list."""


def list_openai_projects(api_key: str, organization_id: str) -> list[tuple[str, str]]:
    """Return active organization projects as ``(id, display name)`` pairs."""
    deadline = monotonic() + OPENAI_PROJECT_REQUEST_BUDGET_SECONDS
    projects: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    after: str | None = None

    for _ in range(OPENAI_PROJECT_MAX_PAGES):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise OpenAIProjectLookupError(
                "OpenAI-Projektliste: Zeitlimit überschritten."
            )
        params: dict[str, str | int] = {"limit": OPENAI_PROJECT_PAGE_SIZE}
        if after is not None:
            params["after"] = after
        response = None
        try:
            with Timeout(max(remaining, 0.001)):
                response = requests.get(
                    OPENAI_PROJECTS_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "OpenAI-Organization": organization_id,
                    },
                    params=params,
                    timeout=(
                        min(OPENAI_PROJECT_REQUEST_TIMEOUT[0], remaining),
                        min(OPENAI_PROJECT_REQUEST_TIMEOUT[1], remaining),
                    ),
                    stream=True,
                )
                if response.status_code >= 400:
                    raise OpenAIProjectLookupError(
                        f"OpenAI-Projektliste antwortet mit HTTP {response.status_code}. "
                        "Prüfe Admin-Key und Organization Administration Read."
                    )
                payload = _read_project_page(response, deadline)
        except Timeout as exc:
            raise OpenAIProjectLookupError(
                "OpenAI-Projektliste: Zeitlimit überschritten."
            ) from exc
        except requests.RequestException as exc:
            raise OpenAIProjectLookupError(
                "OpenAI-Projektliste konnte nicht vollständig gelesen werden."
            ) from exc
        finally:
            if response is not None:
                response.close()

        data = payload.get("data")
        has_more = payload.get("has_more")
        if not isinstance(data, list) or not isinstance(has_more, bool):
            raise OpenAIProjectLookupError(
                "OpenAI-Projektliste hat ein ungültiges Format."
            )

        for project in data:
            if not isinstance(project, dict):
                raise OpenAIProjectLookupError(
                    "OpenAI-Projektliste enthält einen ungültigen Eintrag."
                )
            project_id = project.get("id")
            name = project.get("name")
            if (
                not isinstance(project_id, str)
                or re.fullmatch(r"proj_[A-Za-z0-9_-]+", project_id) is None
                or (name is not None and not isinstance(name, str))
                or project_id in seen_ids
            ):
                raise OpenAIProjectLookupError(
                    "OpenAI-Projektliste enthält ungültige Projektdaten."
                )
            seen_ids.add(project_id)
            projects.append((project_id, name or project_id))

        if not has_more:
            return projects
        after = payload.get("last_id")
        if not isinstance(after, str) or after not in seen_ids:
            raise OpenAIProjectLookupError(
                "OpenAI-Projektliste enthält keinen gültigen Seiten-Cursor."
            )

    raise OpenAIProjectLookupError(
        "OpenAI-Projektliste ist größer als das unterstützte Abruflimit."
    )


def _read_project_page(response, deadline: float) -> dict[str, object]:
    response_body = bytearray()
    for chunk in response.iter_content(chunk_size=1):
        if monotonic() > deadline:
            raise OpenAIProjectLookupError(
                "OpenAI-Projektliste: Zeitlimit überschritten."
            )
        if len(response_body) + len(chunk) > OPENAI_PROJECT_MAX_RESPONSE_BYTES:
            raise OpenAIProjectLookupError("OpenAI-Projektliste ist zu groß.")
        response_body.extend(chunk)
    try:
        payload = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OpenAIProjectLookupError(
            "OpenAI-Projektliste hat ein ungültiges Format."
        ) from exc
    if not isinstance(payload, dict):
        raise OpenAIProjectLookupError("OpenAI-Projektliste hat ein ungültiges Format.")
    return payload
