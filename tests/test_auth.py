"""Authentication behavior for single-tenant bearer comparison."""

AUTH = {"Authorization": "Bearer test-service-api-key"}


def test_non_ascii_bearer_returns_generic_401(testapp):
    """Non-ASCII invalid tokens must not crash compare_digest with TypeError."""
    response = testapp.get(
        "/v1/models",
        headers={"Authorization": "Bearer café-not-the-key"},
        status=401,
    )

    assert "Authentication with the proxy service failed" in response.text


def test_non_ascii_service_api_key_can_authenticate(app, testapp):
    """Configured non-ASCII SERVICE_API_KEY values still authenticate."""
    app.config["SERVICE_API_KEY"] = "sècret-🔑-key"

    response = testapp.get(
        "/v1/models",
        headers={"Authorization": "Bearer sècret-🔑-key"},
        status=200,
    )

    assert response.json["object"] == "list"


def test_ascii_service_api_key_still_works(testapp):
    """Regression: ASCII keys continue to authenticate."""
    response = testapp.get("/v1/models", headers=AUTH, status=200)

    assert response.json["object"] == "list"
