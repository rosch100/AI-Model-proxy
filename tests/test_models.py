"""Functional tests using WebTest.

See: http://webtest.readthedocs.org/
"""

from app.models import SUPPORTED_MODELS


class TestModels:
    """Models."""

    def test_models_endpoint_returns_401(self, testapp):
        """Ensure /models endpoint returns HTTP 401 without auth."""
        testapp.get("/models", status=401)

    def test_models_endpoint_returns_200(self, testapp):
        """Ensure /models endpoint returns HTTP 200 with auth."""
        response = testapp.get(
            "/models",
            status=200,
            headers={"Authorization": "Bearer test-service-api-key"},
        )
        payload = response.json
        returned_models = [item["id"] for item in payload["data"]]

        assert returned_models == list(SUPPORTED_MODELS)
        assert {
            "gpt-6-astra",
            "gpt-6-luna",
            "gpt-6-sol",
            "gpt-6.1-sol",
        }.issubset(returned_models)
        assert "gpt-6-luna" in returned_models
        assert "gpt-5.5" in returned_models
        assert "gpt-high" not in returned_models
        assert "gpt-medium" not in returned_models
        assert "gpt-low" not in returned_models
        assert "gpt-minimal" not in returned_models

    def test_models_endpoint_lists_only_configured_azure_deployments(self, testapp):
        """Limit Azure's model catalog to deployments configured for this resource."""
        testapp.app.config["AZURE_MODEL_DEPLOYMENTS"] = {
            "gpt-6-luna": "gpt-6-luna-api",
            "gpt-5.6-sol": "gpt-5-6-sol-api",
        }

        response = testapp.get(
            "/v1/models",
            status=200,
            headers={"Authorization": "Bearer test-service-api-key"},
        )

        assert [item["id"] for item in response.json["data"]] == [
            "gpt-6-luna",
            "gpt-5.6-sol",
        ]

    def test_health_endpoint_returns_200(self, testapp):
        """Ensure /health endpoint returns a server-generated request ID."""
        response = testapp.get("/health", status=200)
        request_id = response.headers["X-Proxy-Request-ID"]
        assert len(request_id) == 32
        assert all(character in "0123456789abcdef" for character in request_id)
