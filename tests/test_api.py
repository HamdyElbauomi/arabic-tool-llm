from fastapi.testclient import TestClient

from src.api.main import create_app


# --------------------------------------------------
# Fake agents
# --------------------------------------------------

class FakeAgent:

    def run(
        self,
        message: str,
    ) -> dict:

        return {
            "user_message": message,
            "tool_call": {
                "tool": "get_order_status",
                "arguments": {
                    "order_id": 3147,
                },
            },
            "tool_result": {
                "success": True,
                "order_id": 3147,
                "status": (
                    "out_for_delivery"
                ),
            },
        }


class InvalidModelAgent:

    def run(
        self,
        message: str,
    ) -> dict:

        raise ValueError(
            "Model returned invalid JSON"
        )


class BrokenAgent:

    def run(
        self,
        message: str,
    ) -> dict:

        raise RuntimeError(
            "Unexpected failure"
        )


# --------------------------------------------------
# Tests
# --------------------------------------------------

def test_health():

    app = create_app(
        agent_factory=FakeAgent
    )

    with TestClient(app) as client:

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        assert response.json() == {
            "status": "healthy",
            "model_loaded": True,
        }


def test_chat_success():

    app = create_app(
        agent_factory=FakeAgent
    )

    with TestClient(app) as client:

        response = client.post(
            "/chat",
            json={
                "message": (
                    "فين الطلب 3147؟"
                )
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            data["tool_call"]["tool"]
            == "get_order_status"
        )

        assert (
            data["tool_result"]["success"]
            is True
        )


def test_empty_message_rejected():

    app = create_app(
        agent_factory=FakeAgent
    )

    with TestClient(app) as client:

        response = client.post(
            "/chat",
            json={
                "message": "   "
            },
        )

        assert response.status_code == 422


def test_invalid_model_output_returns_422():

    app = create_app(
        agent_factory=InvalidModelAgent
    )

    with TestClient(app) as client:

        response = client.post(
            "/chat",
            json={
                "message": "hello"
            },
        )

        assert response.status_code == 422

        assert response.json() == {
            "detail": (
                "The AI model produced "
                "an invalid tool call."
            )
        }


def test_unexpected_error_returns_500():

    app = create_app(
        agent_factory=BrokenAgent
    )

    with TestClient(app) as client:

        response = client.post(
            "/chat",
            json={
                "message": "hello"
            },
        )

        assert response.status_code == 500

        assert response.json() == {
            "detail": (
                "Internal server error."
            )
        }


def test_request_id_is_returned():

    app = create_app(
        agent_factory=FakeAgent
    )

    with TestClient(app) as client:

        response = client.get(
            "/health",
            headers={
                "X-Request-ID":
                    "test-request-123"
            },
        )

        assert (
            response.headers[
                "X-Request-ID"
            ]
            == "test-request-123"
        )