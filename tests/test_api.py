import json
import httpx
import pytest
from fastapi.testclient import TestClient

import config
import database
from main import app


COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


# fake llama-server
def fake_model_server(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"object": "list", "data": [{"id": "test-model"}]})
    if request.url.path == "/v1/chat/completions":
        body = json.loads(request.content)
        if body.get("max_tokens") == -42:
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "h"}}]},
                {"choices": [{"delta": {"content": "i"}}]},
                {"choices": [], "usage": COMPLETION["usage"]},
            ]
            text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=COMPLETION)
    return httpx.Response(404)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        app.state.http_client = httpx.AsyncClient(
            base_url="http://model", transport=httpx.MockTransport(fake_model_server)
        )
        yield test_client


@pytest.fixture
def auth():
    key = database.create_api_key("tester")
    return {"Authorization": f"Bearer {key}"}


MESSAGES = {"messages": [{"role": "user", "content": "hello"}]}


def usage_rows():
    conn = database.get_db_connection()
    rows = conn.execute(
        "SELECT status_code, prompt_tokens, completion_tokens, total_tokens FROM usage_logs"
    ).fetchall()
    conn.close()
    return rows


def test_health(client):
    response = client.get("/health")
    assert response.json() == {"status": "okay", "model_server": "okay"}


def test_requires_api_key(client):
    assert client.post("/v1/chat/completions", json=MESSAGES).status_code in (401, 403)
    bad = {"Authorization": "Bearer sk_local_nope"}
    assert client.post("/v1/chat/completions", json=MESSAGES, headers=bad).status_code == 401


def test_disabled_key_rejected(client, auth):
    key_id = database.list_api_keys()[0][0]
    assert database.disable_api_key(key_id) is True
    assert database.disable_api_key(key_id) is False
    assert client.post("/v1/chat/completions", json=MESSAGES, headers=auth).status_code == 401


def test_chat_completion_logs_usage(client, auth):
    response = client.post("/v1/chat/completions", json=MESSAGES, headers=auth)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi"
    assert usage_rows() == [(200, 5, 2, 7)]


def test_streaming_logs_usage(client, auth):
    with client.stream(
        "POST", "/v1/chat/completions", json={**MESSAGES, "stream": True}, headers=auth
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "data: [DONE]" in body
    assert '"content": "h"' in body
    assert usage_rows() == [(200, 5, 2, 7)]


def test_upstream_400_passed_through(client, auth):
    response = client.post(
        "/v1/chat/completions", json={**MESSAGES, "max_tokens": -42}, headers=auth
    )
    assert response.status_code == 400
    assert usage_rows() == [(400, 0, 0, 0)]


def test_model_server_down(client, auth):
    def down(request):
        raise httpx.ConnectError("refused")

    app.state.http_client = httpx.AsyncClient(
        base_url="http://model", transport=httpx.MockTransport(down)
    )
    response = client.post("/v1/chat/completions", json=MESSAGES, headers=auth)
    assert response.status_code == 503
    assert usage_rows() == [(503, 0, 0, 0)]
    assert client.get("/health").json()["model_server"] == "unavailable"


def test_list_models(client, auth):
    response = client.get("/v1/models", headers=auth)
    assert response.json()["data"][0]["id"] == "test-model"


def test_usage_summary(client, auth):
    client.post("/v1/chat/completions", json=MESSAGES, headers=auth)
    client.post("/v1/chat/completions", json=MESSAGES, headers=auth)
    (row,) = database.get_usage_summary()
    assert row[1:7] == ("tester", 2, 0, 10, 4, 14)
