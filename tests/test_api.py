"""Integration tests for FastAPI endpoints (model mocked)."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.api.server._model", MagicMock()), \
         patch("src.api.server._tokenizer", MagicMock()), \
         patch("src.engine.inference.generate", return_value={
             "text": "mocked output",
             "prompt_tokens": 10,
             "tokens_generated": 50,
             "total_tokens": 60,
             "latency_ms": 500.0,
             "tokens_per_second": 100.0,
         }):
        from src.api.server import app
        yield TestClient(app, raise_server_exceptions=False)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_generate_endpoint(client):
    response = client.post("/generate", json={
        "prompt": "What is machine learning?",
        "max_new_tokens": 128,
    })
    assert response.status_code in [200, 503]


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
