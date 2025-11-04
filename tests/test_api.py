import os
import pytest
from fastapi.testclient import TestClient

from interfaces.api.app import app


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json().get("status") == "ok"

