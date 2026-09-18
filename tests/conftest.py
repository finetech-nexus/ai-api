"""pytest configuration."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AI_API_SKIP_ML", "1")
os.environ.setdefault("ENV", "test")

from api.api import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
