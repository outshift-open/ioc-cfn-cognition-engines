# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the evidence-gathering agent."""
import pytest
from fastapi.testclient import TestClient

from evidence.app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
