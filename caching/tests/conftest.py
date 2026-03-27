# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the caching layer."""
import pytest

from caching.app.agent.caching_layer import CachingLayer


@pytest.fixture
def caching_layer() -> CachingLayer:
    """Provide a small CachingLayer instance suitable for unit tests."""
    return CachingLayer(vector_dimension=8)
