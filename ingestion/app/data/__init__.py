# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Data access module."""
from .base import DataRepository
from .mock_repo import MockDataRepository

__all__ = ["DataRepository", "MockDataRepository"]

