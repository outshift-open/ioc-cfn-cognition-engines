"""Data access module."""
from .base import DataRepository
from .mock_repo import MockDataRepository

__all__ = ["DataRepository", "MockDataRepository"]

