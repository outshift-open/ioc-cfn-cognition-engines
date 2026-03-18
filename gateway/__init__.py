"""IOC CFN Cognitive Agents - Gateway

Unified FastAPI gateway that mounts ingestion and evidence services with shared cache.
"""

__version__ = "0.1.0"

# Export registration client functions for package users
from gateway.app.client import (
    register_knowledge_management_engine,
    register_semantic_negotiation_engine,
    register_both_engines,
)

__all__ = [
    "register_knowledge_management_engine",
    "register_semantic_negotiation_engine",
    "register_both_engines",
]
