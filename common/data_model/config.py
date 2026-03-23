# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration module for the CFN Cognitive Agents data models."""

import os


class Config:
    """Simple configuration holder for the data-model helpers."""

    def __init__(self) -> None:
        self.service_name = os.environ.get(
            "SERVICE_NAME", "ci-cfn-cognitive-agents-data-model-lib"
        )
        self.application_version = os.environ.get(
            "APPLICATION_VERSION", "0.1.0"
        )
        self.debug = os.environ.get("DEBUG", "false").lower() == "true"

    @property
    def is_production(self) -> bool:
        """Return True when the ENV variable is set to production."""
        return os.environ.get("ENV", "development").lower() == "production"


def get_config() -> "Config":
    """Provide a fresh Config instance."""
    return Config()


config = Config()
