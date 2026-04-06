#!/usr/bin/env bash
# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export PYTHONUNBUFFERED=1
uvicorn evidence.app.main:app --reload --port 8087
