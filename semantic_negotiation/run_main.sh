#!/usr/bin/env bash
# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

lsof -ti:8089 | xargs kill -9 2>/dev/null; sleep 1; cd /Users/melidris/workspace/projects/ioc/ioc-cfn-cognitive-agents/semantic_negotiation && poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089
