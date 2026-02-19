#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
uvicorn evidence-gathering-agent.app.main:app --reload --port 8087

