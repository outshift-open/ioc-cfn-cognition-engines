#!/usr/bin/env bash
# Create symlinks so the unified gateway can import ingestion, evidence, and caching
# when run locally (PYTHONPATH=. uvicorn gateway.app.main:app ...).
# In Docker, the Dockerfile copies these into /app/ingestion/app, etc.; locally we use symlinks.
set -e
cd "$(dirname "$0")/.."
for name in ingestion evidence caching; do
  case $name in
    ingestion)  target="ingestion-cognitive-agent" ;;
    evidence)   target="evidence-gathering-agent" ;;
    caching)    target="caching-layer" ;;
  esac
  if [[ -d "$target" ]]; then
    if [[ -e "$name" ]]; then
      echo "  $name already exists (symlink or dir), skipping"
    else
      ln -s "$target" "$name"
      echo "  $name -> $target"
    fi
  else
    echo "  WARNING: $target not found, skipping $name" >&2
  fi
done
echo "Done. Run: PYTHONPATH=. poetry run uvicorn gateway.app.main:app --host 0.0.0.0 --port 9004"
