#!/usr/bin/env python3
"""
Load KCR_sample1.json (or path from KCR_JSON_PATH) into Neo4j via tkf-data-layer.
Run after Neo4j and (optionally) tkf-data-layer are up; or use direct Neo4j + this script.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root (tkf-data-layer) so app is importable
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv(_root / ".env")  # noqa: E402


async def main():
    kcr_path = os.environ.get("KCR_JSON_PATH")
    if not kcr_path:
        # Default: KCR_sample1.json in the same directory as this script
        kcr_path = Path(__file__).parent / "KCR_sample1.json"
    else:
        kcr_path = Path(kcr_path).resolve()
    if not kcr_path.exists():
        print(f"KCR file not found: {kcr_path}")
        sys.exit(1)
    with open(kcr_path) as f:
        data = json.load(f)

    from app.config import get_settings  # noqa: E402
    from app.graph.neo4j_store import Neo4jStore  # noqa: E402
    settings = get_settings()
    store = Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await store.connect()
    try:
        counts = await store.load_kcr(data)
        print(f"Loaded: {counts}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
