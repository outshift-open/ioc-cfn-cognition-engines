"""
Gateway: single port for ingestion and evidence agents.
Forwards requests by path prefix: /ingestion, /evidence.
Cache is internal only: evidence and ingestion use CACHE_BASE_URL to talk to the caching service
directly on the Docker network; the gateway does not expose /cache.
"""
import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# Backend base URLs (internal hostnames in Docker; override via env for local)
INGESTION_BASE = os.environ.get("INGESTION_BASE_URL", "http://ingestion:8086")
EVIDENCE_BASE = os.environ.get("EVIDENCE_BASE_URL", "http://evidence:8087")

app = FastAPI(
    title="IoC CFN Cognitive Agents Gateway",
    description="Gateway: forwards requests to ingestion, evidence-gathering, and caching agents",
    version="0.1.0",
)


def _forward_headers(request: Request) -> dict:
    """Copy request headers, excluding hop-by-hop and host."""
    skip = {"host", "connection", "transfer-encoding", "keep-alive", "te", "trailer", "upgrade"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


async def _proxy(request: Request, base_url: str, path: str) -> Response:
    """Forward request to backend and return response."""
    path = path.lstrip("/")
    url = f"{base_url.rstrip('/')}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = _forward_headers(request)
    try:
        body = await request.body()
    except Exception:
        body = b""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.request(
                request.method,
                url,
                headers=headers,
                content=body if body else None,
            )
            return Response(
                content=r.content,
                status_code=r.status_code,
                headers={k: v for k, v in r.headers.items() if k.lower() not in {"transfer-encoding", "connection"}},
                media_type=r.headers.get("content-type"),
            )
        except httpx.ConnectError as e:
            return JSONResponse(status_code=503, content={"detail": f"Backend unreachable: {e!s}"})
        except Exception as e:
            return JSONResponse(status_code=502, content={"detail": f"Proxy error: {e!s}"})


@app.get("/health")
async def gateway_health():
    """Gateway health; does not check backends."""
    return {"status": "healthy", "service": "gateway"}


@app.api_route("/ingestion/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_ingestion(request: Request, path: str):
    return await _proxy(request, INGESTION_BASE, path)


@app.api_route("/evidence/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_evidence(request: Request, path: str):
    return await _proxy(request, EVIDENCE_BASE, path)


@app.get("/")
async def root():
    return {
        "message": "IoC CFN Cognitive Agents Gateway",
        "routes": {
            "ingestion": "/ingestion/ (e.g. /ingestion/health, /ingestion/api/v1/...)",
            "evidence": "/evidence/ (e.g. /evidence/health, /evidence/api/knowledge-mgmt/reasoning/evidence)",
        },
        "note": "Cache is internal only; not exposed via gateway.",
    }
