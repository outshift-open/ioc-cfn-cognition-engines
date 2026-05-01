# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Utils file for the semantic negotiation cognition agent.

Evaluation harnesses import :func:`get_llm_provider` for ad-hoc prompts. The returned
callable must be **awaited**; it wraps ``litellm.acompletion`` so OpenAI-compatible
I/O does not block the asyncio event loop (contrast with ``litellm.completion``).
"""

from typing import Awaitable, Callable, Optional

import litellm

from .settings import settings


def get_llm_provider(model: Optional[str] = None) -> Callable[[str], Awaitable[str]]:
    """Return ``async (prompt: str) -> str`` implemented with ``litellm.acompletion``.

    The model string uses litellm provider/model format, e.g.:
        openai/gpt-4o
        anthropic/claude-sonnet-4-6
        azure/gpt-4o
        ollama/llama3
        bedrock/anthropic.claude-3-sonnet-20240229-v1:0
    """
    _model = model or settings.llm_model

    async def _call(prompt: str) -> str:
        # Async completion keeps uvicorn/FastAPI workers responsive under concurrent requests.
        kwargs: dict = {
            "model": _model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8000,
        }
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        resp = await litellm.acompletion(**kwargs)  # network I/O; never use completion() here
        return resp.choices[0].message.content or ""

    return _call
