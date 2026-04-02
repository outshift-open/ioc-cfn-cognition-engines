# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Intent discovery — component 1 of the semantic negotiation pipeline.

Given a context (e.g. a conversation transcript or a structured state object)
this component is responsible for identifying the negotiable issues that exist
between the parties and returning them as a list of issue identifiers.

Run from project root with your venv activated:  python src/intent_discovery_agent.py

"""

from __future__ import annotations
from pathlib import Path
import json
import os
import sys

# ``.../semantic_negotiation/app/agent/this_file.py`` → parent of package ``app``
_semantic_negotiation_root = Path(__file__).resolve().parents[2]
if str(_semantic_negotiation_root) not in sys.path:
    sys.path.insert(0, str(_semantic_negotiation_root))

# Ensure project root is on path when running as script
_project_root = Path(__file__).resolve().parents[1]
_src_root = _project_root / "src"
for _p in (_project_root, _src_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import httpx

from app.agent.http_repo import (
    post_shared_memories_query,
    shared_memories_query_path,
)
from app.config.utils import get_llm_provider


def _format_agent_line_for_intent(agent_names: Optional[List[str]]) -> str:
    if agent_names:
        if isinstance(agent_names, (list, tuple, set)):
            return ", ".join(str(n) for n in agent_names)
        return str(agent_names)
    return "(not specified)"


def build_intent_discovery_shared_memory_intent(
    mission: str,
    agent_names: Optional[List[str]] = None,
) -> str:
    """
    Natural-language intent for shared-memories query **before** negotiable issues exist.

    Uses only the mission text and agent display names (no issue list). Options
    generation uses per-issue intents from :mod:`app.agent.options_generation` instead.
    """
    m = (mission or "").strip() or "(not specified)"
    agents_line = _format_agent_line_for_intent(agent_names)
    agents_q = (
        agents_line
        if agents_line != "(not specified)"
        else "the negotiating agents (agent names not provided)"
    )
    return (
        "Evidence gathering query (intent discovery — before specific issues are identified).\n\n"
        f"Mission: {m}\n"
        f"Agents: {agents_line}\n\n"
        "Please answer the following:\n"
        f"1. What preferences, constraints, or priorities do {agents_q} have that are relevant to this mission?\n"
        "2. From prior negotiation history involving these agents (or this workspace), what themes, trade-offs, "
        "or recurring issues appear that may inform what could be negotiated next?\n"
        "3. What stored memory (facts, commitments, or context) should be surfaced to help discover negotiable "
        "issues for this mission involving these agents?"
    )


@dataclass
class NegotiableEntity:
    """A negotiable entity (term or phrase) identified from the sentence. Used internally when parsing LLM output."""
    term: str
    reasoning: str
    span: Optional[tuple[int, int]] = None


@dataclass
class IntentDiscoveryResult:
    """Structured result of intent/entity extraction."""
    sentence: str
    context: Optional[str] = None
    negotiable_entities: list[str] = field(default_factory=list)
    raw_llm_response: Optional[str] = None


def fetch_shared_memory_for_intent_discovery(
    fabric_node_base_url: str,
    workspace_id: str,
    mas_id: str,
    sentence: str,
    *,
    agent_names: Optional[List[str]] = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    Query the cognition-fabric shared-memories **query** endpoint (evidence) for the
    mission *sentence* and *agent_names* only (no negotiable-entity list).

    Builds ``intent`` via :func:`build_intent_discovery_shared_memory_intent` and performs
    **one** :func:`~app.agent.http_repo.post_shared_memories_query`.

    Returns a dict with ``evidence_message``, ``evidence_response_id``, and ``source``.
    On HTTP 404, raises :class:`~app.agent.http_repo.SharedMemoryNotFoundError`.

    Not invoked by :class:`IntentDiscovery`; wire callers when intent discovery
    should be informed by shared memory.
    """
    intent = build_intent_discovery_shared_memory_intent(sentence, agent_names)
    base = fabric_node_base_url.rstrip("/")
    path = shared_memories_query_path(workspace_id, mas_id)
    with httpx.Client(base_url=base, timeout=timeout) as client:
        data = post_shared_memories_query(client, path, intent)
    return {
        "evidence_message": data.get("message"),
        "evidence_response_id": data.get("response_id"),
        "source": "fabric_node_shared_memories_query",
    }


# Default prompt for extraction (instructs LLM to return JSON only)
_EXTRACT_PROMPT = """Task: You are the issue identifier facilitating potential negotiations in a multi-agent application. Your job is to identify issues—i.e., terms or entities that could need negotiation between agents.

Read the context thoroughly. The context contains the mission or premise of the application and the current conversation. Use that perspective to decide what counts as a negotiable issue: only flag terms or entities that, in this mission and conversation, could reasonably need negotiation between agents.

Then read the sentence and identify all such issues (negotiable entities). Negotiable entities are words or phrases that represent parameters or items that can be negotiated, prioritized, traded off, or adjusted between agents. These include:
- Concrete items or resources that the user mentions as needs, preferences, or priorities (e.g. quantities, types, or relative importance).
- Ambiguous terms: words whose meaning can vary by context or person.


List each distinct issue (negotiable entity) exactly as it appears in the sentence (or a minimal clear phrase). For each, provide brief reasoning for why it could need negotiation between agents in the given context.

Output format—this exact JSON structure only:
{{
  "negotiable_entities": [
    {{
      "term": "exact phrase from sentence",
      "reasoning": "brief explanation of why this could need negotiation between agents in the given context"
    }}
  ]
}}

Sentence: "{sentence}"
Context: {context}

Output (JSON only):"""


class IntentDiscovery:
    """
    Extracts entities and context-dependent terms from a sentence using an LLM.

    Use this to discover what negotiable entities appear in user input
    so downstream logic can resolve meanings (e.g. for search, routing, or clarification).
    """

    def __init__(
        self,
        llm_provider: Optional[Callable[[str], str]] = None,
        *,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the intent discovery agent.

        Args:
            llm_provider: Function that takes a prompt string and returns the LLM response.
                          If None, uses get_llm_provider() from vanilla_react_agent.
            prompt_template: Optional custom prompt. Must contain {sentence} and {context}.
        """
        self._llm = llm_provider if llm_provider is not None else get_llm_provider()
        self._prompt_template = prompt_template or _EXTRACT_PROMPT

    def discover(
        self,
        sentence: str,
        context: Optional[str] = None,
        *,
        return_raw: bool = False,
        agent_names: Optional[List[str]] = None,
        fabric_node_base_url: Optional[str] = None,
        workspace_id: Optional[str] = None,
        mas_id: Optional[str] = None,
    ) -> IntentDiscoveryResult:
        """
        Extract negotiable entities from a sentence.

        Args:
            sentence: The input sentence or phrase.
            context: Optional domain/situation (e.g. "banking", "travel booking")
                     to help disambiguate terms.
            return_raw: If True, populate ``raw_llm_response`` on the result.

        Returns:
            :class:`IntentDiscoveryResult` with ``negotiable_entities`` (issue id strings).
            Callers should read ``result.negotiable_entities`` — the return type is always
            this dataclass (not a bare ``list``).
        """
        context_str = context if context else "not specified"
        prompt = self._prompt_template.format(
            sentence=sentence,
            context=context_str,
        )
        raw = self._llm(prompt)
        if not isinstance(raw, str):
            raw = str(raw)

        entities: list[str] = []

        # Parse JSON from response (allow markdown code blocks and trailing text)
        parsed = self._parse_llm_json(raw)
        if parsed:
            for a in parsed.get("negotiable_entities") or []:
                if isinstance(a, dict) and a.get("term"):
                    entities.append(str(a["term"]).strip())
                elif isinstance(a, str) and a.strip():
                    # Many models return ["price", "delivery"] instead of [{"term": "price"}, …]
                    entities.append(a.strip())

        return IntentDiscoveryResult(
            sentence=sentence,
            context=context,
            negotiable_entities=entities,
            raw_llm_response=raw if return_raw else None,
        )

    def _parse_llm_json(self, raw: str) -> Optional[dict[str, Any]]:
        """Extract a JSON object from LLM output, tolerating markdown and extra text."""
        raw = raw.strip()
        # Strip markdown code blocks (```json ... ``` or ``` ... ```)
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.lower().startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        # Find first complete { ... } object
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        end = -1
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None


def test_intent_discovery() -> None:
    """Demonstrate IntentDiscovery with a few sentences."""
    discovery = IntentDiscovery()
    examples = [
        ("I need a loan for my business.", "banking"),
        ("We want to book a table by the window.", "restaurant"),
        ("Book a reservation at an inexpensive restaurant for 6 people next Friday.", "Restaurant"),
        ("I need a loan of $56000 for my business, what is the grace period?", "banking"),
        ("We need addition food to sustain our camping trip. We would like to use a little more firewood, but it isn't as important. We would like to have a little additional water, however we prefer food.", ""),
        ("to stay hydrated, I will need more water because I need to stay hydrated, If I don't I will faint. because I am diabetic, I need to eat small many meals. to cook and stay warm.", ""),
        ("I brought way too much raw meat, which means I'll need some firewood to cook it. I didn't bring enough water and I'm going to do a ton of hiking, so I need to stay hydrated. I already have plenty of food, and I can cook more, which is why I prioritize firewood over this.", ""),
    ]

    for sentence, context in examples:
        result = discovery.discover(sentence, context=context, return_raw=True)
        print(f"Sentence: {sentence}")
        print(f"Context: {context}")
        print(f"  Negotiable entities: {result.negotiable_entities}")
        print()


if __name__ == "__main__":
    test_intent_discovery()

