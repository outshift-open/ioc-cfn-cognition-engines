# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Options generation — component 2 of the semantic negotiation pipeline.

For each issue identified by component 1 (intent discovery), this component
generates candidate options that the negotiating agents could agree upon.
It has access to the original context as well as each agent and its memory.

Three strategies:
1. LLM-only: LLM proposes plausible meanings using its own judgment.
2. Memory + LLM: Fetch context/preferences from MAS memory (mocked), then LLM generates options.
3. Agent query: Ask the agents that sent the message for their interpretation (mocked).

Run from /app/agent directory with your venv activated:  python options_generation.py

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
_src_root = _project_root / "app"
for _p in (_project_root, _src_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from config.utils import get_llm_provider

# ---------------------------------------------------------------------------
# Mock: memory lookup (to be replaced with real MAS memory later)
# ---------------------------------------------------------------------------

def mock_memory_lookup(sentence: str, context: Optional[str] = None) -> dict[str, Any]:
    """
    Stub for MAS memory lookup. Returns context and preferences relevant to the sentence.

    Replace with real memory access later (e.g. query agent beliefs, past preferences).
    """
    return {
        "preferences": "User prefers budget-friendly options; has mentioned 'affordable' means under $20 per person in the past.",
        "recent_context": "Previous conversation was about restaurant reservations in downtown.",
        "domain_hints": context or "general",
    }


# ---------------------------------------------------------------------------
# Mock: ask sending agents for their interpretation (to be replaced later)
# ---------------------------------------------------------------------------

def mock_agent_interpretation_query(
    negotiable_entities: list[Any],
    sentence: str,
    context: Optional[str] = None,
    sender_id: Optional[str] = None,
) -> dict[str, list[str]]:
    """
    Stub for querying agents that sent the message for how they interpret each negotiable entity.

    Returns a mapping from term (str) to list of interpretations (one per responding agent).
    Replace with real agent query later.
    """
    # Mock: one or two fake interpretations per term
    result: dict[str, list[str]] = {}
    for t in negotiable_entities:
        term = getattr(t, "term", str(t))
        if term not in result:
            result[term] = []
        if "inexpensive" in term.lower() or "affordable" in term.lower():
            result[term].extend(["under $15 per person", "mid-range; not fine dining"])
        elif "soon" in term.lower():
            result[term].extend(["within the next 2–3 days", "this week"])
        else:
            result[term].extend([f"interpretation of '{term}' from agent (mock)"])
    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InterpretationOption:
    """A single possible meaning/interpretation for an ambiguous term."""
    value: str
    source: str  # "llm" | "memory_llm" | "agent"


@dataclass
class TermOptions:
    """All generated options for one ambiguous term."""
    term: str
    reasoning: str = ""
    options: list[InterpretationOption] = field(default_factory=list)


@dataclass
class OptionsGenerationResult:
    """Result of generating options for all ambiguous terms."""
    sentence: str
    context: Optional[str] = None
    term_options: list[TermOptions] = field(default_factory=list)
    strategy_used: str = ""  # "llm_only" | "memory_llm" | "agent"

    # Return a mapping from term to list of option strings (e.g. {"firewood": ["...", "..."], ...}) as needed by the negotiation protocol
    def options_by_term(self) -> dict[str, list[str]]:
        """Return options as a mapping from term to list of option strings (for negotiation)."""
        out: dict[str, list[str]] = {}
        for to in self.term_options:
            out[to.term] = [str(o.value) for o in to.options]
        return out


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_LLM_ONLY_PROMPT = """Read the context thoroughly. The context contains the premise or mission of the application and the current conversation. From it, infer the kind of negotiation at hand: negotiation may be about (a) different interpretations of a word or phrase (e.g. what "affordable" or "soon" means), or (b) negotiating the amount or quantity of an entity (e.g. how many units, what level). Deduce which applies—or both—based on the context and on each negotiable entity below.

For each negotiable entity, suggest 2–4 concrete options that agents could negotiate over. If the entity is interpretation-heavy, suggest distinct plausible meanings; if it is quantity- or amount-heavy, suggest plausible quantities, levels, or integers (e.g. counts, amounts) as appropriate—infer from context whether options should be numeric. Use the sentence and context to keep options relevant. Output ONLY valid JSON.

Sentence: "{sentence}"
Context: {context}

Negotiable entities (with reasoning from intent discovery):
{terms_blob}

Output format—this exact JSON structure only:
{{
  "options_per_term": [
    {{
      "term": "exact term from the list",
      "options": ["option 1", "option 2", ...]
    }}
  ]
}}

Output (JSON only):"""

_MEMORY_LLM_PROMPT = """Read the context thoroughly; it contains the premise or mission and the current conversation. Infer the kind of negotiation: (a) different interpretations of a word or phrase, or (b) negotiating amount/quantity of an entity—or both. Using the sentence, context, and the following memory/preferences, suggest 2–4 concrete options for negotiation for each negotiable entity. Prefer options that align with stated preferences where relevant. Output ONLY valid JSON.

Sentence: "{sentence}"
Context: {context}

Memory / preferences:
{memory_blob}

Negotiable entities (with reasoning from intent discovery):
{terms_blob}

Output format—this exact JSON structure only:
{{
  "options_per_term": [
    {{
      "term": "exact term from the list",
      "options": ["meaning 1", "meaning 2", ...]
    }}
  ]
}}

Output (JSON only):"""


# ---------------------------------------------------------------------------
# Option generator
# ---------------------------------------------------------------------------

class OptionsGeneration:
    """
    Generates interpretation options for ambiguous terms so agents can negotiate.

    Supports three strategies: LLM-only, memory + LLM, and query sending agents (mocked).
    """

    def __init__(
        self,
        llm_provider: Optional[Callable[[str], str]] = None,
        *,
        memory_lookup: Optional[Callable[..., dict[str, Any]]] = None,
        agent_interpretation_query: Optional[Callable[..., dict[str, list[str]]]] = None,
    ):
        self._llm = llm_provider if llm_provider is not None else get_llm_provider()
        self._memory_lookup = memory_lookup or mock_memory_lookup
        self._agent_query = agent_interpretation_query or mock_agent_interpretation_query

    def generate_options_llm_only(
        self,
        negotiable_entities: list[Any],
        sentence: str,
        context: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """
        Strategy 1: LLM proposes plausible meanings using its own judgment.
        Returns a mapping from term to list of option strings (e.g. {"firewood": ["...", "..."], ...}).
        """
        if not negotiable_entities:
            return {}

        terms_blob = self._format_terms_for_prompt(negotiable_entities)
        context_str = context if context else "not specified"
        prompt = _LLM_ONLY_PROMPT.format(
            sentence=sentence,
            context=context_str,
            terms_blob=terms_blob,
        )
        raw = self._llm(prompt)
        if not isinstance(raw, str):
            raw = str(raw)
        term_options = self._parse_options_response(raw, negotiable_entities, source="llm")
        result = OptionsGenerationResult(
            sentence=sentence,
            context=context,
            term_options=term_options,
            strategy_used="llm_only",
        )
        return result.options_by_term()

    def generate_options_with_memory(
        self,
        negotiable_entities: list[Any],
        sentence: str,
        context: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """
        Strategy 2: Get context/preferences from MAS memory (mocked), pass to LLM to generate options.
        Returns a mapping from term to list of option strings (e.g. {"firewood": ["...", "..."], ...}).
        """
        if not negotiable_entities:
            return {}

        memory_data = self._memory_lookup(sentence, context)
        memory_blob = json.dumps(memory_data, indent=2)
        terms_blob = self._format_terms_for_prompt(negotiable_entities)
        context_str = context if context else "not specified"
        prompt = _MEMORY_LLM_PROMPT.format(
            sentence=sentence,
            context=context_str,
            memory_blob=memory_blob,
            terms_blob=terms_blob,
        )
        raw = self._llm(prompt)
        if not isinstance(raw, str):
            raw = str(raw)
        term_options = self._parse_options_response(raw, negotiable_entities, source="memory_llm")
        result = OptionsGenerationResult(
            sentence=sentence,
            context=context,
            term_options=term_options,
            strategy_used="memory_llm",
        )
        return result.options_by_term()

    def generate_options_from_agents(
        self,
        negotiable_entities: list[Any],
        sentence: str,
        context: Optional[str] = None,
        sender_id: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """
        Strategy 3: Ask the agents that sent the message for their interpretation (mocked).
        Returns a mapping from term to list of option strings (e.g. {"firewood": ["...", "..."], ...}).
        """
        if not negotiable_entities:
            return {}

        agent_interpretations = self._agent_query(negotiable_entities, sentence, context, sender_id)
        term_options: list[TermOptions] = []
        for t in negotiable_entities:
            term = getattr(t, "term", str(t))
            reasoning = getattr(t, "reasoning", "") or ""
            opts = agent_interpretations.get(term) or []
            term_options.append(TermOptions(
                term=term,
                reasoning=reasoning,
                options=[InterpretationOption(value=o, source="agent") for o in opts],
            ))
        result = OptionsGenerationResult(
            sentence=sentence,
            context=context,
            term_options=term_options,
            strategy_used="agent",
        )
        return result.options_by_term()

    def generate_options(self, negotiable_entities: list[Any], sentence: str, context: Optional[str] = None) -> dict[str, list[str]]:
        """
        Generate options for a list of negotiable entities.
        LLM-only strategy is used by default
        Other strategies will be added later
        """
        return self.generate_options_llm_only(negotiable_entities, sentence, context) 

    def _format_terms_for_prompt(self, negotiable_entities: list[Any]) -> str:
        lines = []
        for t in negotiable_entities:
            term = getattr(t, "term", str(t))
            reasoning = getattr(t, "reasoning", "") or ""
            lines.append(f"- \"{term}\": {reasoning}")
        return "\n".join(lines) if lines else "(none)"

    def _parse_options_response(
        self,
        raw: str,
        negotiable_entities: list[Any],
        source: str,
    ) -> list[TermOptions]:
        parsed = self._parse_llm_json(raw)
        term_options: list[TermOptions] = []
        by_term: dict[str, list[str]] = {}
        if parsed:
            for item in parsed.get("options_per_term") or []:
                if isinstance(item, dict) and item.get("term") is not None:
                    t = str(item["term"]).strip()
                    opts = item.get("options") or []
                    by_term[t] = [str(o).strip() for o in opts if o]

        # Preserve order from original negotiable_entities
        for t in negotiable_entities:
            term = getattr(t, "term", str(t))
            reasoning = getattr(t, "reasoning", "") or ""
            opts = by_term.get(term) or []
            term_options.append(TermOptions(
                term=term,
                reasoning=reasoning,
                options=[InterpretationOption(value=o, source=source) for o in opts],
            ))
        return term_options

    def _parse_llm_json(self, raw: str) -> Optional[dict[str, Any]]:
        """Extract a JSON object from LLM output, tolerating markdown and extra text."""
        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.lower().startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
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


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_option_generator() -> None:
    """Run a quick test of all three strategies with mock data."""
    try:
        from agent.intent_discovery import IntentDiscovery
    except ImportError:
        from intent_discovery import IntentDiscovery

    discovery = IntentDiscovery()
    # CaSiNo-style utterance: campsite negotiation over Food, Water, Firewood
    sentence = "I brought way too much raw meat, which means I'll need some firewood to cook it. I didn't bring enough water and I'm going to do a ton of hiking, so I need to stay hydrated. I already have plenty of food, and I can cook more, which is why I prioritize firewood over this."
    context = "Two campsite neighbors negotiate for Food, Water, and Firewood packages, based on their individual preferences and requirements"
    negotiable_entities = discovery.discover(sentence, context=context)
    #negotiable_entities = result.negotiable_entities
    if not negotiable_entities:
        print("No negotiable_entities entities found; using mock list for demo (CaSiNo lexicon).")
        negotiable_entities = ["need", "enough", "prioritize"]

    gen = OptionsGeneration()

    print("Strategy 1: LLM-only")
    out1 = gen.generate_options_llm_only(negotiable_entities, sentence, context)
    for term, options in out1.items():
        print(f"  {term}: {options}")

    print("\nStrategy 2: Memory + LLM")
    out2 = gen.generate_options_with_memory(negotiable_entities, sentence, context)
    for term, options in out2.items():
        print(f"  {term}: {options}")

    print("\nStrategy 3: Agent interpretations (mocked)")
    out3 = gen.generate_options_from_agents(negotiable_entities, sentence, context)
    for term, options in out3.items():
        print(f"  {term}: {options}")


if __name__ == "__main__":
    test_option_generator()

