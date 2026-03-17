"""
Telemetry Extraction Service - Core business logic for extracting knowledge from OpenTelemetry data.

This service processes OpenTelemetry (OTEL) trace data and extracts entities and relationships
to build a knowledge graph of system interactions.

Extracted Entities:
- Queries (user query from the root span prompt)
- Agents (from agent_id)
- Services (from ServiceName)
- LLMs (from gen_ai.request.model)
- Tools (from tool_calls attributes)
- Functions (from llm.request.functions.{N}.name)

Extracted Relations:
- Descriptive, context-aware relationship labels generated via LLM
  (e.g. "SENDS_PROMPT_TO", "INVOKES_TOOL", "ORCHESTRATES", "DELEGATES_TASK_TO")
- Falls back to heuristic labels when LLM is unavailable
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base import AdapterSDK
from .prompts import get_concept_prompt, get_relationship_prompt, SUPPORTED_FORMATS
from ..api.schemas import LLMConceptsResult, LLMExtractionResult, LLMRelationshipsResult

logger = logging.getLogger(__name__)

# Try to import Azure OpenAI client
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


class TelemetryExtractionService(AdapterSDK):
    """
    Service for extracting knowledge from OpenTelemetry (OTEL) data.
    """
    
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment: str = "gpt-4o",
        azure_api_version: str = "2024-08-01-preview"
    ):
        super().__init__()
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_deployment = azure_deployment
        self.azure_api_version = azure_api_version
        self._client: Optional[Any] = None
        
    def _get_client(self):
        """Get or create Azure OpenAI client."""
        if self._client is not None:
            return self._client
            
        if not self.azure_endpoint or not self.azure_api_key:
            logger.warning("Azure OpenAI credentials not provided, using basic extraction")
            return None
            
        if AzureOpenAI is None:
            raise ImportError("openai package not installed. Install with: pip install openai")
            
        self._client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=self.azure_api_version
        )
        return self._client
    
    def _load_impl(self) -> Dict[str, Any]:
        """Load implementation - can be overridden for custom data sources."""
        return {"status": "not_implemented", "message": "Use extract_entities_and_relations directly"}
    
    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate deterministic ID from text using MD5 hash."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def extract_entities_and_relations(
        self,
        otel_records: List[Dict[str, Any]],
        request_id: Optional[str] = None,
        format_descriptor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from OTEL data.
        
        Extracts:
        - Concepts: users, agents, tools, LLMs
        - Relations: information flow between concepts
        - Uses Azure OpenAI to generate descriptions and summarize contexts
        
        Args:
            otel_records: List of OpenTelemetry span records
            request_id: Optional client-supplied request id to echo back
            format_descriptor: Data format label (e.g. 'observe-sdk-otel')
            
        Returns:
            Dictionary with knowledge_cognition_request_id, concepts, relations, descriptor, meta
        """
        client = self._get_client()
        descriptor = format_descriptor or "telemetry knowledge extraction"
        
        # Step 0: Filter to Client/Server SpanKinds; records with no SpanKind are kept as-is
        required_span_kinds = {"Client", "Server"}
        otel_records = [
            record for record in otel_records
            if record.get("SpanKind") in required_span_kinds or record.get("SpanKind") is None
        ]
        logger.info(
            "Filtered to %d spans (Client/Server/unset) from %d total",
            len(otel_records), len(otel_records)
        )
        if not otel_records:
            rid = request_id or self._generate_id(f"{datetime.now().isoformat()}_0")
            return {
                "knowledge_cognition_request_id": rid,
                "concepts": [],
                "relations": [],
                "descriptor": descriptor,
                "meta": {
                    "records_processed": 0,
                    "concepts_extracted": 0,
                    "relations_extracted": 0
                }
            }
        
        # Step 1: Build span lookup and identify root spans
        concepts_map = {}  # name -> concept data
        span_lookup = {}   # span_id -> span data
        all_span_ids = set()
        
        for record in otel_records:
            span_id = record.get("SpanId", "")
            span_lookup[span_id] = record
            all_span_ids.add(span_id)
        
        # Root spans: those whose ParentSpanId is empty or not in the current record set
        root_span_ids = set()
        for record in otel_records:
            parent = record.get("ParentSpanId", "")
            if not parent or parent not in all_span_ids:
                root_span_ids.add(record.get("SpanId", ""))
        
        # Step 1a: Extract the main user query only from root spans
        query_concept_name = None
        for record in otel_records:
            if record.get("SpanId", "") not in root_span_ids:
                continue
            span_attrs = record.get("SpanAttributes", {})
            raw_prompt = self._extract_raw_user_prompt(span_attrs)
            if raw_prompt:
                distilled_query = self._distill_user_query(client, raw_prompt)
                short_hash = self._generate_id(raw_prompt)
                query_concept_name = f"query_{short_hash}"
                concepts_map[query_concept_name] = {
                    "name": query_concept_name,
                    "type": "query",
                    "attributes": {},
                    "context": [],
                    "description": distilled_query
                }
                break
        
        # Step 1b: Extract other concepts deterministically
        func_name_pattern = re.compile(r"^llm\.request\.functions\.(\d+)\.name$")
        
        for record in otel_records:
            span_attrs = record.get("SpanAttributes", {})
            
            # Extract agent
            agent_id = span_attrs.get("agent_id")
            if agent_id and agent_id not in concepts_map:
                concepts_map[agent_id] = {
                    "name": agent_id,
                    "type": "agent",
                    "attributes": {},
                    "context": []
                }
            
            # Extract service name as potential agent/system
            service_name = record.get("ServiceName")
            if service_name and service_name not in concepts_map:
                concepts_map[service_name] = {
                    "name": service_name,
                    "type": "service",
                    "attributes": {},
                    "context": []
                }
            
            # Extract LLM model
            model_name = span_attrs.get("gen_ai.request.model") or span_attrs.get("gen_ai.response.model")
            if model_name and model_name not in concepts_map:
                concepts_map[model_name] = {
                    "name": model_name,
                    "type": "llm",
                    "attributes": {},
                    "context": []
                }
            
            # Extract tools from tool calls
            for key in span_attrs:
                if "tool_calls" in key and "name" in key:
                    tool_name = span_attrs.get(key)
                    if tool_name and tool_name not in concepts_map:
                        concepts_map[tool_name] = {
                            "name": tool_name,
                            "type": "tool",
                            "attributes": {},
                            "context": [],
                            "description": ""
                        }
            
            # Extract functions from llm.request.functions.{N}.name and their descriptions
            for key in span_attrs:
                match = func_name_pattern.match(key)
                if match:
                    func_index = match.group(1)
                    func_name = span_attrs[key]
                    desc_key = f"llm.request.functions.{func_index}.description"
                    func_description = span_attrs.get(desc_key, "")
                    
                    if func_name:
                        if func_name not in concepts_map:
                            concepts_map[func_name] = {
                                "name": func_name,
                                "type": "function",
                                "attributes": {},
                                "context": [],
                                "description": func_description
                            }
                        elif func_description and not concepts_map[func_name].get("description"):
                            concepts_map[func_name]["description"] = func_description
            
            # Extract users from message authors
            for key in span_attrs:
                if "author" in key.lower():
                    author = span_attrs.get(key)
                    if author and "user" in author.lower() and author not in concepts_map:
                        concepts_map[author] = {
                            "name": author,
                            "type": "user",
                            "attributes": {},
                            "context": []
                        }
        
        # Step 1c: Extract the final system output from the last span with completion content
        output_concept_name = None
        if query_concept_name:
            # Reuse the same hash suffix from the query name
            query_hash_suffix = query_concept_name[len("query_"):]
            
            # Sort spans by timestamp descending to find the last completion
            sorted_records = sorted(
                otel_records,
                key=lambda r: r.get("Timestamp", ""),
                reverse=True
            )
            for record in sorted_records:
                span_attrs = record.get("SpanAttributes", {})
                raw_completion = self._extract_completion_content(span_attrs)
                if raw_completion:
                    distilled_output = self._distill_system_output(client, raw_completion)
                    output_concept_name = f"output_{query_hash_suffix}"
                    # Track which agent/service produced this output
                    producing_agent = span_attrs.get("agent_id") or record.get("ServiceName")
                    concepts_map[output_concept_name] = {
                        "name": output_concept_name,
                        "type": "output",
                        "attributes": {"produced_by": producing_agent or ""},
                        "context": [],
                        "description": distilled_output
                    }
                    break
        
        # Step 2: Build relations from span hierarchy and attributes
        relations = []
        relation_set = set()  # source||target dedup key
        
        def _add_relation(src: str, tgt: str, ctx: Dict[str, Any]) -> None:
            """Add a relation if both concepts exist and the pair is new."""
            if not src or not tgt or src == tgt:
                return
            if src not in concepts_map or tgt not in concepts_map:
                return
            rel_key = f"{src}||{tgt}"
            if rel_key in relation_set:
                return
            relation_set.add(rel_key)
            src_type = concepts_map[src]["type"]
            tgt_type = concepts_map[tgt]["type"]
            relationship = self._generate_relationship_label(
                client, src, src_type, tgt, tgt_type, ctx
            )
            relations.append({
                "source_name": src,
                "target_name": tgt,
                "relationship": relationship,
                "context": ctx
            })
        
        # 2a: Query -> first agent/service (from the root span that produced the query)
        if query_concept_name:
            for record in otel_records:
                if record.get("SpanId", "") not in root_span_ids:
                    continue
                span_attrs = record.get("SpanAttributes", {})
                agent_id = span_attrs.get("agent_id")
                service_name = record.get("ServiceName")
                if agent_id and agent_id in concepts_map:
                    _add_relation(query_concept_name, agent_id, span_attrs)
                    break
                if service_name and service_name in concepts_map:
                    _add_relation(query_concept_name, service_name, span_attrs)
                    break
        
        # 2a-output: Producing agent -> Output, and Output -> Query (answers)
        if output_concept_name and output_concept_name in concepts_map:
            producing_agent = concepts_map[output_concept_name]["attributes"].get("produced_by", "")
            if producing_agent and producing_agent in concepts_map:
                # Find the span context for the producing agent
                for record in sorted_records:
                    sa = record.get("SpanAttributes", {})
                    if sa.get("agent_id") == producing_agent or record.get("ServiceName") == producing_agent:
                        _add_relation(producing_agent, output_concept_name, sa)
                        break
            if query_concept_name:
                _add_relation(output_concept_name, query_concept_name, {})
        
        for record in otel_records:
            span_attrs = record.get("SpanAttributes", {})
            parent_span_id = record.get("ParentSpanId")
            
            agent_id = span_attrs.get("agent_id")
            service_name = record.get("ServiceName")
            model_name = span_attrs.get("gen_ai.request.model") or span_attrs.get("gen_ai.response.model")
            
            # 2b: Service -> Agent (service hosts the agent)
            if service_name and agent_id and service_name != agent_id:
                _add_relation(service_name, agent_id, span_attrs)
            
            # 2c: Agent/Service -> LLM
            source_name = agent_id or service_name
            if source_name and model_name:
                _add_relation(source_name, model_name, span_attrs)
            
            # 2d: LLM -> Tool (from tool_calls in completions)
            for key in span_attrs:
                if "tool_calls" in key and "name" in key:
                    tool_name = span_attrs.get(key)
                    if tool_name and model_name:
                        _add_relation(model_name, tool_name, span_attrs)
            
            # 2e: Agent -> Function (from llm.request.functions registered on the span)
            for key in span_attrs:
                if func_name_pattern.match(key):
                    func_name = span_attrs[key]
                    if func_name and func_name in concepts_map:
                        if agent_id:
                            _add_relation(agent_id, func_name, span_attrs)
                        elif model_name:
                            _add_relation(model_name, func_name, span_attrs)
            
            # 2f: Parent-child span relations (delegation / orchestration)
            if parent_span_id and parent_span_id in span_lookup:
                parent_record = span_lookup[parent_span_id]
                parent_attrs = parent_record.get("SpanAttributes", {})
                parent_agent = parent_attrs.get("agent_id")
                parent_service = parent_record.get("ServiceName")
                
                parent_name = parent_agent or parent_service
                
                if parent_name and source_name and parent_name != source_name:
                    _add_relation(parent_name, source_name, span_attrs)
        
        # Step 3: Use LLM to generate descriptions and summarize contexts
        # Concepts that already have a telemetry-sourced description are kept as-is.
        if client:
            for name, concept in concepts_map.items():
                if not concept.get("description"):
                    concept["description"] = self._generate_concept_description(
                        client, name, concept["type"], otel_records
                    )
            
            for relation in relations:
                summarized_context = self._summarize_relation_context(
                    client, relation["source_name"], 
                    relation["target_name"], relation["relationship"], 
                    relation["context"]
                )
                relation["summarized_context"] = summarized_context
        else:
            for name, concept in concepts_map.items():
                if not concept.get("description"):
                    concept["description"] = f"{concept['type'].title()}: {name}"
            
            for relation in relations:
                relation["summarized_context"] = f"{relation['relationship']} interaction"
        
        # Step 4: Format output
        concepts = []
        for name, concept in concepts_map.items():
            attributes = concept["attributes"].copy()
            attributes["concept_type"] = concept["type"]
            concepts.append({
                "id": self._generate_id(name),
                "name": name,
                "description": concept.get("description", ""),
                "type": "concept",
                "attributes": attributes
            })
        
        formatted_relations = []
        for relation in relations:
            source_id = self._generate_id(relation["source_name"])
            target_id = self._generate_id(relation["target_name"])
            
            formatted_relations.append({
                "id": self._generate_id(f"{source_id}_{target_id}_{relation['relationship']}"),
                "node_ids": [source_id, target_id],
                "relationship": relation["relationship"],
                "attributes": {
                    "source_name": relation["source_name"],
                    "target_name": relation["target_name"],
                    "summarized_context": relation.get("summarized_context", "")
                }
            })
        
        rid = request_id or self._generate_id(f"{datetime.now().isoformat()}_{len(otel_records)}")
        
        return {
            "knowledge_cognition_request_id": rid,
            "concepts": concepts,
            "relations": formatted_relations,
            "descriptor": descriptor,
            "meta": {
                "records_processed": len(otel_records),
                "concepts_extracted": len(concepts),
                "relations_extracted": len(formatted_relations)
            }
        }
    
    @staticmethod
    def _extract_raw_user_prompt(span_attrs: Dict[str, Any]) -> Optional[str]:
        """
        Extract the raw user prompt content from span attributes.
        
        Scans gen_ai.prompt.{N}.role / gen_ai.prompt.{N}.content pairs and returns
        the content of the first prompt whose role is "user".
        """
        prompt_role_pattern = re.compile(r"^gen_ai\.prompt\.(\d+)\.role$")
        indexed_roles: Dict[str, str] = {}
        for key in span_attrs:
            match = prompt_role_pattern.match(key)
            if match:
                indexed_roles[match.group(1)] = span_attrs[key]
        
        for idx in sorted(indexed_roles.keys(), key=int):
            if indexed_roles[idx].lower() == "user":
                content = span_attrs.get(f"gen_ai.prompt.{idx}.content", "")
                if content and content.strip():
                    return content.strip()
        return None
    
    def _distill_user_query(self, client: Optional[Any], raw_prompt: str) -> str:
        """
        Distill the core question or query from a raw user prompt using the LLM.
        
        The raw prompt may contain lengthy instructions, context, or formatting.
        This method extracts just the user's actual question or query.
        Falls back to returning the raw prompt truncated if no LLM is available.
        """
        if client:
            try:
                prompt = (
                    "Extract ONLY the core user question or query from the text below. "
                    "Ignore any surrounding instructions, system context, formatting, "
                    "or decomposed sub-tasks. Return just the question/query as a single "
                    "concise sentence. If there are multiple questions, return only the "
                    "primary/top-level one.\n\n"
                    f"Text:\n{raw_prompt[:]}\n\n"
                    "Return ONLY the extracted question, nothing else."
                )
                response = client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[
                        {"role": "system", "content": "You extract the core question from text. Return only the question."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                extracted = response.choices[0].message.content.strip()
                if extracted:
                    return extracted
            except Exception as e:
                logger.warning("LLM query distillation failed, using raw prompt: %s", e)
        
        return raw_prompt[:200].strip()
    
    @staticmethod
    def _extract_completion_content(span_attrs: Dict[str, Any]) -> Optional[str]:
        """
        Extract the assistant completion content from span attributes.
        
        Scans gen_ai.completion.{N}.role / gen_ai.completion.{N}.content pairs
        and returns the content of the first completion whose role is "assistant",
        or the first completion content if no roles are present.
        Ignores completions that are only tool calls (no textual answer).
        """
        completion_pattern = re.compile(r"^gen_ai\.completion\.(\d+)\.content$")
        candidates: Dict[str, str] = {}
        for key in span_attrs:
            match = completion_pattern.match(key)
            if match:
                idx = match.group(1)
                content = span_attrs[key]
                if content and isinstance(content, str) and content.strip():
                    candidates[idx] = content.strip()
        
        if not candidates:
            return None
        
        # Prefer completions with role "assistant"
        for idx in sorted(candidates.keys(), key=int):
            role_key = f"gen_ai.completion.{idx}.role"
            role = span_attrs.get(role_key, "").lower()
            if role == "assistant" and candidates[idx]:
                return candidates[idx]
        
        # Fallback: return the first non-empty completion content
        for idx in sorted(candidates.keys(), key=int):
            return candidates[idx]
        return None
    
    def _distill_system_output(self, client: Optional[Any], raw_completion: str) -> str:
        """
        Distill the final answer from raw completion content using the LLM.
        
        The raw completion may contain verbose reasoning or formatting.
        This method extracts the concise final answer.
        """
        if client:
            try:
                prompt = (
                    "Extract ONLY the final answer or conclusion from the text below. "
                    "Ignore intermediate reasoning, chain-of-thought steps, or formatting. "
                    "Return just the concise final answer as stated by the system.\n\n"
                    f"Text:\n{raw_completion[:]}\n\n"
                    "Return ONLY the final answer, nothing else."
                )
                response = client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[
                        {"role": "system", "content": "You extract the final answer from text. Return only the answer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                extracted = response.choices[0].message.content.strip()
                if extracted:
                    return extracted
            except Exception as e:
                logger.warning("LLM output distillation failed, using raw completion: %s", e)
        
        return raw_completion[:].strip()
    
    def _generate_relationship_label(
        self,
        client: Optional[Any],
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        span_attrs: Dict[str, Any]
    ) -> str:
        """
        Generate a descriptive relationship label between two concepts.
        
        Uses the LLM when available to produce a precise, UPPER_SNAKE_CASE verb
        phrase that captures the nature of the interaction. Falls back to
        heuristic labels derived from concept types.
        """
        if client:
            try:
                context_snippet = {
                    k: v for k, v in list(span_attrs.items())[:30]
                    if any(t in k.lower() for t in [
                        "prompt", "content", "message", "tool",
                        "function", "model", "agent", "input", "output"
                    ])
                }
                prompt = (
                    "You are an expert in distributed systems and knowledge graphs.\n"
                    "Given two components in a system trace, produce a single concise "
                    "relationship label in UPPER_SNAKE_CASE (e.g. SENDS_PROMPT_TO, "
                    "INVOKES_TOOL, ORCHESTRATES, DELEGATES_TASK_TO, QUERIES_MODEL, "
                    "EXECUTES_FUNCTION).\n\n"
                    f"Source: {source_name} (type: {source_type})\n"
                    f"Target: {target_name} (type: {target_type})\n\n"
                    f"Span context:\n{json.dumps(context_snippet, indent=2)}\n\n"
                    "Return ONLY the relationship label, nothing else."
                )
                response = client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[
                        {"role": "system", "content": "Return only a single UPPER_SNAKE_CASE relationship label."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                label = response.choices[0].message.content.strip().upper().replace(" ", "_")
                if label:
                    return label
            except Exception as e:
                logger.warning("LLM relationship labelling failed, using heuristic: %s", e)
        
        return self._heuristic_relationship_label(source_type, target_type)
    
    @staticmethod
    def _heuristic_relationship_label(source_type: str, target_type: str) -> str:
        """Derive a descriptive relationship label from concept types."""
        pair = (source_type, target_type)
        heuristics = {
            ("agent", "llm"): "SENDS_PROMPT_TO",
            ("service", "llm"): "SENDS_PROMPT_TO",
            ("llm", "tool"): "INVOKES_TOOL",
            ("llm", "function"): "EXECUTES_FUNCTION",
            ("agent", "agent"): "DELEGATES_TASK_TO",
            ("service", "agent"): "DELEGATES_TASK_TO",
            ("agent", "service"): "DELEGATES_TASK_TO",
            ("service", "service"): "DELEGATES_TASK_TO",
            ("agent", "tool"): "INVOKES_TOOL",
            ("agent", "function"): "EXECUTES_FUNCTION",
            ("query", "agent"): "SUBMITTED_TO",
            ("query", "service"): "SUBMITTED_TO",
            ("agent", "output"): "PRODUCES",
            ("service", "output"): "PRODUCES",
            ("llm", "output"): "GENERATES",
            ("output", "query"): "ANSWERS",
        }
        return heuristics.get(pair, "INTERACTS_WITH")
    
    def _generate_concept_description(
        self,
        client: Any,
        name: str,
        concept_type: str,
        otel_records: List[Dict[str, Any]]
    ) -> str:
        """Generate concept description using Azure OpenAI."""
        try:
            # Collect relevant context for this concept
            context_snippets = []
            for record in otel_records[:100]:  # Truncate to avoid token limits
                span_attrs = record.get("SpanAttributes", {})
                
                # Check if this concept appears in the span
                if (name == span_attrs.get("agent_id") or
                    name == record.get("ServiceName") or
                    name == span_attrs.get("gen_ai.request.model") or
                    name in str(span_attrs)):
                    
                    snippet = {
                        "span_name": record.get("SpanName", ""),
                        "service": record.get("ServiceName", ""),
                        "agent_id": span_attrs.get("agent_id", ""),
                        "gen_ai.prompt.0.role": span_attrs.get("gen_ai.prompt.0.role", ""),
                        "gen_ai.prompt.0.content": span_attrs.get("gen_ai.prompt.0.content", "")
                    }
                    context_snippets.append(snippet)
            
            if not context_snippets:
                return f"{concept_type.title()}: {name}"
            
            # Generate description using LLM
            prompt = f"""Based on the following OpenTelemetry trace data, generate a concise description (2-3 sentences) for this {concept_type}:

Name: {name}
Type: {concept_type}

Context from traces:
{json.dumps(context_snippets[:], indent=2)}

Generate a description that explains what this {concept_type} does in the system."""
            
            response = client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing distributed system traces."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate description for {name}: {str(e)}")
            return f"{concept_type.title()}: {name}"
    
    def _summarize_relation_context(
        self,
        client: Any,
        source_name: str,
        target_name: str,
        relationship: str,
        context: Dict[str, Any]
    ) -> str:
        """Summarize relation context using Azure OpenAI."""
        try:
            # Extract relevant fields from context
            relevant_fields = {}
            for key, value in context.items():
                if any(term in key.lower() for term in ["prompt", "content", "message", "input", "output", "tool", "function"]):
                    relevant_fields[key] = value
            
            if not relevant_fields:
                return f"{source_name} {relationship.lower()} {target_name}"
            
            prompt = f"""Summarize the following interaction between components in a distributed system in a 2-3 sentences describing what is the input and output of the interaction and what information is being exchanged:

Source: {source_name}
Target: {target_name}
Relationship: {relationship}

Context:
{json.dumps(relevant_fields, indent=2)}

Provide a brief summary of what happened in this interaction."""
            
            response = client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing distributed system interactions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to summarize context: {str(e)}")
            return f"{source_name} {relationship.lower()} {target_name}"


class ConceptRelationshipExtractionService(AdapterSDK):
    """
    Service for extracting high-level concepts and relationships from OpenTelemetry
    trace data using LLM-based analysis.

    Unlike TelemetryExtractionService which builds a graph from span hierarchy,
    this service distils traces into a compact JSON payload of important fields
    and delegates concept/relationship identification to the LLM.
    """

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment: str = "gpt-4o",
        azure_api_version: str = "2024-08-01-preview",
    ):
        super().__init__()
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_deployment = azure_deployment
        self.azure_api_version = azure_api_version
        self._client: Optional[Any] = None

    def _get_client(self):
        """Get or create Azure OpenAI client."""
        if self._client is not None:
            return self._client

        if not self.azure_endpoint or not self.azure_api_key:
            logger.warning("Azure OpenAI credentials not provided, LLM extraction unavailable")
            return None

        if AzureOpenAI is None:
            raise ImportError("openai package not installed. Install with: pip install openai")

        self._client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
        )
        return self._client

    def _load_impl(self) -> Dict[str, Any]:
        return {"status": "not_implemented", "message": "Use extract_concepts_and_relationships directly"}

    @staticmethod
    def _generate_id(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Step 1 – Filter spans
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_spans(otel_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep SpanKind 'Client'/'Server' records, plus records with no SpanKind set."""
        required = {"Client", "Server"}
        return [r for r in otel_records if r.get("SpanKind") in required or r.get("SpanKind") is None]

    # ------------------------------------------------------------------
    # Step 2 – Extract important fields from each span
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_important_fields(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For every filtered span, pull out the fields the LLM needs:
        ServiceName, agent_id, system prompt content, function metadata,
        and the user prompt content.
        """
        func_name_pat = re.compile(r"^llm\.request\.functions\.(\d+)\.name$")
        extracted: List[Dict[str, Any]] = []

        for record in records:
            attrs = record.get("SpanAttributes", {})
            entry: Dict[str, Any] = {
                "ServiceName": record.get("ServiceName"),
                "agent_id": attrs.get("agent_id"),
                "model": attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model"),
            }

            # System prompt
            prompt_role_pat = re.compile(r"^gen_ai\.prompt\.(\d+)\.role$")
            for key in attrs:
                m = prompt_role_pat.match(key)
                if m and attrs[key].lower() == "system":
                    idx = m.group(1)
                    entry["system_prompt"] = attrs.get(f"gen_ai.prompt.{idx}.content", "")
                    break

            # User prompt (original query)
            for key in attrs:
                m = prompt_role_pat.match(key)
                if m and attrs[key].lower() == "user":
                    idx = m.group(1)
                    entry["user_prompt"] = attrs.get(f"gen_ai.prompt.{idx}.content", "")
                    break

            # Functions registered on the span
            functions: List[Dict[str, Any]] = []
            for key in attrs:
                fm = func_name_pat.match(key)
                if fm:
                    fi = fm.group(1)
                    functions.append({
                        "name": attrs[key],
                        "description": attrs.get(f"llm.request.functions.{fi}.description", ""),
                        "parameters": attrs.get(f"llm.request.functions.{fi}.parameters", ""),
                    })
            if functions:
                entry["functions"] = functions

            # Tool calls (names only)
            tool_names = []
            for key in attrs:
                if "tool_calls" in key and "name" in key:
                    val = attrs.get(key)
                    if val:
                        tool_names.append(val)
            if tool_names:
                entry["tool_calls"] = tool_names

            # Completion / final output
            completion_pat = re.compile(r"^gen_ai\.completion\.(\d+)\.content$")
            for key in attrs:
                cm = completion_pat.match(key)
                if cm:
                    content = attrs[key]
                    if content and isinstance(content, str) and content.strip():
                        entry["completion"] = content.strip()
                        break

            extracted.append(entry)

        return extracted

    # ------------------------------------------------------------------
    # Step 2b – Extract important fields from OpenClaw records
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_important_fields_openclaw(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pull out relevant fields from OpenClaw records for LLM consumption.

        Each record may be a full openclaw object (with a ``turns`` list) or
        an individual turn dict.  For every turn we emit a flat dict with
        ``userMessage``, ``thinking``, ``toolCalls`` (each stripped to
        ``id``/``name``/``input``/``result``), and ``response``.
        """
        _TOOL_CALL_KEYS = ("id", "name", "input", "result")

        def _flatten_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
            entry: Dict[str, Any] = {}
            for key in ("userMessage", "thinking", "response","timestamp"):
                val = turn.get(key)
                if val is not None:
                    entry[key] = val

            raw_calls = turn.get("toolCalls")
            if raw_calls:
                entry["toolCalls"] = [
                    {k: tc[k] for k in _TOOL_CALL_KEYS if k in tc}
                    for tc in raw_calls
                ]
            return entry

        extracted: List[Dict[str, Any]] = []
        for record in records:
            turns = record.get("turns") if isinstance(record.get("turns"), list) else None
            if turns is not None:
                for turn in turns:
                    flat = _flatten_turn(turn)
                    if flat:
                        extracted.append(flat)
            else:
                flat = _flatten_turn(record)
                if flat:
                    extracted.append(flat)
        return extracted

    # ------------------------------------------------------------------
    # Step 2c – Extract important fields from LoCoMo records
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_important_fields_locomo(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pull out relevant fields from LoCoMo conversation records."""
        extracted: List[Dict[str, Any]] = []
        for record in records:
            entry: Dict[str, Any] = {}
            for key in (
                "speaker", "blip_caption", "dia_id", "text", "query", "session_date_time"
            ):
                val = record.get(key)
                if val is not None:
                    entry[key] = val
            if not entry:
                entry = record
            extracted.append(entry)
        return extracted

    # ------------------------------------------------------------------
    # Format dispatch helpers
    # ------------------------------------------------------------------

    def _filter_records(
        self,
        records: List[Dict[str, Any]],
        data_format: str,
    ) -> List[Dict[str, Any]]:
        """Apply format-specific filtering."""
        if data_format == "observe-sdk-otel":
            return self._filter_spans(records)
        if data_format == "openclaw":
            turns: List[Dict[str, Any]] = []
            for record in records:
                record_turns = record.get("turns")
                if isinstance(record_turns, list):
                    turns.extend(record_turns)
            return turns if turns else records
        return records

    def _build_compact_payload(
        self,
        records: List[Dict[str, Any]],
        data_format: str,
    ) -> List[Dict[str, Any]]:
        """Extract important fields using the format-appropriate method."""
        extractors = {
            "observe-sdk-otel": self._extract_important_fields,
            "openclaw": self._extract_important_fields_openclaw,
            "locomo": self._extract_important_fields_locomo,
        }
        extractor = extractors.get(data_format, self._extract_important_fields)
        return extractor(records)

    # ------------------------------------------------------------------
    # Step 3a – Ask LLM to extract concepts
    # ------------------------------------------------------------------

    def _llm_extract_concepts(
        self,
        client: Any,
        compact_payload: List[Dict[str, Any]],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Send the compact payload to the LLM and return a list of
        concepts, each with a name, type, and detailed description.
        """
        user_msg = json.dumps(compact_payload, indent=2)

        response = client.beta.chat.completions.parse(
            model=self.azure_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format=LLMConceptsResult,
            temperature=0.0,
        )

        parsed: LLMConceptsResult = response.choices[0].message.parsed
        return parsed.model_dump()["concepts"]

    # ------------------------------------------------------------------
    # Step 3b – Ask LLM to extract relationships given concepts + payload
    # ------------------------------------------------------------------

    def _llm_extract_relationships(
        self,
        client: Any,
        concepts: List[Dict[str, Any]],
        compact_payload: List[Dict[str, Any]],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Given the previously extracted concepts and the original
        compact payload, ask the LLM to identify all meaningful
        relationships between concepts.
        """
        user_msg = json.dumps(
            {
                "concepts": concepts,
                "records": compact_payload,
            },
            indent=2,
        )

        response = client.beta.chat.completions.parse(
            model=self.azure_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format=LLMRelationshipsResult,
            temperature=0.0,
        )

        parsed: LLMRelationshipsResult = response.choices[0].message.parsed
        return parsed.model_dump()["relationships"]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract_concepts_and_relationships(
        self,
        records: List[Dict[str, Any]],
        request_id: Optional[str] = None,
        format_descriptor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main pipeline: filter → extract fields → LLM concept/relationship extraction.

        Args:
            records: List of data records (schema depends on format_descriptor)
            request_id: Optional client-supplied request id to echo back
            format_descriptor: Data format label (e.g. 'observe-sdk-otel', 'openclaw', 'locomo')

        Returns a dict matching the knowledge-cognition output schema.
        """
        client = self._get_client()
        data_format = (format_descriptor or "observe-sdk-otel").strip().lower()
        descriptor = data_format

        if data_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported data format: {data_format!r}. Supported: {SUPPORTED_FORMATS}")

        concept_prompt = get_concept_prompt(data_format)
        relationship_prompt = get_relationship_prompt(data_format)

        # Step 1 – format-specific filtering
        filtered = self._filter_records(records, data_format)
        logger.info("Filtered to %d records from %d total (format=%s)", len(filtered), len(records), data_format)

        if not filtered:
            rid = request_id or self._generate_id(f"{datetime.now().isoformat()}_0")
            return {
                "knowledge_cognition_request_id": rid,
                "concepts": [],
                "relations": [],
                "descriptor": descriptor,
                "meta": {
                    "records_processed": 0,
                    "concepts_extracted": 0,
                    "relations_extracted": 0,
                },
            }

        # Step 2 – format-specific compact payload
        compact_payload = self._build_compact_payload(filtered, data_format)
        logger.info("Built compact payload with %d entries (format=%s)", len(compact_payload), data_format)

        # Step 3 – LLM two-stage extraction (requires configured LLM client)
        if not client:
            raise RuntimeError("LLM client is not configured. Provide Azure OpenAI credentials.")

        raw_concepts = self._llm_extract_concepts(client, compact_payload, concept_prompt)
        logger.info("LLM concept extraction returned %d concepts", len(raw_concepts))

        raw_relationships = self._llm_extract_relationships(
            client, raw_concepts, compact_payload, relationship_prompt,
        )
        logger.info("LLM relationship extraction returned %d relationships", len(raw_relationships))

        # Step 4 – format into knowledge-cognition output schema
        # Extract session_time from the last record in the batch, keyed by format
        _session_time_key = {
            "openclaw": "timestamp",
            "locomo": "session_date_time",
        }.get(data_format)
        session_time = ""
        if _session_time_key and compact_payload:
            for rec in reversed(compact_payload):
                val = rec.get(_session_time_key)
                if val:
                    session_time = str(val)
                    break

        concepts = []
        for c in raw_concepts:
            name = c.get("name", "")
            concepts.append({
                "id": self._generate_id(name),
                "name": name,
                "description": c.get("description", ""),
                "type": "concept",
                "attributes": {"concept_type": c.get("type", "unknown")},
            })

        relations = []
        for r in raw_relationships:
            src = r.get("source", "")
            tgt = r.get("target", "")
            rel_label = r.get("relationship", "INTERACTS_WITH")
            source_id = self._generate_id(src)
            target_id = self._generate_id(tgt)
            relations.append({
                "id": self._generate_id(f"{source_id}_{target_id}_{rel_label}"),
                "node_ids": [source_id, target_id],
                "relationship": rel_label,
                "attributes": {
                    "source_name": src,
                    "target_name": tgt,
                    "summarized_context": r.get("description", ""),
                    "session_time": session_time,
                },
            })

        rid = request_id or self._generate_id(f"{datetime.now().isoformat()}_{len(filtered)}")

        return {
            "knowledge_cognition_request_id": rid,
            "concepts": concepts,
            "relations": relations,
            "descriptor": descriptor,
            "meta": {
                "records_processed": len(filtered),
                "concepts_extracted": len(concepts),
                "relations_extracted": len(relations),
            },
        }




