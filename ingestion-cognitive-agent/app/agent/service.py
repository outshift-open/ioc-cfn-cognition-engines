"""
Telemetry Extraction Service - Core business logic for extracting knowledge from OpenTelemetry data.

This service processes OpenTelemetry (OTEL) trace data and extracts entities and relationships
to build a knowledge graph of system interactions.

Extracted Entities:
- Agents (from agent_id)
- Services (from ServiceName)
- LLMs (from gen_ai.request.model)
- Tools (from tool_calls attributes)

Extracted Relations:
- USES: Agent -> LLM
- CALLS: LLM -> Tool
- COORDINATES: Parent Agent -> Child Agent (from span hierarchy)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base import AdapterSDK

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
        otel_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from OTEL data.
        
        Extracts:
        - Concepts: users, agents, tools, LLMs
        - Relations: information flow between concepts
        - Uses Azure OpenAI to generate descriptions and summarize contexts
        
        Args:
            otel_records: List of OpenTelemetry span records
            
        Returns:
            Dictionary with knowledge_cognition_request_id, concepts, relations, descriptor, meta
        """
        client = self._get_client()
        
        # Step 1: Extract concepts deterministically
        concepts_map = {}  # name -> concept data
        span_lookup = {}   # span_id -> span data
        
        for record in otel_records:
            span_id = record.get("SpanId", "")
            span_lookup[span_id] = record
            
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
                            "context": []
                        }
            
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
        
        # Step 2: Build relations from span hierarchy and attributes
        relations = []
        relation_set = set()  # to avoid duplicates
        
        for record in otel_records:
            span_attrs = record.get("SpanAttributes", {})
            parent_span_id = record.get("ParentSpanId")
            
            # Get source concept (current span)
            agent_id = span_attrs.get("agent_id")
            service_name = record.get("ServiceName")
            model_name = span_attrs.get("gen_ai.request.model") or span_attrs.get("gen_ai.response.model")
            
            source_name = agent_id or service_name
            
            # Agent -> LLM relation
            if source_name and model_name and source_name in concepts_map and model_name in concepts_map:
                rel_key = f"{source_name}||USES||{model_name}"
                if rel_key not in relation_set:
                    relation_set.add(rel_key)
                    relations.append({
                        "source_name": source_name,
                        "target_name": model_name,
                        "relationship": "USES",
                        "context": span_attrs
                    })
            
            # LLM -> Tool relations
            for key in span_attrs:
                if "tool_calls" in key and "name" in key:
                    tool_name = span_attrs.get(key)
                    if tool_name and model_name and model_name in concepts_map and tool_name in concepts_map:
                        rel_key = f"{model_name}||CALLS||{tool_name}"
                        if rel_key not in relation_set:
                            relation_set.add(rel_key)
                            relations.append({
                                "source_name": model_name,
                                "target_name": tool_name,
                                "relationship": "CALLS",
                                "context": span_attrs
                            })
            
            # Parent-child span relations
            if parent_span_id and parent_span_id in span_lookup:
                parent_record = span_lookup[parent_span_id]
                parent_attrs = parent_record.get("SpanAttributes", {})
                parent_agent = parent_attrs.get("agent_id")
                parent_service = parent_record.get("ServiceName")
                
                parent_name = parent_agent or parent_service
                
                if parent_name and source_name and parent_name != source_name:
                    if parent_name in concepts_map and source_name in concepts_map:
                        rel_key = f"{parent_name}||COORDINATES||{source_name}"
                        if rel_key not in relation_set:
                            relation_set.add(rel_key)
                            relations.append({
                                "source_name": parent_name,
                                "target_name": source_name,
                                "relationship": "COORDINATES",
                                "context": span_attrs
                            })
        
        # Step 3: Use LLM to generate descriptions and summarize contexts
        if client:
            # Generate descriptions for concepts
            for name, concept in concepts_map.items():
                description = self._generate_concept_description(
                    client, name, concept["type"], otel_records
                )
                concept["description"] = description
            
            # Summarize contexts for relations
            for relation in relations:
                summarized_context = self._summarize_relation_context(
                    client, relation["source_name"], 
                    relation["target_name"], relation["relationship"], 
                    relation["context"]
                )
                relation["summarized_context"] = summarized_context
        else:
            # Fallback: basic descriptions without LLM
            for name, concept in concepts_map.items():
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
        
        # Generate request ID
        request_id = self._generate_id(f"{datetime.now().isoformat()}_{len(otel_records)}")
        
        return {
            "knowledge_cognition_request_id": request_id,
            "concepts": concepts,
            "relations": formatted_relations,
            "descriptor": "telemetry knowledge extraction",
            "meta": {
                "records_processed": len(otel_records),
                "concepts_extracted": len(concepts),
                "relations_extracted": len(formatted_relations)
            }
        }
    
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
            for record in otel_records[:50]:  # Truncate to avoid token limits
                span_attrs = record.get("SpanAttributes", {})
                
                # Check if this concept appears in the span
                if (name == span_attrs.get("agent_id") or
                    name == record.get("ServiceName") or
                    name == span_attrs.get("gen_ai.request.model") or
                    name in str(span_attrs)):
                    
                    snippet = {
                        "span_name": record.get("SpanName", ""),
                        "service": record.get("ServiceName", ""),
                        "agent_id": span_attrs.get("agent_id", "")
                    }
                    context_snippets.append(snippet)
            
            if not context_snippets:
                return f"{concept_type.title()}: {name}"
            
            # Generate description using LLM
            prompt = f"""Based on the following OpenTelemetry trace data, generate a concise description (1-2 sentences) for this {concept_type}:

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
            
            prompt = f"""Summarize the following interaction between components in a distributed system in a few sentences describing what is the input and output of the interaction and what information is being exchanged:

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

