"""
Telemetry Extraction Adapter - Extracts knowledge from OpenTelemetry data

This adapter processes OpenTelemetry (OTEL) trace data and extracts entities and relationships
to build a knowledge graph of system interactions.

Example OTEL Trace Format (JSON/JSONL):
---------------------------------------
{
  "Timestamp": "2025-12-22 18:37:22.545847221",
  "TraceId": "162b29522a339e6b1acb21b8041dcda5",
  "SpanId": "2b6a701a27797f5c",
  "ParentSpanId": "",
  "SpanName": "farm_agent.build_graph.agent",
  "SpanKind": "Internal",
  "ServiceName": "corto.farm_agent",
  "ResourceAttributes": {
    "service.name": "corto.farm_agent"
  },
  "SpanAttributes": {
    "agent_id": "farm_agent.build_graph",
    "execution.success": "true",
    "ioa_observe.entity.name": "farm_agent.build_graph",
    "ioa_observe.span.kind": "agent",
    "gen_ai.request.model": "gpt-4",
    "gen_ai.response.model": "gpt-4"
  },
  "Duration": 21346166,
  "StatusCode": "Unset",
  "Events.Name": ["agent_start_event"],
  "Events.Attributes": [
    {
      "agent_name": "farm_agent.build_graph",
      "type": "agent"
    }
  ]
}

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

import json
import logging
import hashlib
import os
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

from kxp_base import (
    AdapterSDK, AdapterConfig, MetricsObject
)

# Load environment variables from .env file
load_dotenv()
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


class TelemetryExtractionAdapter(AdapterSDK):
    """
    Adapter for extracting knowledge from OpenTelemetry (OTEL) data
    """
    
    def __init__(self):
        # Get CSP manager URL from parameter, environment variable, or use default
        csp_url = os.getenv("CSP_MANAGER_URL", "http://0.0.0.0:8000")
        super().__init__(csp_manager_url=csp_url)
        self.id = "uuid5"
        self.logger = logging.getLogger("TelemetryAdapter")
        
    def _load_impl(self) -> Dict[str, Any]:
        """
        Load implementation - reads from example OTEL file
        """
        # For now, load from example file
        file_path = Path(__file__).parent.parent.parent / "test_files" / "example_otel_2.json"
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return {"status": "success", "records": len(data), "data": data}
        except Exception as e:
            self.logger.error(f"Failed to load OTEL data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    
    def _generate_id(self, text: str) -> str:
        """Generate deterministic ID from text using MD5 hash"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def extract_entities_and_relations(
        self,
        otel_records: List[Dict[str, Any]],
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from OTEL data in the specified format.
        
        Extracts:
        - Concepts: users, agents, tools, LLMs
        - Relations: information flow between concepts
        - Uses Azure OpenAI to generate descriptions and summarize contexts
        
        Args:
            otel_records: List of OpenTelemetry span records
            azure_endpoint: Azure OpenAI endpoint (defaults to env var AZURE_OPENAI_ENDPOINT)
            azure_api_key: Azure OpenAI API key (defaults to env var AZURE_OPENAI_API_KEY)
            azure_deployment: Azure deployment name (defaults to env var AZURE_OPENAI_DEPLOYMENT)
            
        Returns:
            Dictionary with knowledge_cognition_request_id, concepts, relations, descriptor, meta
        """
        # Initialize Azure OpenAI client
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        
        if not endpoint or not api_key:
            self.logger.warning("Azure OpenAI credentials not provided, using basic extraction")
            client = None
        else:
            if AzureOpenAI is None:
                raise ImportError("openai package not installed. Install with: pip install openai")
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-08-01-preview"
            )
        
        # Step 1: Extract concepts deterministically
        concepts_map = {}  # name -> concept data
        span_lookup = {}  # span_id -> span data
        
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
            span_name = record.get("SpanName", "")
            parent_span_id = record.get("ParentSpanId")
            
            # Get source concept (current span)
            agent_id = span_attrs.get("agent_id")
            service_name = record.get("ServiceName")
            model_name = span_attrs.get("gen_ai.request.model") or span_attrs.get("gen_ai.response.model")
            
            source_name = agent_id or service_name
            
            # Extract relations based on span type
            
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
                    client, deployment, name, concept["type"], otel_records
                )
                concept["description"] = description
            
            # Summarize contexts for relations
            for relation in relations:
                summarized_context = self._summarize_relation_context(
                    client, deployment, relation["source_name"], 
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
        
        # Step 4: Format output according to format.json
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
        deployment: str,
        name: str,
        concept_type: str,
        otel_records: List[Dict[str, Any]]
    ) -> str:
        """Generate concept description using Azure OpenAI"""
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
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing distributed system traces."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate description for {name}: {str(e)}")
            return f"{concept_type.title()}: {name}"
    
    def _summarize_relation_context(
        self,
        client: Any,
        deployment: str,
        source_name: str,
        target_name: str,
        relationship: str,
        context: Dict[str, Any]
    ) -> str:
        """Summarize relation context using Azure OpenAI"""
        try:
            # Extract relevant fields from context
            relevant_fields = {}
            for key, value in context.items():
                if any(term in key.lower() for term in ["prompt", "content", "message", "input", "output", "tool", "function"]):

                    relevant_fields[key] = value
            
            if not relevant_fields:
                return f"{source_name} {relationship.lower()} {target_name}"
            
            prompt = f"""Summarize the following interaction between components in a distributed system in a few sentences describing what is the input and output of the interaction and what information is being exchanged. :

Source: {source_name}
Target: {target_name}
Relationship: {relationship}

Context:
{json.dumps(relevant_fields, indent=2)}

Provide a brief summary of what happened in this interaction."""
            
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing distributed system interactions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to summarize context: {str(e)}")
            return f"{source_name} {relationship.lower()} {target_name}"


# Initialize FastAPI app
app = FastAPI(title="Telemetry Extraction Service", version="1.0.0")

# Initialize adapter
adapter = TelemetryExtractionAdapter()


@app.on_event("startup")
async def startup_event():
    """Initialize adapter on startup"""
    logging.info("Telemetry adapter starting up")



@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Open Telemetry Extraction Service", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_info = adapter.reportHealthAndOtherDiagnosticInfo()
    return JSONResponse(content=health_info)


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get operational metrics"""
    metrics = adapter.getOperationalMetrics()
    return {
        "records_processed": metrics.records_processed,
        "records_sent": metrics.records_sent,
        "records_failed": metrics.records_failed,
        "last_run_timestamp": metrics.last_run_timestamp.isoformat() if metrics.last_run_timestamp else None,
        "last_run_duration_seconds": metrics.last_run_duration_seconds,
        "recent_errors": metrics.errors[-10:]
    }



@app.get("/api/v1/extract/entities_and_relations/from_file")
async def extract_entities_and_relations_from_file(
    file_path: str,
    save_output: bool = False,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_deployment: Optional[str] = None
):
    """
    Load OTEL data from specified JSON file and extract entities and relations.
    
    Args:
        file_path: Path to the JSON file containing OTEL data
        save_output: Optional flag to save the output to a file
        azure_endpoint: Optional Azure OpenAI endpoint
        azure_api_key: Optional Azure OpenAI API key
        azure_deployment: Optional Azure OpenAI deployment name
    """
    try:
        # Load data from file
        path = Path(file_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if path.suffix not in ['.json', '.jsonl']:
            raise HTTPException(status_code=400, detail="File must be a JSON or JSONL file")
        
        # Handle JSON or JSONL format
        with open(path, 'r') as f:
            if path.suffix == '.jsonl':
                # JSONL: each line is a separate JSON object
                otel_data = [json.loads(line.strip()) for line in f if line.strip()]
            else:
                # Regular JSON
                otel_data = json.load(f)
        
        # Extract entities and relations
        result = adapter.extract_entities_and_relations(
            otel_data,
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_deployment=azure_deployment
        )
        
        # Save to file if requested
        if save_output:
            output_filename = f"extracted_entities_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            try:
                with open(output_filename, "w") as outfile:
                    json.dump(result, outfile, indent=2)
            except Exception as e:
                logging.error(f"Failed to save extraction result to {output_filename}: {e}")
        
        adapter.send_to_csp_manager(result, adapter.id, result["descriptor"], result["meta"])

        return result
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8086)

