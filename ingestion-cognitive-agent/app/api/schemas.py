"""
Pydantic request/response models for API endpoints.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# ============== Request Models ==============

class OTelSpanAttributes(BaseModel):
    """OpenTelemetry span attributes."""
    agent_id: Optional[str] = None
    execution_success: Optional[str] = Field(None, alias="execution.success")
    gen_ai_request_model: Optional[str] = Field(None, alias="gen_ai.request.model")
    gen_ai_response_model: Optional[str] = Field(None, alias="gen_ai.response.model")
    
    class Config:
        extra = "allow"  # Allow additional attributes


class OTelRecord(BaseModel):
    """Single OpenTelemetry trace record."""
    Timestamp: Optional[str] = None
    TraceId: Optional[str] = None
    SpanId: Optional[str] = None
    ParentSpanId: Optional[str] = None
    SpanName: Optional[str] = None
    SpanKind: Optional[str] = None
    ServiceName: Optional[str] = None
    ResourceAttributes: Optional[Dict[str, Any]] = None
    SpanAttributes: Optional[Dict[str, Any]] = None
    Duration: Optional[int] = None
    StatusCode: Optional[str] = None
    Events_Name: Optional[List[str]] = Field(None, alias="Events.Name")
    Events_Attributes: Optional[List[Dict[str, Any]]] = Field(None, alias="Events.Attributes")
    
    class Config:
        extra = "allow"
        populate_by_name = True


# ============== Response Models ==============

class ConceptAttributes(BaseModel):
    """Attributes for a concept."""
    concept_type: str
    embedding: Optional[List[List[float]]] = None
    
    class Config:
        extra = "allow"


class Concept(BaseModel):
    """Extracted concept/entity."""
    id: str
    name: str
    description: str
    type: str = "concept"
    attributes: ConceptAttributes


class RelationAttributes(BaseModel):
    """Attributes for a relation."""
    source_name: str
    target_name: str
    summarized_context: str
    
    class Config:
        extra = "allow"


class Relation(BaseModel):
    """Extracted relationship between concepts."""
    id: str
    node_ids: List[str]
    relationship: str
    attributes: RelationAttributes


class ExtractionMeta(BaseModel):
    """Metadata about the extraction process."""
    records_processed: int
    concepts_extracted: int
    relations_extracted: int
    dedup_enabled: Optional[bool] = None
    concepts_deduped: Optional[int] = None
    relations_deduped: Optional[int] = None


class ExtractionResponse(BaseModel):
    """Response from entity and relation extraction."""
    knowledge_cognition_request_id: str
    concepts: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    descriptor: str
    meta: ExtractionMeta


class MetricsResponse(BaseModel):
    """Operational metrics response."""
    records_processed: int
    records_sent: int
    records_failed: int
    last_run_timestamp: Optional[str] = None
    last_run_duration_seconds: float
    recent_errors: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    initialized: bool
    metrics: Dict[str, Any]
    recent_errors: List[str]

