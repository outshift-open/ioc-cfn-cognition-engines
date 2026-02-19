from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ReasonerCognitionRequest(BaseModel):
    reasoner_cognition_request_id: str = Field(default="auto")
    intent: str
    records: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TKFKnowledgeRecord(BaseModel):
    id: str = Field(default="auto")
    type: Literal["json"] = "json"
    content: Dict[str, Any]


class ReasonerCognitionResponse(BaseModel):
    status: Literal["OK", "ERROR"] = "OK"
    reasoner_cognition_request_id: str
    records: List[TKFKnowledgeRecord] = Field(default_factory=list)
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ---- Placeholder DB-facing schemas (to be finalized later) ----

class GraphPathsRequest(BaseModel):
    source_id: str
    target_id: str
    max_depth: int = 3
    limit: int = 10
    relations: Optional[List[str]] = None


class TkfPathEdge(BaseModel):
    from_id: str
    relation: str
    to_id: str
    from_name: Optional[str] = None
    to_name: Optional[str] = None


class TkfPath(BaseModel):
    node_ids: Optional[List[str]] = None
    edges: List[TkfPathEdge]
    path_length: Optional[int] = None
    symbolic: str


class GraphPathsResponse(BaseModel):
    status: Literal["success", "error"] = "success"
    paths: List[TkfPath] = Field(default_factory=list)


class NeighborsResponse(BaseModel):
    # Keep flexible for now; can be specialized later
    records: List[Dict[str, Any]] = Field(default_factory=list)


class ConceptsByIdsRequest(BaseModel):
    ids: List[str]


class Concept(BaseModel):
    id: str
    name: str
    type: str
    description: str = ""


class ConceptsByIdsResponse(BaseModel):
    concepts: List[Concept] = Field(default_factory=list)
