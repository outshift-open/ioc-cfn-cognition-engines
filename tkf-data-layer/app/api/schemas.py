"""Request/response schemas aligned with evidence-gathering-agent contract."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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


class SemanticSimilarRequest(BaseModel):
    query_vector: List[float] = Field(default_factory=list, description="Embedding vector (list of floats)")
    k: int = 5

    @field_validator("query_vector", mode="before")
    @classmethod
    def coerce_query_vector(cls, v: Any) -> List[float]:
        if v is None:
            return []
        if isinstance(v, list):
            return [float(x) for x in v]
        return []


class SemanticSimilarItem(BaseModel):
    distance: float
    concept: Dict[str, Any]
    relations: List[Dict[str, Any]] = Field(default_factory=list)
    neighbor_concepts: List[Dict[str, Any]] = Field(default_factory=list)
