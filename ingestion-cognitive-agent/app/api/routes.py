"""
API routes for the Telemetry Extraction Service.
"""

import logging
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..dependencies import (
    get_extraction_service,
    get_concept_relationship_service,
    get_knowledge_processor,
    get_data_repository,
)
from ..agent.service import TelemetryExtractionService, ConceptRelationshipExtractionService
from ..data.mock_repo import MockDataRepository
from .schemas import ExtractionRequest, ExtractionResponseModel, ExtractionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["extraction"])
extraction_router = APIRouter(prefix="/api/knowledge-mgmt", tags=["knowledge-mgmt"])


def _build_response_id(request_id: str | None) -> str:
    """Return the client-supplied request_id or generate a new UUID."""
    return request_id if request_id else str(uuid.uuid4())


def _build_error_response(
    header_dict: dict,
    response_id: str,
    message: str,
    detail: dict | None = None,
) -> dict:
    """Build a standard error response envelope."""
    return {
        "header": header_dict,
        "response_id": response_id,
        "error": {
            "message": message,
            "detail": detail or {},
        },
    }


def _build_success_response(
    header_dict: dict,
    response_id: str,
    concepts: list,
    relations: list,
    descriptor: str,
    metadata: dict,
) -> dict:
    """Build a standard success response envelope."""
    return {
        "header": header_dict,
        "response_id": response_id,
        "concepts": concepts,
        "relations": relations,
        "descriptor": descriptor,
        "metadata": metadata,
    }


# ============== New Unified Extraction Endpoint ==============


@extraction_router.post(
    "/extraction",
    response_model=ExtractionResponseModel,
    response_model_exclude_none=True
)
async def knowledge_extraction(
    body: ExtractionRequest,
    concept_service: ConceptRelationshipExtractionService = Depends(get_concept_relationship_service),
):
    """
    Unified knowledge extraction endpoint.

    Accepts a structured request with header, request_id, and a payload
    whose ``metadata.format`` declares the data format (e.g. 'observe-sdk-otel',
    'openclaw').  All formats are processed through the ConceptRelationship
    extraction pipeline (LLM-based concept and relationship identification).

    Pipeline:
    1. Validate header and format
    2. Extract concepts and relationships via ConceptRelationshipExtractionService
    3. Generate embeddings and optionally deduplicate
    4. Return response with header echo and response_id
    """
    response_id = body.request_id
    data_format = body.payload.metadata.format.strip().lower()

    otel_data = body.payload.data
    if not otel_data:
        error_resp = ExtractionResponseModel(
            header=body.header,
            response_id=response_id,
            error=ExtractionError(
                message="BAD_REQUEST",
                detail={"Validation error": "payload.data must be a non-empty array of records."}
            ),
        )
        return JSONResponse(status_code=400, content=error_resp.model_dump())

    try:
        result = concept_service.extract_concepts_and_relationships(
            otel_data,
            request_id=response_id,
            format_descriptor=data_format,
        )

        processor = get_knowledge_processor()
        result = processor.process(result)

        return ExtractionResponseModel(
            header=body.header,
            response_id=response_id,
            concepts=result.get("concepts", []),
            relations=result.get("relations", []),
            descriptor=result.get("descriptor", data_format),
            metadata=result.get("meta", {}),
        )

    except Exception as e:
        logger.error(f"Error in batch extraction {response_id}: {e}")
        error_resp = ExtractionResponseModel(
            header=body.header,
            response_id=response_id,
            error=ExtractionError(
                message="INTERNAL_ERROR",
                detail={"traceback": traceback.format_exc()},
            ),
            concepts=[]
        )
        return JSONResponse(status_code=500, content=error_resp.model_dump())


# ============== Legacy Batch Endpoints (updated to new signature) ==============


@router.post("/extract/entities_and_relations/batch")
async def extract_entities_and_relations_batch(
    body: ExtractionRequest,
    service: TelemetryExtractionService = Depends(get_extraction_service),
):
    """
    Batch entity & relation extraction (legacy path, new request/response envelope).

    Pipeline:
    1. Parse structured request body
    2. Extract concepts and relationships
    3. Generate embeddings and optionally deduplicate
    4. Return response in the unified envelope format
    """
    header_dict = body.header.model_dump()
    response_id = _build_response_id(body.request_id)

    try:
        otel_data = body.payload.data
        if not otel_data:
            return JSONResponse(
                status_code=400,
                content=_build_error_response(
                    header_dict, response_id,
                    "payload.data must be a non-empty array of records.",
                ),
            )

        data_format = body.payload.metadata.format.strip().lower()
        result = service.extract_entities_and_relations(
            otel_data,
            request_id=response_id,
            format_descriptor=data_format,
        )

        processor = get_knowledge_processor()
        result = processor.process(result)

        return _build_success_response(
            header_dict=header_dict,
            response_id=response_id,
            concepts=result.get("concepts", []),
            relations=result.get("relations", []),
            descriptor=result.get("descriptor", data_format),
            metadata=result.get("meta", {}),
        )

    except Exception as exc:
        logger.error("Batch entity extraction failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=_build_error_response(
                header_dict, response_id,
                "An internal error occurred during entity extraction.",
                detail={"traceback": traceback.format_exc()},
            ),
        )


@router.post("/extract/concepts_and_relationships/batch")
async def extract_concepts_and_relationships_batch(
    body: ExtractionRequest,
    service: ConceptRelationshipExtractionService = Depends(get_concept_relationship_service),
):
    """
    Batch concept & relationship extraction (legacy path, new request/response envelope).

    Pipeline:
    1. Parse structured request body
    2. Extract high-level concepts and relationships via LLM
    3. Generate embeddings and optionally deduplicate
    4. Return response in the unified envelope format
    """
    header_dict = body.header.model_dump()
    response_id = _build_response_id(body.request_id)

    try:
        otel_data = body.payload.data
        if not otel_data:
            return JSONResponse(
                status_code=400,
                content=_build_error_response(
                    header_dict, response_id,
                    "payload.data must be a non-empty array of records.",
                ),
            )

        data_format = body.payload.metadata.format.strip().lower()
        result = service.extract_concepts_and_relationships(
            otel_data,
            request_id=response_id,
            format_descriptor=data_format,
        )

        processor = get_knowledge_processor()
        result = processor.process(result)

        return _build_success_response(
            header_dict=header_dict,
            response_id=response_id,
            concepts=result.get("concepts", []),
            relations=result.get("relations", []),
            descriptor=result.get("descriptor", data_format),
            metadata=result.get("meta", {}),
        )

    except Exception as exc:
        logger.error("Concept-relationship extraction failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=_build_error_response(
                header_dict, response_id,
                "An internal error occurred during concept extraction.",
                detail={"traceback": traceback.format_exc()},
            ),
        )


# ============== Operational Endpoints ==============


@router.get("/metrics")
async def get_metrics(
    service: TelemetryExtractionService = Depends(get_extraction_service)
):
    """Get operational metrics."""
    metrics = service.get_operational_metrics()
    return {
        "records_processed": metrics.records_processed,
        "records_sent": metrics.records_sent,
        "records_failed": metrics.records_failed,
        "last_run_timestamp": metrics.last_run_timestamp.isoformat() if metrics.last_run_timestamp else None,
        "last_run_duration_seconds": metrics.last_run_duration_seconds,
        "recent_errors": metrics.errors[-10:]
    }


# ============== File-based Endpoints (kept for dev/testing) ==============


@router.get("/extract/entities_and_relations/from_file")
async def extract_entities_and_relations_from_file(
    file_path: str,
    save_output: bool = False,
    service: TelemetryExtractionService = Depends(get_extraction_service),
    repository: MockDataRepository = Depends(get_data_repository),
):
    """
    Load OTEL data from a JSON file, extract entities and relations,
    generate embeddings, and optionally perform semantic deduplication.
    """
    try:
        path = Path(file_path)
        otel_data = repository.load_from_file(path)
        
        result = service.extract_entities_and_relations(otel_data)
        
        processor = get_knowledge_processor()
        result = processor.process(result)
        
        if save_output:
            output_filename = f"extracted_entities_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            repository.save_output(result, output_filename)
        
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error extracting entities from file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract/concepts_and_relationships/from_file")
async def extract_concepts_and_relationships_from_file(
    file_path: str,
    save_output: bool = False,
    service: ConceptRelationshipExtractionService = Depends(get_concept_relationship_service),
    repository: MockDataRepository = Depends(get_data_repository),
):
    """
    Load OTEL data from a JSON file and extract high-level concepts and relationships.
    """
    try:
        path = Path(file_path)
        otel_data = repository.load_from_file(path)

        result = service.extract_concepts_and_relationships(otel_data)

        processor = get_knowledge_processor()
        result = processor.process(result)

        if save_output:
            output_filename = (
                f"concept_relationships_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            )
            repository.save_output(result, output_filename)

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error extracting concepts from file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

