"""
API routes for the Telemetry Extraction Service.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Depends

from ..config.settings import settings
from ..dependencies import (
    get_extraction_service,
    get_knowledge_processor,
    get_data_repository,
)
from ..agent.service import TelemetryExtractionService
from ..agent.knowledge_processor import KnowledgeProcessor
from ..data.mock_repo import MockDataRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["extraction"])


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


@router.get("/extract/entities_and_relations/from_file")
async def extract_entities_and_relations_from_file(
    file_path: str,
    save_output: bool = False,
    service: TelemetryExtractionService = Depends(get_extraction_service),
    repository: MockDataRepository = Depends(get_data_repository),
):
    """
    Load OTEL data from specified JSON file, extract entities and relations,
    generate embeddings, and optionally perform semantic deduplication.
    
    Args:
        file_path: Path to the JSON file containing OTEL data
        save_output: Save the output to a file (default: False)
    
    Environment variables used:
        ENABLE_EMBEDDINGS: Enable embedding generation (default: true)
        ENABLE_DEDUP: Enable deduplication (default: true)
        SIMILARITY_THRESHOLD: Cosine similarity threshold (default: 0.95)
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
        AZURE_OPENAI_API_KEY: Azure OpenAI API key
        AZURE_OPENAI_DEPLOYMENT: Azure OpenAI deployment name
    """
    try:
        # Load data from file using repository
        path = Path(file_path)
        otel_data = repository.load_from_file(path)
        
        # Step 1: Extract entities and relations
        result = service.extract_entities_and_relations(otel_data)
        
        # Step 2: Process through knowledge processor (embeddings + optional dedup)
        processor = get_knowledge_processor()
        result = processor.process(result)
        
        # Save to file if requested
        if save_output:
            output_filename = f"extracted_entities_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            repository.save_output(result, output_filename)
        
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error extracting entities from file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/entities_and_relations/batch")
async def extract_entities_and_relations_batch(
    request: Request,
    save_output: bool = False,
    service: TelemetryExtractionService = Depends(get_extraction_service),
    repository: MockDataRepository = Depends(get_data_repository),
):
    """
    Batch processing API for OTEL traces.
    
    Accepts a large batch of OTEL data (JSON array or NDJSON format), extracts entities 
    and relations, generates embeddings, optionally performs semantic deduplication, 
    and returns the final knowledge cognition request.
    
    Pipeline:
    1. Parse OTEL traces from request body
    2. Extract concepts and relationships
    3. Generate embeddings for concepts
    4. Semantic deduplication using cosine similarity (if enabled)
    5. Deduplicate relations (if enabled)
    6. Return final output in knowledge cognition request format
    
    Args:
        request: The incoming request body (JSON array or NDJSON)
        save_output: Save the output to a file (default: False)
    
    Environment variables used:
        ENABLE_EMBEDDINGS: Enable embedding generation (default: true)
        ENABLE_DEDUP: Enable deduplication (default: true)
        SIMILARITY_THRESHOLD: Cosine similarity threshold (default: 0.95)
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
        AZURE_OPENAI_API_KEY: Azure OpenAI API key
        AZURE_OPENAI_DEPLOYMENT: Azure OpenAI deployment name
    """
    try:
        # Read body and parse using repository
        body = await request.body()
        otel_data = repository.parse_body(body)
        
        # Step 1: Extract entities and relations
        result = service.extract_entities_and_relations(otel_data)
        
        # Step 2: Process through knowledge processor (embeddings + optional dedup)
        processor = get_knowledge_processor()
        result = processor.process(result)
        
        # Save to file if requested
        if save_output:
            output_filename = f"extracted_entities_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            repository.save_output(result, output_filename)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

