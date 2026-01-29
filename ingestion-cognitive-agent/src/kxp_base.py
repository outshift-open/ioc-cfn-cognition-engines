"""
Base SDK for Knowledge Extraction and Ingestion Platform (KXP) Adapters
Provides common functionality for data source adapters
"""
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging
import time
from datetime import datetime
import requests
from tkf_csp_data_model_lib.csp import KnowledgeCognitionRequest, Concept, Relation


@dataclass
class MetricsObject:
    """Operational metrics for the adapter"""
    records_processed: int = 0
    records_sent: int = 0
    records_failed: int = 0
    last_run_timestamp: Optional[datetime] = None
    last_run_duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class AdapterConfig:
    """Configuration for adapter initialization"""
    data_source_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    extraction_rules: Dict[str, Any]


class AdapterSDK:
    """
    Base SDK class for KXP adapters
    Provides common functionality for data extraction and communication with CSP Manager
    """

    def __init__(self, csp_manager_url: str = "http://0.0.0.0:8000"):
        self.config: Optional[AdapterConfig] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._default_log_level = logging.INFO
        self.logger.setLevel(self._default_log_level)
        self.metrics = MetricsObject()
        self._initialized = False
        self.csp_manager_url = csp_manager_url

    def load(self) -> Dict[str, Any]:
        """
        Load data from the configured data source
        
        Returns:
            Dict containing loaded data and status
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call init() first.")

        start_time = time.time()
        try:
            self.logger.info("Loading data from source...")

            # To be implemented by specific adapters
            result = self._load_impl()

            duration = time.time() - start_time
            self.metrics.last_run_timestamp = datetime.now()
            self.metrics.last_run_duration_seconds = duration

            self.logger.info(f"Data loaded successfully in {duration:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            self.metrics.errors.append(str(e))
            raise

    def _load_impl(self) -> Dict[str, Any]:
        """
        Implementation of load logic - to be overridden by specific adapters
        """
        raise NotImplementedError("Subclasses must implement _load_impl()")

    def build_csp_payload(
            self,
            extraction_output: Dict[str, Any],
            description="",
            metadata=None,
    ) -> KnowledgeCognitionRequest:
        """
        Transform LLM output into KnowledgeCognitionRequest object.

        Args:
            :param extraction_output:  Raw output from extraction with concepts and relations
            :param description: General Description about the knowledge extracted if nothing provided it will be empty
            :param metadata: dictionary with additional information for the payload if nothing provided it will be empty

        Returns:
            KnowledgeCognitionRequest

        """
        # Create concept objects and build name->id mapping
        if metadata is None:
            metadata = {}
        concept_map = {}  # name -> concept object
        concepts = []

        for concept_data in extraction_output.get("concepts", []):
            if "id" in concept_data:
                concept = Concept(
                    id=concept_data.get("id"),
                    name=concept_data.get("name"),
                    description=concept_data.get("description"),
                    type=concept_data.get("type"),
                    attributes=concept_data.get("attributes", {})
                )
            else:
                concept = Concept(
                        name=concept_data.get("name"),
                        description=concept_data.get("description"),
                        type=concept_data.get("type"),
                        attributes=concept_data.get("attributes", {})
                    )
            concepts.append(concept)
            # Store mapping for relation building
            if concept.id:
                concept_map[concept.id] = concept
        # Create relation objects
        relations = []
        for relation_data in extraction_output.get("relations", []):
            # Check if node_ids are already provided (new format)
            node_ids = relation_data.get("node_ids")
            
            if node_ids:
                # New format: node_ids are already provided
                relation_kwargs = {
                    "node_ids": node_ids,
                    "relationship": relation_data.get("relationship"),
                    "attributes": relation_data.get("attributes", {})
                }
                
                # Include relation ID if provided
                if relation_data.get("id"):
                    relation_kwargs["id"] = relation_data.get("id")
                
                relation = Relation(**relation_kwargs)
                relations.append(relation)


        # Build the final request
        return KnowledgeCognitionRequest(
            concepts=concepts,
            relations=relations,
            description=description,
            meta=metadata
        )

    def reportHealthAndOtherDiagnosticInfo(self) -> Dict[str, Any]:
        """
        Report health status and diagnostic information
        
        Returns:
            Dict containing health status and diagnostics
        """
        health_info = {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "metrics": {
                "records_processed": self.metrics.records_processed,
                "records_sent": self.metrics.records_sent,
                "records_failed": self.metrics.records_failed,
                "last_run": self.metrics.last_run_timestamp.isoformat() if self.metrics.last_run_timestamp else None,
                "last_run_duration_seconds": self.metrics.last_run_duration_seconds
            },
            "recent_errors": self.metrics.errors[-5:]  # Last 5 errors
        }
        return health_info

    def send_to_csp_manager(self, llm_response: dict, kxp_id: str, description , meta_data) -> bool:
        """
        Send knowledge graph to CSP Manager (placeholder for your integration).


        """
        kg_request = self.build_csp_payload(llm_response, description, meta_data)
        self.logger.info(f"sending KG {kg_request.knowledge_cognition_request_id} to CSP Manager")
        response = requests.post(f"{self.csp_manager_url}/api/knowledge/adapters/{kxp_id}", json.dumps(kg_request.model_dump()))
        if response.status_code == 200:
            response = response.json()
            self.logger.info(
                f"The knowledge record is successfully sent to CSP manager and received knowledge cognition response id {response['knowledge_cognition_response_id']}")
            return True
        else:
            self.logger.error(f"Failed to send knowledge record to CSP manager. Status code: {response.status_code}, "
                              f"Response: {response.text}")
            return False

    def setLogLevel(self, level: int):
        """
        Set the logging level for the adapter
        
        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO, etc.)
        """
        self.logger.setLevel(level)
        self.logger.info(f"Log level set to: {logging.getLevelName(level)}")

    def resetLogLevel(self):
        """
        Reset logging level to default (INFO)
        """
        self.logger.setLevel(self._default_log_level)
        self.logger.info(f"Log level reset to: {logging.getLevelName(self._default_log_level)}")

    def getOperationalMetrics(self) -> MetricsObject:
        """
        Get operational metrics for the adapter
        
        Returns:
            MetricsObject with current metrics
        """
        return self.metrics

