#!/usr/bin/env python3
"""Run basic evaluations for the common data models."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data_model import (  # noqa: E402  # pylint: disable=wrong-import-position
    CAKnowledgeRecord,
    CFNEvidenceRecord,
    CFNKnowledgeRecord,
    CFNQueryRequest,
    CFNQueryResponse,
    Concept,
    Config,
    EmbeddingRecord,
    EntityRecord,
    KnowledgeCognitionRequest,
    KnowledgeCognitionResponse,
    MemoryType,
    Path,
    ReasonerCognitionRequest,
    ReasonerCognitionResponse,
    ReasoningRequest,
    ReasoningResponse,
    RecordType,
    Relation,
)


@dataclass
class EvaluationResult:
    name: str
    passed: bool
    details: str


def expect_success(name: str, fn: Callable[[], None]) -> EvaluationResult:
    try:
        fn()
    except Exception as exc:  # pylint: disable=broad-except
        return EvaluationResult(name, False, f"Unexpected error: {exc}")
    return EvaluationResult(name, True, "ok")


def expect_failure(name: str, fn: Callable[[], None], error: type[BaseException]) -> EvaluationResult:
    try:
        fn()
    except error:
        return EvaluationResult(name, True, f"raised {error.__name__} as expected")
    except Exception as exc:  # pylint: disable=broad-except
        return EvaluationResult(name, False, f"raised {type(exc).__name__}, expected {error.__name__}")
    return EvaluationResult(name, False, "did not raise")


def evaluate_config() -> List[EvaluationResult]:
    return [
        expect_success(
            "Config instantiation",
            lambda: Config(),
        )
    ]


def evaluate_graph_models() -> List[EvaluationResult]:
    results: List[EvaluationResult] = []

    results.append(
        expect_success(
            "Concept defaults",
            lambda: Concept(name="router", description="Core router"),
        )
    )

    relation = Relation(node_ids=["node-a", "node-b"], relationship="links")
    results.append(
        expect_success(
            "Relation basic",
            lambda: relation,
        )
    )

    results.append(
        expect_success(
            "Path validation",
            lambda: Path(path_sequence=["node-a", "rel-1", "node-b"]),
        )
    )

    results.append(
        expect_failure(
            "Path invalid length",
            lambda: Path(path_sequence=["node-a", "rel-1"]),
            ValueError,
        )
    )

    return results


def evaluate_cognition_models() -> List[EvaluationResult]:
    concept_a = Concept(id="c1", name="router")
    concept_b = Concept(id="c2", name="switch")
    relation = Relation(id="r1", node_ids=["c1", "c2"], relationship="connects")

    valid_request = lambda: KnowledgeCognitionRequest(
        knowledge_cognition_request_id="req-1",
        concepts=[concept_a, concept_b],
        relations=[relation],
        descriptor="simple topology",
    )

    results = [expect_success("KnowledgeCognitionRequest valid", valid_request)]

    invalid_request = lambda: KnowledgeCognitionRequest(
        knowledge_cognition_request_id="req-2",
        concepts=[concept_a],
        relations=[Relation(id="r2", node_ids=["c1", "missing"])],
    )

    results.append(
        expect_failure(
            "KnowledgeCognitionRequest invalid relation",
            invalid_request,
            ValueError,
        )
    )

    results.append(
        expect_success(
            "KnowledgeCognitionResponse basic",
            lambda: KnowledgeCognitionResponse(
                knowledge_cognition_response_id="resp-1",
                status="complete",
                knowledge_cognition_request_id="req-1",
            ),
        )
    )

    results.append(
        expect_success(
            "CAKnowledgeRecord typing",
            lambda: CAKnowledgeRecord(
                record_id="rec-1",
                record_type=RecordType.string,
                content="core router",
            ),
        )
    )

    results.append(
        expect_failure(
            "CAKnowledgeRecord invalid bytes requirement",
            lambda: CAKnowledgeRecord(
                record_id="rec-2",
                record_type=RecordType.image,
                content="not-bytes",
            ),
            ValueError,
        )
    )

    results.append(
        expect_success(
            "ReasonerCognitionRequest",
            lambda: ReasonerCognitionRequest(
                reasoner_cognition_request_id="rcr-1",
                records=[
                    CAKnowledgeRecord(
                        record_id="rec-3",
                        record_type=RecordType.json,
                        content={"vendor": "cisco"},
                    )
                ],
                intent="summarize",
            ),
        )
    )

    results.append(
        expect_success(
            "ReasonerCognitionResponse",
            lambda: ReasonerCognitionResponse(
                reasoner_cognition_response_id="rcp-1",
                status="ok",
                reasoner_cognition_request_id="rcr-1",
                records=[
                    CFNKnowledgeRecord(
                        id="kn-1",
                        type=RecordType.string,
                        content="stored summary",
                    )
                ],
            ),
        )
    )

    return results


def evaluate_memory_models() -> List[EvaluationResult]:
    results: List[EvaluationResult] = []

    results.append(
        expect_success(
            "CFNKnowledgeRecord string",
            lambda: CFNKnowledgeRecord(id="kn-1", content="details"),
        )
    )
    results.append(
        expect_success(
            "CFNKnowledgeRecord dict",
            lambda: CFNKnowledgeRecord(id="kn-2", content={"field": "value"}),
        )
    )
    results.append(
        expect_failure(
            "CFNKnowledgeRecord invalid content",
            lambda: CFNKnowledgeRecord(id="kn-3", content=[1, 2, 3]),
            ValueError,
        )
    )

    results.append(
        expect_success(
            "CFNEvidenceRecord string",
            lambda: CFNEvidenceRecord(id="ev-1", content="text"),
        )
    )
    results.append(
        expect_failure(
            "CFNEvidenceRecord invalid content",
            lambda: CFNEvidenceRecord(id="ev-2", content=42),
            ValueError,
        )
    )

    records = [CFNKnowledgeRecord(id="kn-10", content="details")]
    evidence = [CFNEvidenceRecord(id="ev-10", content={"score": 0.9})]

    results.append(
        expect_success(
            "ReasoningRequest",
            lambda: ReasoningRequest(request_id="req-100", records=records),
        )
    )

    results.append(
        expect_success(
            "ReasoningResponse",
            lambda: ReasoningResponse(response_id="req-100", evidence=evidence),
        )
    )

    sample_entity = EntityRecord(entity_name="router_a", embeddings=EmbeddingRecord(data=[0.1, 0.2]))

    results.append(
        expect_failure(
            "CFNQueryRequest missing entities",
            lambda: CFNQueryRequest(),
            ValueError,
        )
    )

    results.append(
        expect_success(
            "CFNQueryRequest with entity",
            lambda: CFNQueryRequest(entities=[sample_entity], memory_type=MemoryType.auto),
        )
    )

    results.append(
        expect_success(
            "CFNQueryResponse basic",
            lambda: CFNQueryResponse(
                queried_entities=[sample_entity],
                retrieved_concepts=records,
                relations=[{"id": "rel-1"}],
            ),
        )
    )

    return results


def format_results(results: Iterable[EvaluationResult]) -> None:
    failures = 0
    total = 0
    for result in results:
        total += 1
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name} - {result.details}")
        if not result.passed:
            failures += 1
    print("-" * 60)
    print(f"Evaluations: {total}, Passed: {total - failures}, Failed: {failures}")
    if failures:
        sys.exit(1)


def main() -> None:
    evaluations: List[EvaluationResult] = []
    evaluations += evaluate_config()
    evaluations += evaluate_graph_models()
    evaluations += evaluate_cognition_models()
    evaluations += evaluate_memory_models()
    format_results(evaluations)


if __name__ == "__main__":
    main()
