"""Unit tests for the RAG preprocessing pipeline."""

import json

import numpy as np
import pytest

from ingestion.app.agent.rag import RagPipeline


class _StubEmbeddingManager:
    def generate_embedding(self, text):
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def test_document_text_and_metadata_merges_keys_with_top_level_precedence():
    pipeline = RagPipeline(embedding_manager=_StubEmbeddingManager())
    doc = {
        "text": "  hello world  ",
        "metadata": {"source": "nested", "owner": "nested-owner"},
        "source": "top-level",
        "doc_id": "doc-1",
    }

    body, meta = pipeline.document_text_and_metadata(doc, "text", "metadata")

    assert body == "hello world"
    assert meta["source"] == "top-level"
    assert meta["owner"] == "nested-owner"
    assert meta["doc_id"] == "doc-1"


def test_extract_ingest_config_validates_required_sections():
    pipeline = RagPipeline(embedding_manager=_StubEmbeddingManager())
    with pytest.raises(ValueError, match="missing required section"):
        pipeline.extract_ingest_config({"chunking": {"chunk_size": 10, "chunk_overlap": 1}})


def test_chunk_text_rejects_invalid_overlap():
    pipeline = RagPipeline(embedding_manager=_StubEmbeddingManager())
    with pytest.raises(ValueError, match="strictly less than"):
        pipeline.chunk_text("one two three four", chunk_size=4, overlap=4)


def test_ingest_chunks_adds_chunk_and_doc_indices():
    pipeline = RagPipeline(embedding_manager=_StubEmbeddingManager())
    cfg = {
        "ingestion": {"text_key": "text", "metadata_key": "metadata"},
        "chunking": {"chunk_size": 2, "chunk_overlap": 0},
    }
    docs = [{"text": "alpha beta gamma", "metadata": {"source": "test"}}]

    rows = pipeline.ingest_chunks(
        docs=docs,
        cfg=cfg,
        embed_fn=lambda chunk: np.array([[1.0, 2.0]], dtype=np.float32),
    )

    assert len(rows) >= 1
    assert rows[0]["metadata"]["doc_index"] == 0
    assert "chunk_index" in rows[0]["metadata"]
    assert rows[0]["embedding"] == [[1.0, 2.0]]


def test_load_config_rejects_non_object_json(tmp_path):
    config_file = tmp_path / "rag_config.json"
    config_file.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    pipeline = RagPipeline(embedding_manager=_StubEmbeddingManager())

    with pytest.raises(ValueError, match="root must be a JSON object"):
        pipeline.load_config(config_file)
