# Caching Layer

An in-process FAISS vector store library used by other agents in this workspace. It wraps a FAISS flat index behind a simple Python API for storing and retrieving embeddings.

## Features

- `CachingLayer` class backed by a FAISS flat index (`IndexFlatL2` or `IndexFlatIP`)
- Store text (auto-embedded) or pre-computed vectors
- Nearest-neighbor similarity search over cached entries
- Pluggable embedding function — callers supply their own `embed_fn`

## Project Layout

```
caching/
├── Taskfile.yml
├── requirements.txt
├── app/
│   ├── __init__.py
│   └── agent/
│       ├── __init__.py
│       └── caching_layer.py
└── tests/
    ├── conftest.py
    └── unit/
        └── test_caching_layer.py
```

## Usage

Import `CachingLayer` directly from the package:

```python
from app.agent.caching_layer import CachingLayer

layer = CachingLayer(vector_dimension=384, metric="l2", embed_fn=my_embed_fn)

layer.store_knowledge(text="some concept name")

results = layer.search_similar(text="query", k=5)
```

The `ingestion` service loads this class via `importlib.util` to avoid namespace collisions between sibling `app` packages.

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `vector_dimension` | `1536` | Embedding dimension for the FAISS index. Set to `384` when using `all-MiniLM-L6-v2`. |
| `metric` | `l2` | FAISS distance metric (`l2` or `ip`) |
| `embed_fn` | hash-based stub | Callable that maps a string to a numpy array. Supply a real embedding function in production. |

## Running Tests

```bash
cd caching
poetry run pytest tests -v
```

## Taskfile Shortcuts

```bash
cd caching && task test    # run tests
cd caching && task lint    # lint
cd caching && task format  # auto-format
```
