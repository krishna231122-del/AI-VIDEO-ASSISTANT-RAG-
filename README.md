# AI-VIDEO-ASSISTANT-RAG-


# AI Video Agent

Lightweight toolkit for extracting, transcribing, summarizing, and building vector stores from video/audio assets.

## Features
- Audio extraction and preprocessing: see [`Core.extractor`](Core/extractor.py) and [`utils.audio_processor`](utils/audio_processor.py).
- Speech-to-text/transcription: see [`Core.transcriber`](Core/transcriber.py).
- Retrieval-Augmented Generation pipeline: see [`Core.rag_engine`](Core/rag_engine.py).
- Summarization utilities: see [`Core.summary`](Core/summary.py).
- Vector store management: see [`Core.vector_store`](Core/vector_store.py) and the [`vector_db/`](vector_db) folder.

## Quickstart

1. Install dependencies
```sh
pip install -r [REQUIREMENTS.txt](http://_vscodecontentref_/0)
# or use pyproject
pip install .
