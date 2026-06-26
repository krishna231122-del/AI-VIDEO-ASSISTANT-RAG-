# AI-VIDEO-ASSISTANT-RAG-


# AI Video Agent

Lightweight toolkit for extracting, transcribing, summarizing, and building vector stores from video/audio assets.


# Overview

AI Video Assistant RAG transforms unstructured video and audio files into searchable knowledge. Instead of manually watching long videos to find information, users can ask natural language questions and receive context-aware answers generated from the video's content.

The system combines speech recognition, text processing, vector embeddings, and large language models to create an efficient pipeline for information retrieval.

---


## Features
- Audio extraction and preprocessing: see [`Core.extractor`](Core/extractor.py) and [`utils.audio_processor`](utils/audio_processor.py).
- Speech-to-text/transcription: see [`Core.transcriber`](Core/transcriber.py).
- Retrieval-Augmented Generation pipeline: see [`Core.rag_engine`](Core/rag_engine.py).
- Summarization utilities: see [`Core.summary`](Core/summary.py).
- Vector store management: see [`Core.vector_store`](Core/vector_store.py) and the [`vector_db/`](vector_db) folder.


 # Features

- Audio extraction from video files
- Audio preprocessing for improved transcription quality
- Automatic speech-to-text transcription
- AI-powered text summarization
- Semantic search using vector embeddings
- Retrieval-Augmented Generation (RAG)
- Persistent vector database for efficient retrieval
- Natural language question answering over video content

---


## Quickstart

1. Install dependencies
```sh
pip install -r [REQUIREMENTS.txt](http://_vscodecontentref_/0)
# or use pyproject
pip install .
