# Web Clipper Knowledge Base MVP

This repository contains a minimal, local-first MVP for the workflow we discussed:

`web clip / HTML -> extract -> normalize -> compile into Markdown notes -> search / answer -> health check`

It is intentionally simple:

- `vault/00_raw` stores raw inputs from web clipping
- `kb_mvp` extracts and normalizes HTML / Markdown / text files
- compiled source notes are written into `vault/10_sources`
- a basic index is written into `vault/30_indexes`
- questions can be answered from compiled notes without a vector DB
- a health check reports missing summaries and broken wiki links

The current MVP requires an AI backend for compilation and question answering.
Right now the supported runtime is `OllamaLLMClient`, so you should have a local Ollama server available before running `compile` or `answer`.

## Quick Start

```powershell
python -m kb_mvp.cli init
python -m kb_mvp.cli --ollama-model qwen2.5:3b compile
python -m kb_mvp.cli --ollama-model qwen2.5:3b answer "What is this vault mainly about?"
python -m kb_mvp.cli health-check
```

## Using Ollama

If Ollama is already installed and the local server is running, switch the provider:

```powershell
python -m kb_mvp.cli --ollama-model qwen2.5:3b compile
python -m kb_mvp.cli --ollama-model qwen2.5:3b answer "What is this vault mainly about?"
```

On CPU-only machines, if compile feels slow, increase the timeout or reduce the input size:

```powershell
python -m kb_mvp.cli --ollama-model qwen2.5:3b --ollama-timeout 900 --compile-max-chars 4000 compile
```

You can also set environment variables:

```powershell
$env:OLLAMA_MODEL="qwen2.5:3b"
$env:OLLAMA_HOST="http://127.0.0.1:11434"
python -m kb_mvp.cli compile
```

If `ollama` is installed but the shell command is not on your `PATH`, this still works because the MVP talks to the local HTTP API rather than shelling out to the `ollama` binary.

## Vault Layout

```text
vault/
  00_raw/
  10_sources/
  20_concepts/
  30_indexes/
  40_outputs/
  90_logs/
```

## What The MVP Does

1. Reads `.html`, `.htm`, `.md`, and `.txt` files from `vault/00_raw`
2. Extracts title, text, headings, and basic metadata
3. Builds a normalized JSON artifact in `data/normalized`
4. Compiles each raw document into a Markdown source note
5. Updates a source index note
6. Answers questions by ranking notes lexically and synthesizing a concise response
7. Generates a health-check report

## Suggested Next Step

If you want this to become a stronger local knowledge base, the next step after Ollama integration is to add chunked compilation and concept-note generation on top of [`kb_mvp/llm.py`](C:\Users\User\Desktop\dev\LLM-knowledge-base\kb_mvp\llm.py).
