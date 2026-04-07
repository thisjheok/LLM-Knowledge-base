# LLM Knowledge Base Agent Guide

This repository is a local-first knowledge base that turns raw source files into a persistent markdown wiki.

## Mission

- Treat `vault/00_raw` as immutable source material.
- Treat the rest of the vault as a maintained wiki that compounds over time.
- Prefer updating existing wiki pages over creating redundant new ones.
- Keep edits grounded in raw sources. When making a claim, preserve where it came from.

## Repo Layout

- `vault/00_raw/`: raw source documents captured from the web or written manually. Never edit these unless the user explicitly asks.
- `vault/10_sources/`: source notes compiled from raw documents.
- `vault/20_concepts/`: concept, topic, or entity pages synthesized across sources.
- `vault/30_indexes/`: navigational pages such as indexes and overviews.
- `vault/40_outputs/`: generated answers, reports, and temporary analysis artifacts.
- `vault/90_logs/`: append-only operational logs for ingest, query, and lint activity.
- `data/normalized/`: normalized machine-readable extraction artifacts.
- `kb_mvp/`: Python MVP code for extraction, compilation, search, and health checks.
- `tests/`: Python tests.

## Working Rules

- Read this file before making wiki changes.
- Prefer small, explainable edits across the wiki instead of one oversized page.
- Preserve user-authored content if you encounter it. Do not overwrite content blindly.
- Keep wiki links valid and use Obsidian-style wikilinks where they help navigation.
- If a source is low quality, incomplete, or appears garbled, say so in the updated note instead of pretending it is clean.
- If a source contains Korean or other non-English text, preserve meaning faithfully. Do not force English unless the user asked for it.

## Ingest Workflow

When the user asks to ingest a source:

1. Read the raw file from `vault/00_raw/`.
2. Inspect related source notes, concept pages, and indexes before editing.
3. Update or create the source note in `vault/10_sources/`.
4. Update any affected concept pages in `vault/20_concepts/`.
5. Update relevant index pages in `vault/30_indexes/`.
6. Append a dated entry to `vault/90_logs/log.md`.
7. Call out contradictions, uncertainty, encoding issues, or missing context explicitly.

## Query Workflow

When the user asks a question:

1. Start from indexes if they exist.
2. Read the most relevant source and concept pages.
3. Answer from the wiki, not from unsupported guesswork.
4. If the answer is durable and reusable, offer to save it into `vault/40_outputs/` or promote it into the wiki.

## Lint Workflow

When the user asks for a health check or lint pass:

- Check for broken wikilinks.
- Check for placeholder or missing summaries.
- Check for concept pages that should exist but do not.
- Check for orphan pages with no useful inbound references.
- Check for stale or contradictory claims when newer sources disagree.
- Check for encoding issues or obviously garbled extracted text.
- Write findings to `vault/40_outputs/health-check.md`.
- Append the lint run to `vault/90_logs/log.md`.

## Commands

- Run tests with `python -m unittest discover -s tests`.
- The current Python MVP entrypoint is `python -m kb_mvp.cli`.

## Notes For This Project

- The current codebase is still an MVP. It compiles raw files into source notes, but the long-term direction is a more agentic multi-page wiki maintainer.
- If you are asked to improve the architecture, prefer adding agent workflows and better wiki maintenance over adding vector infrastructure first.
