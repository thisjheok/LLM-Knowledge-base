---
description: Ingest one raw source into the wiki
---

Ingest the raw source at `$ARGUMENTS` into this repository's knowledge base.

Follow `AGENTS.md` closely.

Required behavior:

- Read the raw file under `vault/00_raw/`.
- Inspect any related pages before editing.
- Update or create the relevant source note in `vault/10_sources/`.
- Update affected concept pages in `vault/20_concepts/` when appropriate.
- Update relevant index pages in `vault/30_indexes/`.
- Append a dated entry to `vault/90_logs/log.md`.
- If the source is garbled, incomplete, or has encoding problems, document that clearly instead of hiding it.

At the end, summarize what changed and what still needs human judgment.
