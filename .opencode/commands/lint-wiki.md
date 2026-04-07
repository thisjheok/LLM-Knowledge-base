---
description: Run a wiki maintenance pass
---

Run a maintenance and lint pass on this wiki.

Required behavior:

- Check for broken wikilinks.
- Check for missing or placeholder summaries.
- Check for orphan pages.
- Check for likely missing concept pages.
- Check for stale, conflicting, or superseded claims when visible from the current wiki.
- Check for encoding or extraction quality issues.
- Write findings to `vault/40_outputs/health-check.md`.
- Append a dated entry to `vault/90_logs/log.md`.

Summarize the highest priority findings at the end.
