from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass
class SearchHit:
    path: Path
    title: str
    score: int
    snippet: str


def search_notes(notes_dir: Path, query: str, limit: int = 5) -> list[SearchHit]:
    query_tokens = set(token.lower() for token in TOKEN_RE.findall(query))
    hits: list[SearchHit] = []
    for path in sorted(notes_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = extract_title(text) or path.stem
        haystack_tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        score = sum(haystack_tokens.count(token) for token in query_tokens)
        if score <= 0:
            continue
        hits.append(
            SearchHit(
                path=path,
                title=title,
                score=score,
                snippet=build_snippet(text, query_tokens),
            )
        )
    hits.sort(key=lambda hit: (-hit.score, hit.title.lower()))
    return hits[:limit]


def extract_title(markdown: str) -> str | None:
    match = re.search(r'^title:\s*"?(?P<title>.+?)"?$', markdown, re.MULTILINE)
    return match.group("title") if match else None


def build_snippet(markdown: str, query_tokens: set[str], limit: int = 220) -> str:
    plain = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    plain = re.sub(r"^---.*?---", "", plain, flags=re.DOTALL)
    words = plain.split()
    if not words:
        return ""
    best_start = 0
    best_score = -1
    for start in range(0, max(len(words) - 30, 1)):
        window = words[start : start + 30]
        score = sum(1 for word in window if word.lower().strip(".,:;!?()[]") in query_tokens)
        if score > best_score:
            best_score = score
            best_start = start
    snippet = " ".join(words[best_start : best_start + 30]).strip()
    if len(snippet) > limit:
        snippet = snippet[: limit - 3].rstrip() + "..."
    return snippet
