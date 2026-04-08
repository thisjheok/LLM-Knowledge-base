from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import VaultPaths


TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class SearchHit:
    path: Path
    title: str
    score: int
    note_type: str
    note_id: str
    answer_context: str


def search_notes(notes_dir: Path, query: str, limit: int = 5) -> list[SearchHit]:
    return search_markdown_dir(notes_dir, query, limit=limit, note_type="source")


def search_wiki(paths: VaultPaths, query: str) -> list[SearchHit]:
    index_hits = search_markdown_dir(paths.indexes, query, limit=3, note_type="index")
    concept_hits = search_markdown_dir(paths.concepts, query, limit=4, note_type="concept")

    concept_ids = collect_link_targets(index_hits) | {hit.note_id for hit in concept_hits}
    enriched_concepts = load_specific_hits(paths.concepts, concept_ids, query, note_type="concept")
    merged_concepts = dedupe_hits_by_id(concept_hits + enriched_concepts)
    merged_concepts.sort(key=lambda hit: (-hit.score, hit.title.lower()))
    merged_concepts = merged_concepts[:4]

    source_ids = collect_link_targets(index_hits) | collect_link_targets(merged_concepts)
    source_hits = load_specific_hits(paths.sources, source_ids, query, note_type="source")
    if len(source_hits) < 3:
        source_hits.extend(search_markdown_dir(paths.sources, query, limit=3, note_type="source"))
    merged_sources = dedupe_hits_by_id(source_hits)
    merged_sources.sort(key=lambda hit: (-hit.score, hit.title.lower()))
    merged_sources = merged_sources[:3]

    staged_hits: list[SearchHit] = []
    staged_hits.extend(index_hits[:1])
    staged_hits.extend(merged_concepts[:2])
    staged_hits.extend(merged_sources[:2])
    staged_hits.extend(index_hits[1:2])
    staged_hits.extend(merged_concepts[2:])
    staged_hits.extend(merged_sources[2:])
    return dedupe_hits_by_type_and_id(staged_hits)


def search_markdown_dir(directory: Path, query: str, *, limit: int, note_type: str) -> list[SearchHit]:
    query_tokens = tokenize(query)
    hits: list[SearchHit] = []
    if not directory.exists():
        return hits
    for path in sorted(directory.glob("*.md")):
        hit = build_search_hit(path, note_type=note_type, query_tokens=query_tokens)
        if hit is None:
            continue
        hits.append(hit)
    hits.sort(key=lambda hit: (-hit.score, hit.title.lower()))
    return hits[:limit]


def load_specific_hits(directory: Path, note_ids: set[str], query: str, *, note_type: str) -> list[SearchHit]:
    query_tokens = tokenize(query)
    hits: list[SearchHit] = []
    for note_id in sorted(note_ids):
        path = directory / f"{note_id}.md"
        if not path.exists():
            continue
        hit = build_search_hit(path, note_type=note_type, query_tokens=query_tokens)
        if hit is None:
            continue
        hits.append(hit)
    return hits


def build_search_hit(path: Path, *, note_type: str, query_tokens: set[str]) -> SearchHit | None:
    text = path.read_text(encoding="utf-8")
    haystack_tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    score = sum(haystack_tokens.count(token) for token in query_tokens)
    if score <= 0:
        return None
    title = extract_title(text) or extract_heading(text) or path.stem.replace("-", " ").title()
    return SearchHit(
        path=path,
        title=title,
        score=score,
        note_type=note_type,
        note_id=path.stem,
        answer_context=build_answer_context(text, note_type=note_type, title=title, note_id=path.stem),
    )


def build_answer_context(markdown: str, *, note_type: str, title: str, note_id: str) -> str:
    sections = []
    if note_type == "index":
        snippet = build_snippet(markdown, set())
        sections.append(f"Type: index")
        sections.append(f"Note ID: {note_id}")
        sections.append(f"Title: {title}")
        sections.append(f"Excerpt:\n{snippet}")
    elif note_type == "concept":
        summary = extract_section(markdown, "Summary")
        key_ideas = extract_section(markdown, "Key Ideas")
        related_sources = extract_section(markdown, "Related Sources")
        sections.append("Type: concept")
        sections.append(f"Note ID: {note_id}")
        sections.append(f"Title: {title}")
        if summary:
            sections.append(f"Summary:\n{summary}")
        if key_ideas:
            sections.append(f"Key Ideas:\n{key_ideas}")
        if related_sources:
            sections.append(f"Related Sources:\n{related_sources}")
    else:
        summary = extract_section(markdown, "Summary")
        key_points = extract_section(markdown, "Key Points")
        related_concepts = extract_section(markdown, "Related Concepts")
        sections.append("Type: source")
        sections.append(f"Note ID: {note_id}")
        sections.append(f"Title: {title}")
        if summary:
            sections.append(f"Summary:\n{summary}")
        if key_points:
            sections.append(f"Key Points:\n{key_points}")
        if related_concepts:
            sections.append(f"Related Concepts:\n{related_concepts}")
    return "\n\n".join(sections)


def collect_link_targets(hits: list[SearchHit]) -> set[str]:
    targets: set[str] = set()
    for hit in hits:
        text = hit.path.read_text(encoding="utf-8")
        targets.update(find_wikilinks(text))
    return targets


def find_wikilinks(markdown: str) -> set[str]:
    found: set[str] = set()
    for match in re.findall(r"\[\[([^\]]+)\]\]", markdown):
        target = match.split("|", 1)[0].strip()
        if target:
            found.add(target)
    return found


def dedupe_hits_by_id(hits: list[SearchHit]) -> list[SearchHit]:
    best: dict[str, SearchHit] = {}
    for hit in hits:
        existing = best.get(hit.note_id)
        if existing is None or hit.score > existing.score:
            best[hit.note_id] = hit
    return list(best.values())


def dedupe_hits_by_type_and_id(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[tuple[str, str]] = set()
    ordered: list[SearchHit] = []
    for hit in hits:
        key = (hit.note_type, hit.note_id)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(hit)
    return ordered


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def extract_title(markdown: str) -> str | None:
    match = re.search(r'^title:\s*"?(?P<title>.+?)"?$', markdown, re.MULTILINE)
    return match.group("title") if match else None


def extract_heading(markdown: str) -> str | None:
    match = re.search(r"^# (?P<title>.+?)$", markdown, re.MULTILINE)
    return match.group("title").strip() if match else None


def build_snippet(markdown: str, query_tokens: set[str], limit: int = 220) -> str:
    plain = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    plain = re.sub(r"^---.*?---", "", plain, flags=re.DOTALL)
    words = plain.split()
    if not words:
        return ""
    if not query_tokens:
        snippet = " ".join(words[:30]).strip()
    else:
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


def extract_section(markdown: str, heading: str) -> str:
    match = re.search(rf"## {re.escape(heading)}\n(?P<body>.*?)(?:\n## |\Z)", markdown, re.DOTALL)
    return match.group("body").strip() if match else ""
