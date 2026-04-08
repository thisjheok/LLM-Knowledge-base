from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import VaultPaths
from .extractors import ExtractedDocument


TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class WikiNote:
    note_id: str
    title: str
    note_type: str
    path: Path
    summary: str


@dataclass(frozen=True)
class WikiSnapshot:
    notes: list[WikiNote]


def load_wiki_snapshot(paths: VaultPaths) -> WikiSnapshot:
    notes: list[WikiNote] = []
    notes.extend(load_notes(paths.sources, "source"))
    notes.extend(load_notes(paths.concepts, "concept"))
    notes.extend(load_notes(paths.indexes, "index"))
    return WikiSnapshot(notes=notes)


def load_notes(directory: Path, note_type: str) -> list[WikiNote]:
    loaded: list[WikiNote] = []
    if not directory.exists():
        return loaded
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        loaded.append(
            WikiNote(
                note_id=path.stem,
                title=extract_title(text) or path.stem.replace("-", " ").title(),
                note_type=note_type,
                path=path,
                summary=extract_note_summary(text),
            )
        )
    return loaded


def find_related_notes(
    snapshot: WikiSnapshot,
    document: ExtractedDocument,
    *,
    limit: int = 6,
) -> list[WikiNote]:
    query_tokens = build_document_tokens(document)
    if not query_tokens:
        return []
    scored: list[tuple[int, WikiNote]] = []
    for note in snapshot.notes:
        note_tokens = build_note_tokens(note)
        if not note_tokens:
            continue
        overlap = len(query_tokens & note_tokens)
        if overlap <= 0:
            continue
        score = overlap
        if note.note_id == document.doc_id:
            continue
        if note.note_type == "concept":
            score += 2
        elif note.note_type == "source":
            score += 1
        scored.append((score, note))
    scored.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].note_id))
    return [note for _, note in scored[:limit]]


def append_note(snapshot: WikiSnapshot, note: WikiNote) -> WikiSnapshot:
    remaining = [existing for existing in snapshot.notes if existing.note_id != note.note_id]
    remaining.append(note)
    return WikiSnapshot(notes=remaining)


def build_related_context(notes: list[WikiNote]) -> str:
    if not notes:
        return "None."
    lines: list[str] = []
    for note in notes:
        lines.extend(
            [
                f"- page_id: {note.note_id}",
                f"  type: {note.note_type}",
                f"  title: {note.title}",
                f"  summary: {note.summary or 'No summary available.'}",
            ]
        )
    return "\n".join(lines)


def build_document_tokens(document: ExtractedDocument) -> set[str]:
    raw_tokens = tokenize(
        "\n".join(
            [
                document.title,
                *document.headings[:12],
                document.text[:2000],
            ]
        )
    )
    return {token for token in raw_tokens if len(token) >= 3}


def build_note_tokens(note: WikiNote) -> set[str]:
    raw_tokens = tokenize(f"{note.note_id}\n{note.title}\n{note.summary}")
    return {token for token in raw_tokens if len(token) >= 3}


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def extract_title(markdown: str) -> str | None:
    match = re.search(r'^title:\s*"?(?P<title>.+?)"?$', markdown, re.MULTILINE)
    return match.group("title") if match else None


def extract_note_summary(markdown: str) -> str:
    summary = extract_section(markdown, "Summary")
    if summary:
        return collapse_whitespace(summary)
    plain = re.sub(r"^---.*?---", "", markdown, flags=re.DOTALL).strip()
    return collapse_whitespace(plain[:300])


def extract_section(markdown: str, heading: str) -> str:
    match = re.search(rf"## {re.escape(heading)}\n(?P<body>.*?)(?:\n## |\Z)", markdown, re.DOTALL)
    return match.group("body").strip() if match else ""


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())
