from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .concepts import ConceptNote, load_concept_notes
from .wiki_state import extract_note_summary


@dataclass(frozen=True)
class SourceIndexEntry:
    note_id: str
    title: str
    summary: str
    related_concepts: list[str]


def write_indexes(index_dir: Path, sources_dir: Path, concepts_dir: Path) -> list[Path]:
    index_dir.mkdir(parents=True, exist_ok=True)
    source_entries = load_source_entries(sources_dir)
    concept_entries = load_concept_notes(concepts_dir)
    written = [
        write_source_index(index_dir, source_entries),
        write_concept_index(index_dir, concept_entries),
        write_overview(index_dir, source_entries, concept_entries),
    ]
    return written


def load_source_entries(sources_dir: Path) -> list[SourceIndexEntry]:
    entries: list[SourceIndexEntry] = []
    if not sources_dir.exists():
        return entries
    for path in sorted(sources_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        entries.append(
            SourceIndexEntry(
                note_id=path.stem,
                title=extract_title(text) or path.stem.replace("-", " ").title(),
                summary=extract_note_summary(text),
                related_concepts=extract_related_concepts(text),
            )
        )
    return entries


def write_source_index(index_dir: Path, entries: list[SourceIndexEntry]) -> Path:
    index_path = index_dir / "source-index.md"
    lines = ["# Source Index", "", "Compiled source notes in the wiki.", ""]
    if not entries:
        lines.append("No compiled sources yet.")
    else:
        for entry in entries:
            concept_line = ", ".join(f"[[{concept_id}]]" for concept_id in entry.related_concepts) or "None"
            lines.append(f"## [[{entry.note_id}|{entry.title}]]")
            lines.append(entry.summary or "Summary unavailable.")
            lines.append(f"- Related concepts: {concept_line}")
            lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def write_concept_index(index_dir: Path, concepts: list[ConceptNote]) -> Path:
    index_path = index_dir / "concept-index.md"
    lines = ["# Concept Index", "", "Concept pages synthesized across sources.", ""]
    if not concepts:
        lines.append("No concept pages yet.")
    else:
        for concept in concepts:
            source_count = len(concept.related_sources)
            lines.append(f"## [[{concept.concept_id}|{concept.title}]]")
            lines.append(concept.summary or "Summary unavailable.")
            lines.append(f"- Related sources: {source_count}")
            lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def write_overview(
    index_dir: Path,
    source_entries: list[SourceIndexEntry],
    concepts: list[ConceptNote],
) -> Path:
    overview_path = index_dir / "overview.md"
    lines = [
        "# Wiki Overview",
        "",
        "Primary entry points for navigating the knowledge base.",
        "",
        f"- Sources: {len(source_entries)} notes in [[source-index]].",
        f"- Concepts: {len(concepts)} pages in [[concept-index]].",
        "",
        "## Suggested Starting Points",
        "",
    ]
    if concepts:
        for concept in sorted(concepts, key=lambda item: (-len(item.related_sources), item.title.lower()))[:5]:
            lines.append(
                f"- [[{concept.concept_id}|{concept.title}]] | {len(concept.related_sources)} related source(s)"
            )
    else:
        lines.append("- No concept pages yet.")
    lines.append("")
    if source_entries:
        lines.append("## Recent Source Notes")
        lines.append("")
        for entry in source_entries[:5]:
            lines.append(f"- [[{entry.note_id}|{entry.title}]]")
        lines.append("")
    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return overview_path


def extract_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith('title: "'):
            return line[len('title: "') : -1]
    return None


def extract_related_concepts(markdown: str) -> list[str]:
    body = extract_section(markdown, "Related Concepts")
    concepts: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        value = stripped[2:].strip()
        if not value or value == "None":
            continue
        concepts.append(strip_wikilink(value))
    return concepts


def extract_section(markdown: str, heading: str) -> str:
    marker = f"## {heading}\n"
    if marker not in markdown:
        return ""
    _, tail = markdown.split(marker, 1)
    next_marker = tail.find("\n## ")
    return tail[:next_marker].strip() if next_marker >= 0 else tail.strip()


def strip_wikilink(value: str) -> str:
    value = value.strip()
    if value.startswith("[[") and value.endswith("]]"):
        inner = value[2:-2]
        return inner.split("|", 1)[0].strip()
    return value
