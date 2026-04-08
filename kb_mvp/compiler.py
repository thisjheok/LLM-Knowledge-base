from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .concepts import (
    ConceptNote,
    ResolvedConcept,
    load_concept_note,
    load_concept_notes,
    normalize_extracted_concepts,
    resolve_concept,
    write_concept_note,
)
from .config import VaultPaths
from .extractors import ExtractedDocument, extract_document, save_normalized, SUPPORTED_EXTENSIONS
from .indexes import write_indexes
from .llm import CompiledNote, LLMClient
from .wiki_state import WikiNote, append_note, find_related_notes, load_wiki_snapshot


@dataclass
class CompileResult:
    document: ExtractedDocument
    note_path: Path
    normalized_path: Path


def compile_vault(paths: VaultPaths, llm_client: LLMClient) -> list[CompileResult]:
    results: list[CompileResult] = []
    snapshot = load_wiki_snapshot(paths)
    for raw_file in sorted(iter_raw_files(paths.raw)):
        document = extract_document(raw_file)
        normalized_path = save_normalized(document, paths.normalized)
        related_notes = find_related_notes(snapshot, document)
        compiled = llm_client.compile_document(document, related_notes=related_notes)
        resolved_concepts = resolve_related_concepts(paths.concepts, compiled.related_concepts)
        note_path = write_source_note(
            paths.sources,
            document,
            compiled.summary,
            compiled.key_points,
            resolved_concepts,
            compiled.related_existing_pages,
        )
        concept_paths = update_concepts(paths.concepts, llm_client, document, compiled, resolved_concepts)
        snapshot = append_note(
            snapshot,
            WikiNote(
                note_id=document.doc_id,
                title=document.title,
                note_type="source",
                path=note_path,
                summary=compiled.summary,
            ),
        )
        for concept_path in concept_paths:
            concept_note = load_concept_note(concept_path)
            if concept_note is None:
                continue
            snapshot = append_note(
                snapshot,
                WikiNote(
                    note_id=concept_note.concept_id,
                    title=concept_note.title,
                    note_type="concept",
                    path=concept_path,
                    summary=concept_note.summary,
                ),
            )
        results.append(CompileResult(document=document, note_path=note_path, normalized_path=normalized_path))
    write_indexes(paths.indexes, paths.sources, paths.concepts)
    return results


def iter_raw_files(raw_dir: Path) -> Iterable[Path]:
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def write_source_note(
    output_dir: Path,
    document: ExtractedDocument,
    summary: str,
    key_points: list[str],
    related_concepts: list[ResolvedConcept],
    related_existing_pages: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    note_path = output_dir / f"{document.doc_id}.md"
    concept_lines = (
        "\n".join(f"- [[{concept.concept_id}|{concept.title}]]" for concept in related_concepts)
        if related_concepts
        else "- None"
    )
    existing_page_lines = "\n".join(f"- [[{page_id}]]" for page_id in related_existing_pages) if related_existing_pages else "- None"
    heading_lines = "\n".join(f"- {heading}" for heading in document.headings) or "- None"
    key_point_lines = "\n".join(f"- {point}" for point in key_points) or "- None"
    snippet = shorten(document.text, limit=1200)
    note_path.write_text(
        "\n".join(
            [
                "---",
                f'title: "{escape_quotes(document.title)}"',
                f"source_type: {document.source_type}",
                f'source_path: "{escape_quotes(document.source_path)}"',
                f'source_url: "{escape_quotes(document.source_url or "")}"',
                f'extracted_at: "{document.extracted_at}"',
                "---",
                "",
                "## Summary",
                summary or "Summary unavailable.",
                "",
                "## Key Points",
                key_point_lines,
                "",
                "## Headings",
                heading_lines,
                "",
                "## Related Concepts",
                concept_lines,
                "",
                "## Related Existing Pages",
                existing_page_lines,
                "",
                "## Raw Snippet",
                "```text",
                snippet,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return note_path


def update_concepts(
    output_dir: Path,
    llm_client: LLMClient,
    document: ExtractedDocument,
    compiled: CompiledNote,
    resolved_concepts: list[ResolvedConcept],
) -> list[Path]:
    concept_paths: list[Path] = []
    for resolved in resolved_concepts:
        concept_name = resolved.title
        concept_id = resolved.concept_id
        concept_path = output_dir / f"{concept_id}.md"
        existing_concept = load_concept_note(concept_path)
        update = llm_client.update_concept(
            concept_name=concept_name,
            source_note_title=document.title,
            source_note_summary=compiled.summary,
            source_note_key_points=compiled.key_points,
            existing_concept=existing_concept,
        )
        related_sources = dedupe_keep_order(
            [
                *(existing_concept.related_sources if existing_concept else []),
                document.doc_id,
            ]
        )
        concept_note = ConceptNote(
            concept_id=concept_id,
            title=existing_concept.title if existing_concept else concept_name,
            summary=update.summary,
            key_ideas=update.key_ideas,
            related_sources=related_sources,
            open_questions=update.open_questions,
        )
        concept_paths.append(write_concept_note(output_dir, concept_note))
    return concept_paths


def resolve_related_concepts(output_dir: Path, concept_names: list[str]) -> list[ResolvedConcept]:
    existing_notes = load_concept_notes(output_dir)
    normalized_names = normalize_extracted_concepts(concept_names)
    resolved: list[ResolvedConcept] = []
    seen_ids: set[str] = set()
    for concept_name in normalized_names:
        candidate = resolve_concept(concept_name, existing_notes)
        if candidate.concept_id in seen_ids:
            continue
        seen_ids.add(candidate.concept_id)
        resolved.append(candidate)
        if not any(note.concept_id == candidate.concept_id for note in existing_notes):
            existing_notes.append(
                ConceptNote(
                    concept_id=candidate.concept_id,
                    title=candidate.title,
                    summary="Summary unavailable.",
                    key_ideas=[],
                    related_sources=[],
                    open_questions=[],
                )
            )
    return resolved


def shorten(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def escape_quotes(value: str) -> str:
    return value.replace('"', '\\"')


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
