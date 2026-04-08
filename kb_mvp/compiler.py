from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import VaultPaths
from .extractors import ExtractedDocument, extract_document, save_normalized, SUPPORTED_EXTENSIONS
from .llm import LLMClient


@dataclass
class CompileResult:
    document: ExtractedDocument
    note_path: Path
    normalized_path: Path


def compile_vault(paths: VaultPaths, llm_client: LLMClient) -> list[CompileResult]:
    results: list[CompileResult] = []
    for raw_file in sorted(iter_raw_files(paths.raw)):
        document = extract_document(raw_file)
        normalized_path = save_normalized(document, paths.normalized)
        compiled = llm_client.compile_document(document)
        note_path = write_source_note(paths.sources, document, compiled.summary, compiled.key_points, compiled.related_concepts)
        results.append(CompileResult(document=document, note_path=note_path, normalized_path=normalized_path))
    write_source_index(paths.indexes, results)
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
    related_concepts: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    note_path = output_dir / f"{document.doc_id}.md"
    concept_lines = "\n".join(f"- {concept}" for concept in related_concepts) if related_concepts else "- None"
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


def write_source_index(index_dir: Path, results: list[CompileResult]) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "source-index.md"
    lines = ["# Source Index", ""]
    if not results:
        lines.append("No compiled sources yet.")
    else:
        for result in results:
            rel_note = Path("..") / "10_sources" / result.note_path.name
            lines.append(f"- [[{result.document.doc_id}]] | {result.document.title} | {rel_note.as_posix()}")
    lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def shorten(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def escape_quotes(value: str) -> str:
    return value.replace('"', '\\"')
