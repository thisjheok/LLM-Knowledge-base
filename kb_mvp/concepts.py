from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ConceptNote:
    concept_id: str
    title: str
    summary: str
    key_ideas: list[str]
    related_sources: list[str]
    open_questions: list[str]


@dataclass(frozen=True)
class ResolvedConcept:
    concept_id: str
    title: str
    matched_existing: bool


GENERIC_SINGLE_TOKEN_CONCEPTS = {
    "api",
    "apis",
    "feature",
    "features",
    "guide",
    "guides",
    "library",
    "libraries",
    "reference",
    "references",
    "server",
    "servers",
    "serving",
    "system",
    "systems",
    "workflow",
    "workflows",
}

CANONICAL_CONCEPT_NAMES = {
    "llm": "Large Language Model",
    "llms": "Large Language Model",
    "large language model llm": "Large Language Model",
    "large language models llms": "Large Language Model",
    "openai compatible server": "OpenAI-Compatible Server",
    "openai compatible servers": "OpenAI-Compatible Server",
    "openai api": "OpenAI API",
    "chat api": "Chat API",
    "completions api": "Completions API",
    "embeddings api": "Embeddings API",
    "transcriptions api": "Transcriptions API",
    "translation api": "Translation API",
}


def concept_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return slug or "concept"


def normalize_concept_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip()
    cleaned = re.sub(r"\s*\((?:[A-Za-z]{2,}|[A-Za-z]{2,}s)\)\s*$", "", cleaned).strip()
    return cleaned


def normalize_extracted_concepts(names: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        candidate = canonicalize_concept_name(raw_name)
        if not candidate:
            continue
        key = concept_match_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)
    return normalized


def canonicalize_concept_name(name: str) -> str | None:
    cleaned = normalize_concept_name(name)
    if not cleaned:
        return None
    key = concept_match_key(cleaned)
    if not key:
        return None
    canonical = CANONICAL_CONCEPT_NAMES.get(key, cleaned)
    if is_overly_generic_concept(canonical):
        return None
    return apply_display_casing(canonical)


def resolve_concept(name: str, existing_notes: list[ConceptNote]) -> ResolvedConcept:
    normalized_name = normalize_concept_name(name)
    candidate_key = concept_match_key(normalized_name)
    candidate_slug = concept_slug(normalized_name)
    for existing in existing_notes:
        if candidate_slug == existing.concept_id:
            return ResolvedConcept(existing.concept_id, existing.title, True)
        if candidate_key == concept_match_key(existing.title):
            return ResolvedConcept(existing.concept_id, existing.title, True)
        if singularize_slug(candidate_slug) == singularize_slug(existing.concept_id):
            return ResolvedConcept(existing.concept_id, existing.title, True)
    return ResolvedConcept(candidate_slug, normalized_name or name.strip() or "Concept", False)


def load_concept_note(path: Path) -> ConceptNote | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    title = extract_heading(text) or path.stem.replace("-", " ").title()
    summary = extract_section(text, "Summary") or "Summary unavailable."
    key_ideas = extract_list_section(text, "Key Ideas")
    related_sources = [strip_wikilink(item) for item in extract_list_section(text, "Related Sources")]
    open_questions = extract_list_section(text, "Open Questions")
    return ConceptNote(
        concept_id=path.stem,
        title=title,
        summary=summary,
        key_ideas=key_ideas,
        related_sources=related_sources,
        open_questions=open_questions,
    )


def write_concept_note(output_dir: Path, note: ConceptNote) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    note_path = output_dir / f"{note.concept_id}.md"
    key_idea_lines = "\n".join(f"- {idea}" for idea in note.key_ideas) if note.key_ideas else "- None"
    source_lines = "\n".join(f"- [[{source_id}]]" for source_id in note.related_sources) if note.related_sources else "- None"
    open_question_lines = "\n".join(f"- {item}" for item in note.open_questions) if note.open_questions else "- None"
    note_path.write_text(
        "\n".join(
            [
                f"# {note.title}",
                "",
                "## Summary",
                note.summary or "Summary unavailable.",
                "",
                "## Key Ideas",
                key_idea_lines,
                "",
                "## Related Sources",
                source_lines,
                "",
                "## Open Questions",
                open_question_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return note_path


def extract_heading(markdown: str) -> str | None:
    match = re.search(r"^# (?P<title>.+?)$", markdown, re.MULTILINE)
    return match.group("title").strip() if match else None


def extract_section(markdown: str, heading: str) -> str:
    match = re.search(rf"## {re.escape(heading)}\n(?P<body>.*?)(?:\n## |\Z)", markdown, re.DOTALL)
    return match.group("body").strip() if match else ""


def extract_list_section(markdown: str, heading: str) -> list[str]:
    body = extract_section(markdown, heading)
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        value = stripped[2:].strip()
        if value and value != "None":
            lines.append(value)
    return lines


def strip_wikilink(value: str) -> str:
    match = re.fullmatch(r"\[\[(?P<target>[^\]|]+)(?:\|[^\]]+)?\]\]", value.strip())
    return match.group("target") if match else value.strip()


def load_concept_notes(directory: Path) -> list[ConceptNote]:
    notes: list[ConceptNote] = []
    if not directory.exists():
        return notes
    for path in sorted(directory.glob("*.md")):
        note = load_concept_note(path)
        if note is not None:
            notes.append(note)
    return notes


def concept_match_key(name: str) -> str:
    normalized = normalize_concept_name(name).lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    tokens = [singularize_token(token) for token in normalized.split() if token]
    return " ".join(tokens)


def is_overly_generic_concept(name: str) -> bool:
    key = concept_match_key(name)
    tokens = key.split()
    if not tokens:
        return True
    return len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_TOKEN_CONCEPTS


def apply_display_casing(name: str) -> str:
    words = re.split(r"(\s+|-)", name)
    formatted: list[str] = []
    for word in words:
        if not word or word.isspace() or word == "-":
            formatted.append(word)
            continue
        lowered = word.lower()
        if lowered in {"api", "llm", "cpu", "gpu", "tpu", "ibm", "rocm"}:
            formatted.append(lowered.upper())
        elif lowered == "openai":
            formatted.append("OpenAI")
        else:
            formatted.append(word[0].upper() + word[1:] if word else word)
    return "".join(formatted)


def singularize_slug(slug: str) -> str:
    return "-".join(singularize_token(part) for part in slug.split("-") if part)


def singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
