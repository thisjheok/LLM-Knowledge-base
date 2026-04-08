from __future__ import annotations

from dataclasses import dataclass
import json
import re
import socket
from urllib import error, request

from .concepts import ConceptNote
from .extractors import ExtractedDocument
from .wiki_state import WikiNote, build_related_context


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


@dataclass
class CompiledNote:
    summary: str
    key_points: list[str]
    related_concepts: list[str]
    related_existing_pages: list[str]


@dataclass
class ConceptUpdate:
    summary: str
    key_ideas: list[str]
    open_questions: list[str]


@dataclass
class ConceptRelationshipReview:
    left_id: str
    left_title: str
    right_id: str
    right_title: str
    relationship: str
    recommendation: str
    reason: str
    heuristic_reasons: list[str]


class LLMClient:
    def compile_document(self, document: ExtractedDocument, related_notes: list[WikiNote] | None = None) -> CompiledNote:
        raise NotImplementedError

    def update_concept(
        self,
        concept_name: str,
        source_note_title: str,
        source_note_summary: str,
        source_note_key_points: list[str],
        existing_concept: ConceptNote | None = None,
    ) -> ConceptUpdate:
        raise NotImplementedError

    def review_concept_relationship(
        self,
        left: ConceptNote,
        right: ConceptNote,
        *,
        heuristic_reasons: list[str],
    ) -> ConceptRelationshipReview:
        raise NotImplementedError

    def answer_question(
        self,
        question: str,
        contexts: list[tuple[str, str]],
        *,
        max_contexts: int = 3,
        max_tokens: int = 384,
    ) -> str:
        raise NotImplementedError


class OllamaLLMClient(LLMClient):
    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 600,
        compile_max_chars: int = 6000,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.compile_max_chars = compile_max_chars

    def compile_document(self, document: ExtractedDocument, related_notes: list[WikiNote] | None = None) -> CompiledNote:
        prompt = build_compile_prompt(document, max_chars=self.compile_max_chars, related_notes=related_notes or [])
        response = self._generate(
            prompt,
            temperature=0.2,
            json_mode=True,
            num_predict=320,
        )
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned a non-JSON compile response. "
                "Try a stronger model or inspect the raw output."
            ) from exc
        summary = str(payload.get("summary", "")).strip() or "Summary unavailable."
        key_points = normalize_string_list(payload.get("key_points"))
        related_concepts = normalize_string_list(payload.get("related_concepts"))
        related_existing_pages = normalize_string_list(payload.get("related_existing_pages"))
        if contains_cjk_text(summary, key_points, related_concepts):
            payload = self._rewrite_compiled_note_in_english(payload)
            summary = str(payload.get("summary", "")).strip() or "Summary unavailable."
            key_points = normalize_string_list(payload.get("key_points"))
            related_concepts = normalize_string_list(payload.get("related_concepts"))
            related_existing_pages = normalize_string_list(payload.get("related_existing_pages"))
        allowed_pages = {note.note_id for note in (related_notes or [])}
        filtered_existing_pages = [page_id for page_id in related_existing_pages if page_id in allowed_pages]
        return CompiledNote(
            summary=summary,
            key_points=key_points[:5],
            related_concepts=related_concepts[:8],
            related_existing_pages=filtered_existing_pages[:6],
        )

    def answer_question(
        self,
        question: str,
        contexts: list[tuple[str, str]],
        *,
        max_contexts: int = 3,
        max_tokens: int = 384,
    ) -> str:
        prompt = build_answer_prompt(question, contexts, max_contexts=max_contexts)
        return self._generate(
            prompt,
            temperature=0.2,
            json_mode=False,
            num_predict=max_tokens,
        ).strip()

    def update_concept(
        self,
        concept_name: str,
        source_note_title: str,
        source_note_summary: str,
        source_note_key_points: list[str],
        existing_concept: ConceptNote | None = None,
    ) -> ConceptUpdate:
        prompt = build_concept_update_prompt(
            concept_name=concept_name,
            source_note_title=source_note_title,
            source_note_summary=source_note_summary,
            source_note_key_points=source_note_key_points,
            existing_concept=existing_concept,
        )
        response = self._generate(
            prompt,
            temperature=0.2,
            json_mode=True,
            num_predict=320,
        )
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON concept update response.") from exc
        summary = str(payload.get("summary", "")).strip() or "Summary unavailable."
        key_ideas = normalize_string_list(payload.get("key_ideas"))[:5]
        open_questions = normalize_string_list(payload.get("open_questions"))[:4]
        if contains_cjk_text(summary, key_ideas, open_questions):
            payload = self._rewrite_concept_update_in_english(payload)
            summary = str(payload.get("summary", "")).strip() or "Summary unavailable."
            key_ideas = normalize_string_list(payload.get("key_ideas"))[:5]
            open_questions = normalize_string_list(payload.get("open_questions"))[:4]
        return ConceptUpdate(summary=summary, key_ideas=key_ideas, open_questions=open_questions)

    def review_concept_relationship(
        self,
        left: ConceptNote,
        right: ConceptNote,
        *,
        heuristic_reasons: list[str],
    ) -> ConceptRelationshipReview:
        prompt = build_concept_relationship_review_prompt(
            left=left,
            right=right,
            heuristic_reasons=heuristic_reasons,
        )
        response = self._generate(
            prompt,
            temperature=0.1,
            json_mode=True,
            num_predict=260,
        )
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON concept relationship review.") from exc
        relationship = str(payload.get("relationship", "")).strip().lower()
        if relationship not in {"likely_duplicate", "related_but_distinct", "unclear"}:
            relationship = "unclear"
        recommendation = str(payload.get("recommendation", "")).strip() or "Review manually."
        reason = str(payload.get("reason", "")).strip() or "The relationship is unclear from the available concept pages."
        return ConceptRelationshipReview(
            left_id=left.concept_id,
            left_title=left.title,
            right_id=right.concept_id,
            right_title=right.title,
            relationship=relationship,
            recommendation=recommendation,
            reason=reason,
            heuristic_reasons=heuristic_reasons,
        )

    def _generate(self, prompt: str, temperature: float, json_mode: bool, num_predict: int) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json" if json_mode else "",
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            if exc.code == 404 and "model" in error_body and "not found" in error_body:
                raise RuntimeError(
                    f"Ollama model `{self.model}` was not found at {self.host}. "
                    "Run `ollama list` to see installed models and use one of those names."
                ) from exc
            if exc.code == 404:
                raise RuntimeError(
                    f"Ollama returned HTTP 404 from {self.host}/api/generate. "
                    "The server is reachable, but this does not look like a native Ollama generate endpoint."
                ) from exc
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code}: {error_body or exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout_seconds} seconds. "
                "On CPU-only setups, try a smaller model, a shorter document, or a larger timeout."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout_seconds} seconds. "
                "On CPU-only setups, try a smaller model, a shorter document, or a larger timeout."
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. "
                "Make sure the Ollama app or `ollama serve` is running."
            ) from exc
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned malformed JSON.") from exc
        response_text = payload.get("response")
        if not isinstance(response_text, str):
            raise RuntimeError("Ollama response did not include a text `response` field.")
        if not response_text.strip():
            done_reason = payload.get("done_reason", "unknown")
            if done_reason == "length":
                raise RuntimeError(
                    "Ollama returned an empty response after hitting the token limit. "
                    f"Model: {self.model}. Requested num_predict={num_predict}. "
                    "Try a larger token limit, a smaller prompt, or a different model."
                )
            raise RuntimeError(
                "Ollama returned an empty response. "
                f"Model: {self.model}. done_reason={done_reason}."
            )
        return response_text

    def _rewrite_compiled_note_in_english(self, payload: dict[str, object]) -> dict[str, object]:
        prompt = build_translate_compiled_note_prompt(payload)
        response = self._generate(
            prompt,
            temperature=0.1,
            json_mode=True,
            num_predict=320,
        )
        try:
            rewritten = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON translation response.") from exc
        return rewritten

    def _rewrite_concept_update_in_english(self, payload: dict[str, object]) -> dict[str, object]:
        prompt = build_translate_concept_update_prompt(payload)
        response = self._generate(
            prompt,
            temperature=0.1,
            json_mode=True,
            num_predict=320,
        )
        try:
            rewritten = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON concept translation response.") from exc
        return rewritten


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned:
            normalized.append(cleaned)
    return dedupe_keep_order(normalized)


def contains_cjk_text(summary: str, key_points: list[str], related_concepts: list[str]) -> bool:
    combined = "\n".join([summary, *key_points, *related_concepts])
    return bool(CJK_RE.search(combined))


def build_compile_prompt(
    document: ExtractedDocument,
    *,
    max_chars: int,
    related_notes: list[WikiNote],
) -> str:
    snippet = document.text[:max_chars]
    headings = "\n".join(f"- {heading}" for heading in document.headings[:20]) or "- None"
    related_context = build_related_context(related_notes)
    return f"""You are compiling a web-clipped source into a markdown knowledge base.

Return valid JSON only with this exact shape:
{{
  "summary": "2-4 sentence summary",
  "key_points": ["point 1", "point 2"],
  "related_concepts": ["concept-a", "concept-b"],
  "related_existing_pages": ["page-id-1", "page-id-2"]
}}

Rules:
- Write every field in English only.
- If the source text is not English, translate its meaning into natural English.
- Be concise and faithful to the source.
- `key_points` should have 3 to 5 items.
- `related_concepts` should contain 2 to 6 narrow, durable concept names.
- Prefer stable canonical names over ad hoc wording.
- Prefer the smallest concept that is still meaningful in the source.
- Avoid vague or overly broad labels such as "API", "Serving", "Library", "Guide", "Workflow", or "Reference" unless the source is specifically about that concept itself.
- Reuse an existing concept name from the provided wiki context when it already matches the same idea.
- Prefer singular concept names unless the plural form is the standard term.
- Expand bare abbreviations into canonical concept names when possible, for example `LLM` -> `Large Language Model`.
- `related_existing_pages` must contain only `page_id` values from the provided existing wiki context.
- Use `related_existing_pages` to identify existing notes that should be reviewed or updated together with this source.
- Do not include markdown fences.

Document title: {document.title}
Source type: {document.source_type}
Source URL: {document.source_url or "unknown"}
Headings:
{headings}

Existing wiki context:
{related_context}

Document text:
{snippet}
"""


def build_answer_prompt(
    question: str,
    contexts: list[tuple[str, str]],
    *,
    max_contexts: int = 3,
) -> str:
    context_lines: list[str] = []
    for title, snippet in contexts[:max_contexts]:
        context_lines.append(f"Title: {title}\nSnippet: {snippet}")
    joined_context = "\n\n".join(context_lines)
    return f"""You answer questions using a maintained markdown knowledge base.

Your job is to produce a faithful, information-dense answer grounded only in the provided context.

Rules:
- Answer in English only.
- Use only the provided context. Do not use outside knowledge.
- The provided context may include index notes, concept pages, and source notes. Treat index notes as navigation hints, concept pages as synthesized topic summaries, and source notes as supporting detail.
- If the context is incomplete, weak, or ambiguous, say so explicitly.
- Prefer specific facts over vague summaries.
- Preserve the specificity of the source context.
- If the context contains concrete items such as names, commands, APIs, steps, dates, categories, limitations, or comparisons, include them explicitly instead of replacing them with generic wording.
- Do not collapse supported items into broad phrases like "and more" when the context names them concretely.
- Do not invent missing details.
- Keep the answer concise, but do not omit important specifics that are directly supported by the context.
- When multiple notes contribute, synthesize them into one coherent answer.
- If the question asks "how", explain the mechanism, workflow, or usage steps described in the context, not just the purpose.
- If the question asks for comparison, state the compared items and their differences explicitly.
- End with a short `Basis:` line listing the most relevant note titles.

Question:
{question}

Context:
{joined_context}

Write the answer in this format:

Answer:
<grounded answer>

Basis:
- <note title 1>
- <note title 2>
"""


def build_translate_compiled_note_prompt(payload: dict[str, object]) -> str:
    raw_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""Rewrite the following compiled-note JSON into natural English.

Return valid JSON only with this exact shape:
{{
  "summary": "2-4 sentence summary in English",
  "key_points": ["point 1 in English", "point 2 in English"],
  "related_concepts": ["english concept", "another english concept"],
  "related_existing_pages": ["page-id-1", "page-id-2"]
}}

Rules:
- Translate every field into English only.
- Keep the meaning faithful.
- Keep `key_points` concise.
- Keep `related_concepts` short.
- Preserve `related_existing_pages` exactly when present.
- Do not include markdown fences.

Input JSON:
{raw_json}
"""


def build_concept_update_prompt(
    *,
    concept_name: str,
    source_note_title: str,
    source_note_summary: str,
    source_note_key_points: list[str],
    existing_concept: ConceptNote | None,
) -> str:
    key_point_lines = "\n".join(f"- {item}" for item in source_note_key_points) or "- None"
    existing_summary = existing_concept.summary if existing_concept else "None."
    existing_key_ideas = "\n".join(f"- {item}" for item in (existing_concept.key_ideas if existing_concept else [])) or "- None"
    existing_questions = "\n".join(f"- {item}" for item in (existing_concept.open_questions if existing_concept else [])) or "- None"
    return f"""You are maintaining a concept page in a markdown knowledge base.

Return valid JSON only with this exact shape:
{{
  "summary": "2-4 sentence concept summary",
  "key_ideas": ["idea 1", "idea 2"],
  "open_questions": ["question 1", "question 2"]
}}

Rules:
- Write every field in English only.
- Synthesize the concept using the existing concept page when present and the new source note update.
- Preserve durable information already supported by the existing concept.
- Add new information from the source note when it sharpens or expands the concept.
- If uncertainty or missing context is visible, capture it in `open_questions`.
- Keep `key_ideas` concise and factual.
- Do not mention information not present in the supplied context.
- Do not include markdown fences.

Concept name: {concept_name}

Existing concept summary:
{existing_summary}

Existing concept key ideas:
{existing_key_ideas}

Existing concept open questions:
{existing_questions}

New source note title: {source_note_title}
New source note summary:
{source_note_summary}

New source note key points:
{key_point_lines}
"""


def build_translate_concept_update_prompt(payload: dict[str, object]) -> str:
    raw_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""Rewrite the following concept-update JSON into natural English.

Return valid JSON only with this exact shape:
{{
  "summary": "2-4 sentence concept summary in English",
  "key_ideas": ["idea 1 in English", "idea 2 in English"],
  "open_questions": ["question 1 in English", "question 2 in English"]
}}

Rules:
- Translate every field into English only.
- Keep the meaning faithful.
- Keep `key_ideas` concise.
- Keep `open_questions` short and specific.
- Do not include markdown fences.

Input JSON:
{raw_json}
"""


def build_concept_relationship_review_prompt(
    *,
    left: ConceptNote,
    right: ConceptNote,
    heuristic_reasons: list[str],
) -> str:
    left_sources = "\n".join(f"- {item}" for item in left.related_sources) or "- None"
    right_sources = "\n".join(f"- {item}" for item in right.related_sources) or "- None"
    left_key_ideas = "\n".join(f"- {item}" for item in left.key_ideas) or "- None"
    right_key_ideas = "\n".join(f"- {item}" for item in right.key_ideas) or "- None"
    heuristic_lines = "\n".join(f"- {item}" for item in heuristic_reasons) or "- None"
    return f"""You are reviewing two concept pages in a markdown knowledge base.

Return valid JSON only with this exact shape:
{{
  "relationship": "likely_duplicate | related_but_distinct | unclear",
  "recommendation": "short action recommendation",
  "reason": "1-3 sentence explanation"
}}

Rules:
- Write every field in English only.
- Use only the supplied concept page data.
- `likely_duplicate` means the two pages probably represent the same concept and should likely be merged.
- `related_but_distinct` means the pages are connected but should remain separate.
- `unclear` means the evidence is insufficient or mixed.
- Keep the recommendation short and practical.
- Do not include markdown fences.

Heuristic signals:
{heuristic_lines}

Left concept:
- id: {left.concept_id}
- title: {left.title}
- summary: {left.summary}
- key ideas:
{left_key_ideas}
- related sources:
{left_sources}

Right concept:
- id: {right.concept_id}
- title: {right.title}
- summary: {right.summary}
- key ideas:
{right_key_ideas}
- related sources:
{right_sources}
"""
