from __future__ import annotations

from dataclasses import dataclass
import json
import re
import socket
from urllib import error, request

from .extractors import ExtractedDocument


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


@dataclass
class CompiledNote:
    summary: str
    key_points: list[str]
    related_concepts: list[str]


class LLMClient:
    def compile_document(self, document: ExtractedDocument) -> CompiledNote:
        raise NotImplementedError

    def answer_question(self, question: str, contexts: list[tuple[str, str]]) -> str:
        raise NotImplementedError


class HeuristicLLMClient:
    """A deterministic stand-in so the MVP works without external APIs."""

    def compile_document(self, document: ExtractedDocument) -> CompiledNote:
        sentences = split_sentences(document.text)
        summary = " ".join(sentences[:3]).strip() or "Summary unavailable."
        key_points = []
        if document.description:
            key_points.append(document.description)
        key_points.extend(sentences[:5])
        key_points = dedupe_keep_order([point for point in key_points if point])[:5]
        related_concepts = top_keywords(document.text, limit=5)
        return CompiledNote(summary=summary, key_points=key_points, related_concepts=related_concepts)

    def answer_question(self, question: str, contexts: list[tuple[str, str]]) -> str:
        if not contexts:
            return "No relevant notes were found yet. Run the compile step first or add more raw documents."
        lines = [f"Question: {question}", ""]
        lines.append("Most relevant notes:")
        for title, snippet in contexts[:3]:
            lines.append(f"- {title}: {snippet}")
        return "\n".join(lines)


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

    def compile_document(self, document: ExtractedDocument) -> CompiledNote:
        prompt = build_compile_prompt(document, max_chars=self.compile_max_chars)
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
        if contains_cjk_text(summary, key_points, related_concepts):
            payload = self._rewrite_compiled_note_in_english(payload)
            summary = str(payload.get("summary", "")).strip() or "Summary unavailable."
            key_points = normalize_string_list(payload.get("key_points"))
            related_concepts = normalize_string_list(payload.get("related_concepts"))
        return CompiledNote(summary=summary, key_points=key_points[:5], related_concepts=related_concepts[:8])

    def answer_question(self, question: str, contexts: list[tuple[str, str]]) -> str:
        prompt = build_answer_prompt(question, contexts)
        return self._generate(
            prompt,
            temperature=0.2,
            json_mode=False,
            num_predict=220,
        ).strip()

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


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def top_keywords(text: str, limit: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower()):
        if token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


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


def build_compile_prompt(document: ExtractedDocument, max_chars: int) -> str:
    snippet = document.text[:max_chars]
    headings = "\n".join(f"- {heading}" for heading in document.headings[:20]) or "- None"
    return f"""You are compiling a web-clipped source into a markdown knowledge base.

Return valid JSON only with this exact shape:
{{
  "summary": "2-4 sentence summary",
  "key_points": ["point 1", "point 2"],
  "related_concepts": ["concept-a", "concept-b"]
}}

Rules:
- Write every field in English only.
- If the source text is not English, translate its meaning into natural English.
- Be concise and faithful to the source.
- `key_points` should have 3 to 5 items.
- `related_concepts` should be short noun phrases or keywords.
- Do not include markdown fences.

Document title: {document.title}
Source type: {document.source_type}
Source URL: {document.source_url or "unknown"}
Headings:
{headings}

Document text:
{snippet}
"""


def build_answer_prompt(question: str, contexts: list[tuple[str, str]]) -> str:
    context_lines: list[str] = []
    for title, snippet in contexts[:5]:
        context_lines.append(f"Title: {title}\nSnippet: {snippet}")
    joined_context = "\n\n".join(context_lines)
    return f"""You answer questions using compiled knowledge-base notes.

Rules:
- Answer in English only.
- Use only the provided note context.
- If the context is weak or incomplete, say so briefly.
- Answer in a concise, practical style.

Question:
{question}

Context:
{joined_context}
"""


def build_translate_compiled_note_prompt(payload: dict[str, object]) -> str:
    raw_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""Rewrite the following compiled-note JSON into natural English.

Return valid JSON only with this exact shape:
{{
  "summary": "2-4 sentence summary in English",
  "key_points": ["point 1 in English", "point 2 in English"],
  "related_concepts": ["english concept", "another english concept"]
}}

Rules:
- Translate every field into English only.
- Keep the meaning faithful.
- Keep `key_points` concise.
- Keep `related_concepts` short.
- Do not include markdown fences.

Input JSON:
{raw_json}
"""
