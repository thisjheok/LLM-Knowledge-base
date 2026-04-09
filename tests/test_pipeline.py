from __future__ import annotations

import shutil
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from kb_mvp.cli import ensure_sample_raw
from kb_mvp.compiler import compile_vault, resolve_related_concepts
from kb_mvp.concepts import normalize_extracted_concepts
from kb_mvp.config import build_paths, ensure_layout
from kb_mvp.health import run_health_check
from kb_mvp.llm import (
    CompiledNote,
    ConceptRelationshipReview,
    QueryPlan,
    ConceptUpdate,
    LLMClient,
    OllamaLLMClient,
)
from kb_mvp.maintenance import run_concept_review
from kb_mvp.search import search_notes, search_wiki


class StubLLMClient(LLMClient):
    def __init__(self) -> None:
        self.compile_related_notes: list[list[str]] = []

    def plan_query(self, question: str) -> QueryPlan:
        return QueryPlan(
            intent="broad",
            primary_concepts=[],
            source_focus_terms=[],
            prefer_direct_source=False,
            prefer_concepts=True,
        )

    def compile_document(self, document, related_notes=None) -> CompiledNote:
        self.compile_related_notes.append([note.note_id for note in (related_notes or [])])
        return CompiledNote(
            summary="A concise AI-generated summary for testing.",
            key_points=[
                "Web clipping stores source material first.",
                "Compiled notes help answer later questions.",
                "Health checks catch wiki issues.",
            ],
            related_concepts=["knowledge base", "web clipping", "health check"],
            related_existing_pages=["knowledge-base"] if related_notes else [],
        )

    def update_concept(
        self,
        concept_name: str,
        source_note_title: str,
        source_note_summary: str,
        source_note_key_points: list[str],
        existing_concept=None,
    ) -> ConceptUpdate:
        return ConceptUpdate(
            summary=f"{concept_name} is updated from {source_note_title}.",
            key_ideas=source_note_key_points[:3],
            open_questions=["What additional sources could refine this concept?"],
        )

    def answer_question(
        self,
        question: str,
        contexts: list[tuple[str, str]],
        *,
        max_contexts: int = 3,
        max_tokens: int = 384,
    ) -> str:
        return f"Stub answer for: {question}"

    def review_concept_relationship(
        self,
        left,
        right,
        *,
        heuristic_reasons: list[str],
    ) -> ConceptRelationshipReview:
        relationship = "likely_duplicate" if "openai" in left.concept_id and "openai" in right.concept_id else "related_but_distinct"
        return ConceptRelationshipReview(
            left_id=left.concept_id,
            left_title=left.title,
            right_id=right.concept_id,
            right_title=right.title,
            relationship=relationship,
            recommendation="Review manually before any merge.",
            reason="Stub review based on concept ids.",
            heuristic_reasons=heuristic_reasons,
        )


class PipelineTests(unittest.TestCase):
    def make_workspace_tempdir(self) -> Path:
        base_dir = Path.cwd() / ".tmp_tests"
        root = base_dir / f"case-{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        return root

    def test_compile_and_search_pipeline(self) -> None:
        root = self.make_workspace_tempdir()
        try:
            paths = build_paths(root)
            ensure_layout(paths)
            ensure_sample_raw(paths.raw)

            results = compile_vault(paths, StubLLMClient())

            self.assertEqual(len(results), 1)
            note_path = paths.sources / "sample-web-clip.md"
            self.assertTrue(note_path.exists())
            text = note_path.read_text(encoding="utf-8")
            self.assertIn("## Summary", text)
            self.assertIn("Personal Knowledge Bases with Web Clippers", text)
            self.assertIn("[[knowledge-base|Knowledge Base]]", text)

            hits = search_notes(paths.sources, "web clippers markdown")
            self.assertTrue(hits)
            self.assertEqual(hits[0].path.name, "sample-web-clip.md")
            wiki_hits = search_wiki(paths, "knowledge base web clipping")
            self.assertTrue(wiki_hits)
            self.assertIn("index", {hit.note_type for hit in wiki_hits})
            self.assertIn("concept", {hit.note_type for hit in wiki_hits})
            self.assertIn("source", {hit.note_type for hit in wiki_hits})

            concept_note = paths.concepts / "knowledge-base.md"
            self.assertTrue(concept_note.exists())
            self.assertIn("## Related Sources", concept_note.read_text(encoding="utf-8"))
            self.assertTrue((paths.indexes / "source-index.md").exists())
            self.assertTrue((paths.indexes / "concept-index.md").exists())
            self.assertTrue((paths.indexes / "overview.md").exists())
            overview_text = (paths.indexes / "overview.md").read_text(encoding="utf-8")
            self.assertIn("[[concept-index]]", overview_text)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_health_check_report_is_written(self) -> None:
        root = self.make_workspace_tempdir()
        try:
            paths = build_paths(root)
            ensure_layout(paths)
            ensure_sample_raw(paths.raw)
            compile_vault(paths, StubLLMClient())

            findings, report_path = run_health_check(paths)

            self.assertIsInstance(findings, list)
            self.assertTrue(report_path.exists())
            self.assertIn("# Health Check", report_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_ollama_empty_answer_raises_clear_error(self) -> None:
        client = OllamaLLMClient(model="gemma4:26b")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return (
                    b'{"response":"","done":true,"done_reason":"length"}'
                )

        with patch("kb_mvp.llm.request.urlopen", return_value=FakeResponse()):
            with self.assertRaises(RuntimeError) as ctx:
                client.answer_question("How does vLLM support OpenAI-compatible serving?", [])
        self.assertIn("empty response", str(ctx.exception))
        self.assertIn("token limit", str(ctx.exception))

    def test_compile_uses_existing_wiki_context(self) -> None:
        root = self.make_workspace_tempdir()
        try:
            paths = build_paths(root)
            ensure_layout(paths)
            ensure_sample_raw(paths.raw)
            concept_path = paths.concepts / "knowledge-base.md"
            concept_path.write_text(
                "\n".join(
                    [
                        "# Knowledge Base",
                        "",
                        "## Summary",
                        "Knowledge bases organize clipped material into maintained markdown notes.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            client = StubLLMClient()
            compile_vault(paths, client)

            self.assertTrue(client.compile_related_notes)
            self.assertIn("knowledge-base", client.compile_related_notes[0])
            note_text = (paths.sources / "sample-web-clip.md").read_text(encoding="utf-8")
            self.assertIn("## Related Existing Pages", note_text)
            self.assertIn("[[knowledge-base]]", note_text)
            concept_text = (paths.concepts / "web-clipping.md").read_text(encoding="utf-8")
            self.assertIn("[[sample-web-clip]]", concept_text)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_resolve_related_concepts_reuses_existing_similar_concept(self) -> None:
        root = self.make_workspace_tempdir()
        try:
            paths = build_paths(root)
            ensure_layout(paths)
            existing_concept = paths.concepts / "openai-compatible-server.md"
            existing_concept.write_text(
                "\n".join(
                    [
                        "# OpenAI-Compatible Server",
                        "",
                        "## Summary",
                        "Existing canonical concept.",
                        "",
                        "## Key Ideas",
                        "- Existing idea",
                        "",
                        "## Related Sources",
                        "- None",
                        "",
                        "## Open Questions",
                        "- None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            resolved = resolve_related_concepts(paths.concepts, ["OpenAI-Compatible Servers"])

            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0].concept_id, "openai-compatible-server")
            self.assertEqual(resolved[0].title, "OpenAI-Compatible Server")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_concept_review_writes_report(self) -> None:
        root = self.make_workspace_tempdir()
        try:
            paths = build_paths(root)
            ensure_layout(paths)
            (paths.concepts / "openai-compatible-server.md").write_text(
                "\n".join(
                    [
                        "# OpenAI-Compatible Server",
                        "",
                        "## Summary",
                        "Server concept.",
                        "",
                        "## Key Ideas",
                        "- Idea one",
                        "",
                        "## Related Sources",
                        "- [[source-a]]",
                        "",
                        "## Open Questions",
                        "- None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (paths.concepts / "openai-compatible-servers.md").write_text(
                "\n".join(
                    [
                        "# OpenAI-Compatible Servers",
                        "",
                        "## Summary",
                        "Server concept with plural naming.",
                        "",
                        "## Key Ideas",
                        "- Idea one",
                        "",
                        "## Related Sources",
                        "- [[source-a]]",
                        "",
                        "## Open Questions",
                        "- None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            reviews, report_path = run_concept_review(paths, StubLLMClient())

            self.assertEqual(len(reviews), 1)
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("# Concept Review", report_text)
            self.assertIn("Likely Duplicates", report_text)
            self.assertIn("[[openai-compatible-server|OpenAI-Compatible Server]]", report_text)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_normalize_extracted_concepts_prefers_narrow_canonical_names(self) -> None:
        normalized = normalize_extracted_concepts(
            [
                "LLM",
                "large language models (LLMs)",
                "Serving",
                "OpenAI-Compatible Servers",
                "chat api",
            ]
        )

        self.assertEqual(
            normalized,
            [
                "Large Language Model",
                "OpenAI-Compatible Server",
                "Chat API",
            ],
        )

    def test_search_wiki_prioritizes_direct_source_match(self) -> None:
        root = self.make_workspace_tempdir()
        try:
            paths = build_paths(root)
            ensure_layout(paths)
            ensure_sample_raw(paths.raw)
            (paths.raw / "Offline Inference - vLLM.md").write_text(
                "\n".join(
                    [
                        "# Offline Inference",
                        "",
                        "Offline inference is possible in your own code using vLLM's LLM class.",
                        "Use the LLM class to run local model inference.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            compile_vault(paths, StubLLMClient())

            hits = search_wiki(paths, "How is offline inference described in the current wiki?")

            self.assertTrue(hits)
            source_hits = [hit for hit in hits if hit.note_type == "source"]
            self.assertTrue(source_hits)
            self.assertEqual(source_hits[0].note_id, "offline-inference-vllm")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
