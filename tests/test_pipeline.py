from __future__ import annotations

import shutil
from pathlib import Path
import unittest
from uuid import uuid4

from kb_mvp.cli import ensure_sample_raw
from kb_mvp.compiler import compile_vault
from kb_mvp.config import build_paths, ensure_layout
from kb_mvp.health import run_health_check
from kb_mvp.search import search_notes


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

            results = compile_vault(paths)

            self.assertEqual(len(results), 1)
            note_path = paths.sources / "sample-web-clip.md"
            self.assertTrue(note_path.exists())
            text = note_path.read_text(encoding="utf-8")
            self.assertIn("## Summary", text)
            self.assertIn("Personal Knowledge Bases with Web Clippers", text)

            hits = search_notes(paths.sources, "web clippers markdown")
            self.assertTrue(hits)
            self.assertEqual(hits[0].path.name, "sample-web-clip.md")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_health_check_report_is_written(self) -> None:
        root = self.make_workspace_tempdir()
        try:
            paths = build_paths(root)
            ensure_layout(paths)
            ensure_sample_raw(paths.raw)
            compile_vault(paths)

            findings, report_path = run_health_check(paths)

            self.assertIsInstance(findings, list)
            self.assertTrue(report_path.exists())
            self.assertIn("# Health Check", report_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
