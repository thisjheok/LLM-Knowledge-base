from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .compiler import compile_vault
from .config import build_paths, ensure_layout
from .health import run_health_check
from .llm import LLMClient, OllamaLLMClient
from .search import search_notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web clipper knowledge base MVP")
    parser.add_argument("--project-root", default=".", help="Project root containing the vault directory")
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", ""),
        help="Required Ollama model name, for example qwen2.5:3b",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Base URL for the Ollama HTTP API",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=int(os.environ.get("OLLAMA_TIMEOUT", "600")),
        help="Timeout in seconds for a single Ollama request",
    )
    parser.add_argument(
        "--compile-max-chars",
        type=int,
        default=int(os.environ.get("KB_COMPILE_MAX_CHARS", "6000")),
        help="Maximum number of source characters sent to Ollama for compile",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the vault layout and sample raw input")
    subparsers.add_parser("compile", help="Compile raw documents into source notes")

    answer = subparsers.add_parser("answer", help="Answer a question using compiled notes")
    answer.add_argument("question", help="Question to ask against the compiled notes")

    subparsers.add_parser("health-check", help="Run a simple vault health check")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    paths = build_paths(project_root)
    ensure_layout(paths)
    llm_client = build_llm_client(args)

    if args.command == "init":
        ensure_sample_raw(paths.raw)
        print(f"Initialized vault at {paths.root}")
        return 0

    if args.command == "compile":
        results = compile_vault(paths, llm_client)
        print(f"Compiled {len(results)} raw document(s)")
        for result in results:
            print(f"- {result.document.title} -> {result.note_path}")
        return 0

    if args.command == "answer":
        hits = search_notes(paths.sources, args.question)
        contexts = [(hit.title, hit.snippet) for hit in hits]
        answer = llm_client.answer_question(args.question, contexts)
        output_path = paths.outputs / "latest-answer.md"
        output_path.write_text(answer + "\n", encoding="utf-8")
        print(answer)
        print(f"\nSaved answer to {output_path}")
        return 0

    if args.command == "health-check":
        findings, report_path = run_health_check(paths)
        print(f"Health check complete: {len(findings)} finding(s)")
        print(f"Report: {report_path}")
        return 0

    parser.print_help()
    return 1


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    if not args.ollama_model:
        raise SystemExit(
            "--ollama-model is required. This MVP now requires an AI backend for compile and answer."
        )
    return OllamaLLMClient(
        model=args.ollama_model,
        host=args.ollama_host,
        timeout_seconds=args.ollama_timeout,
        compile_max_chars=args.compile_max_chars,
    )


def ensure_sample_raw(raw_dir: Path) -> None:
    sample_path = raw_dir / "sample-web-clip.html"
    if sample_path.exists():
        return
    sample_path.write_text(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Personal Knowledge Bases with Web Clippers</title>
    <meta name="description" content="A sample clipped article about using markdown vaults and LLMs together." />
    <meta property="og:url" content="https://example.com/personal-knowledge-bases" />
  </head>
  <body>
    <article>
      <h1>Personal Knowledge Bases with Web Clippers</h1>
      <p>Web clipping can turn scattered articles into a durable research archive.</p>
      <p>The core workflow is raw capture, structured notes, question answering, and health checks.</p>
      <h2>Why it works</h2>
      <p>Markdown notes are stable, diffable, and easy for language models to summarize.</p>
      <h2>Practical habit</h2>
      <p>Save source material first, then let a model compile source notes and concept pages later.</p>
    </article>
  </body>
</html>
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
