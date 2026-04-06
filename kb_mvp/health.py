from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import VaultPaths


WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass
class HealthFinding:
    severity: str
    note: str
    detail: str


def run_health_check(paths: VaultPaths) -> tuple[list[HealthFinding], Path]:
    findings: list[HealthFinding] = []
    note_names = {path.stem for path in paths.sources.glob("*.md")} | {path.stem for path in paths.concepts.glob("*.md")}
    for path in sorted(list(paths.sources.glob("*.md")) + list(paths.concepts.glob("*.md"))):
        text = path.read_text(encoding="utf-8")
        if "## Summary" in text:
            summary_body = extract_section(text, "Summary")
            if not summary_body or summary_body.strip() in {"Summary unavailable.", "LLM summary pending."}:
                findings.append(HealthFinding("warning", path.name, "Missing or placeholder summary."))
        for wikilink in WIKILINK_RE.findall(text):
            target = wikilink.split("|", 1)[0].strip()
            if target and target not in note_names:
                findings.append(HealthFinding("warning", path.name, f"Broken wiki link: [[{target}]]"))
    report_path = write_report(paths.outputs, findings)
    return findings, report_path


def extract_section(markdown: str, heading: str) -> str:
    match = re.search(rf"## {re.escape(heading)}\n(?P<body>.*?)(?:\n## |\Z)", markdown, re.DOTALL)
    return match.group("body").strip() if match else ""


def write_report(output_dir: Path, findings: list[HealthFinding]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "health-check.md"
    lines = ["# Health Check", ""]
    if not findings:
        lines.append("No issues found.")
    else:
        for finding in findings:
            lines.append(f"- [{finding.severity}] {finding.note}: {finding.detail}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
