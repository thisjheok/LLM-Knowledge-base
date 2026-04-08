from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import json
import re
from urllib.parse import urlparse


SUPPORTED_EXTENSIONS = {".html", ".htm", ".md", ".txt"}
WHITESPACE_RE = re.compile(r"\s+")
META_RE = re.compile(
    r'<meta[^>]+(?:name|property)=["\'](?P<key>[^"\']+)["\'][^>]+content=["\'](?P<value>[^"\']+)["\']',
    re.IGNORECASE,
)
TITLE_RE = re.compile(r"<title>(?P<title>.*?)</title>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"^\s{0,3}#{1,2}\s+(?P<heading>.+?)\s*$")


@dataclass
class ExtractedDocument:
    doc_id: str
    title: str
    source_type: str
    source_path: str
    source_url: str | None
    extracted_at: str
    headings: list[str]
    text: str
    description: str | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._current_tag: str | None = None
        self.text_parts: list[str] = []
        self.headings: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._current_tag = tag.lower()
        if self._current_tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        self._current_tag = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        normalized = WHITESPACE_RE.sub(" ", data).strip()
        if not normalized:
            return
        if self._current_tag == "title":
            self._title_parts.append(normalized)
        if self._current_tag in {"h1", "h2", "h3"}:
            self.headings.append(normalized)
        self.text_parts.append(normalized)

    @property
    def title(self) -> str | None:
        if not self._title_parts:
            return None
        return " ".join(self._title_parts).strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def extract_document(path: Path) -> ExtractedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {suffix}")
    if suffix in {".html", ".htm"}:
        return extract_html(path)
    return extract_markdown_or_text(path)


def extract_html(path: Path) -> ExtractedDocument:
    raw_html = path.read_text(encoding="utf-8")
    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    parser.close()

    meta = {match.group("key").lower(): match.group("value").strip() for match in META_RE.finditer(raw_html)}
    title_match = TITLE_RE.search(raw_html)
    title = (
        parser.title
        or (title_match.group("title").strip() if title_match else None)
        or meta.get("og:title")
        or path.stem.replace("-", " ").replace("_", " ").title()
    )
    source_url = meta.get("og:url") or meta.get("twitter:url") or infer_url_from_title(title)
    description = meta.get("description") or meta.get("og:description")
    text = "\n".join(parser.text_parts)
    return ExtractedDocument(
        doc_id=slugify(path.stem),
        title=title,
        source_type="html",
        source_path=str(path),
        source_url=source_url,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        headings=dedupe_keep_order(parser.headings),
        text=text,
        description=description,
    )


def extract_markdown_or_text(path: Path) -> ExtractedDocument:
    raw_text = path.read_text(encoding="utf-8")
    headings = [match.group("heading") for match in HEADING_RE.finditer(raw_text)]
    title = headings[0] if headings else path.stem.replace("-", " ").replace("_", " ").title()
    return ExtractedDocument(
        doc_id=slugify(path.stem),
        title=title,
        source_type="markdown" if path.suffix.lower() == ".md" else "text",
        source_path=str(path),
        source_url=None,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        headings=dedupe_keep_order(headings),
        text=raw_text.strip(),
        description=None,
    )


def save_normalized(document: ExtractedDocument, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{document.doc_id}.json"
    output_path.write_text(document.to_json(), encoding="utf-8")
    return output_path


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def infer_url_from_title(title: str) -> str | None:
    try:
        parsed = urlparse(title)
        return title if parsed.scheme and parsed.netloc else None
    except ValueError:
        return None
