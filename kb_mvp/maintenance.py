from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .concepts import ConceptNote, concept_match_key, load_concept_notes, singularize_slug
from .config import VaultPaths
from .llm import ConceptRelationshipReview, LLMClient


@dataclass(frozen=True)
class ConceptReviewCandidate:
    left: ConceptNote
    right: ConceptNote
    score: int
    reasons: list[str]


def run_concept_review(paths: VaultPaths, llm_client: LLMClient) -> tuple[list[ConceptRelationshipReview], Path]:
    concepts = load_concept_notes(paths.concepts)
    candidates = find_review_candidates(concepts)
    reviews: list[ConceptRelationshipReview] = []
    for candidate in candidates:
        reviews.append(
            llm_client.review_concept_relationship(
                candidate.left,
                candidate.right,
                heuristic_reasons=candidate.reasons,
            )
        )
    report_path = write_concept_review_report(paths.outputs, reviews)
    return reviews, report_path


def find_review_candidates(concepts: list[ConceptNote], *, limit: int = 12) -> list[ConceptReviewCandidate]:
    candidates: list[ConceptReviewCandidate] = []
    for index, left in enumerate(concepts):
        for right in concepts[index + 1 :]:
            score, reasons = score_concept_pair(left, right)
            if score <= 0:
                continue
            candidates.append(
                ConceptReviewCandidate(
                    left=left,
                    right=right,
                    score=score,
                    reasons=reasons,
                )
            )
    candidates.sort(
        key=lambda item: (
            -item.score,
            item.left.title.lower(),
            item.right.title.lower(),
        )
    )
    return candidates[:limit]


def score_concept_pair(left: ConceptNote, right: ConceptNote) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    left_key = concept_match_key(left.title)
    right_key = concept_match_key(right.title)
    if left_key == right_key:
        score += 5
        reasons.append("Normalized titles match.")
    if singularize_slug(left.concept_id) == singularize_slug(right.concept_id):
        score += 4
        reasons.append("Concept slugs differ only by small normalization changes.")
    source_overlap = len(set(left.related_sources) & set(right.related_sources))
    if source_overlap:
        score += min(source_overlap, 3)
        reasons.append(f"Related source overlap: {source_overlap}.")
    summary_overlap = summary_token_overlap(left.summary, right.summary)
    if summary_overlap >= 4:
        score += 2
        reasons.append(f"Summary token overlap: {summary_overlap}.")
    elif summary_overlap >= 2:
        score += 1
        reasons.append(f"Some summary token overlap: {summary_overlap}.")
    if score >= 3 and not reasons:
        reasons.append("Multiple weak signals suggest a review is worthwhile.")
    if score < 3:
        return 0, []
    return score, reasons


def summary_token_overlap(left: str, right: str) -> int:
    left_tokens = {token for token in concept_match_key(left).split() if len(token) >= 4}
    right_tokens = {token for token in concept_match_key(right).split() if len(token) >= 4}
    return len(left_tokens & right_tokens)


def write_concept_review_report(output_dir: Path, reviews: list[ConceptRelationshipReview]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "concept-review.md"
    lines = ["# Concept Review", "", "Report-only review of similar concept pages.", ""]
    grouped: dict[str, list[ConceptRelationshipReview]] = {
        "likely_duplicate": [],
        "related_but_distinct": [],
        "unclear": [],
    }
    for review in reviews:
        grouped.setdefault(review.relationship, []).append(review)
    for key, heading in (
        ("likely_duplicate", "Likely Duplicates"),
        ("related_but_distinct", "Related But Distinct"),
        ("unclear", "Unclear"),
    ):
        lines.append(f"## {heading}")
        lines.append("")
        entries = grouped.get(key, [])
        if not entries:
            lines.append("- None")
            lines.append("")
            continue
        for review in entries:
            lines.append(
                f"- [[{review.left_id}|{review.left_title}]] <-> [[{review.right_id}|{review.right_title}]]"
            )
            lines.append(f"  - Recommendation: {review.recommendation}")
            lines.append(f"  - Reason: {review.reason}")
            if review.heuristic_reasons:
                lines.append(f"  - Heuristics: {'; '.join(review.heuristic_reasons)}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
