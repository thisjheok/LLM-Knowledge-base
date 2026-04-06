from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VaultPaths:
    root: Path
    raw: Path
    sources: Path
    concepts: Path
    indexes: Path
    outputs: Path
    logs: Path
    normalized: Path


def build_paths(project_root: Path) -> VaultPaths:
    vault_root = project_root / "vault"
    data_root = project_root / "data" / "normalized"
    return VaultPaths(
        root=vault_root,
        raw=vault_root / "00_raw",
        sources=vault_root / "10_sources",
        concepts=vault_root / "20_concepts",
        indexes=vault_root / "30_indexes",
        outputs=vault_root / "40_outputs",
        logs=vault_root / "90_logs",
        normalized=data_root,
    )


def ensure_layout(paths: VaultPaths) -> None:
    for path in (
        paths.root,
        paths.raw,
        paths.sources,
        paths.concepts,
        paths.indexes,
        paths.outputs,
        paths.logs,
        paths.normalized,
    ):
        path.mkdir(parents=True, exist_ok=True)
