"""Orchestrates the checks over one model repo and produces a decision."""

from __future__ import annotations

from pathlib import Path

from .checks import provenance, remote_code, sbom, weights
from .findings import CheckResult
from .policy import Decision, decide


def scan(root: Path, metadata_path: Path | None = None) -> tuple[Decision, CheckResult]:
    combined = CheckResult()
    combined.merge(weights.run(root))
    combined.merge(remote_code.run(root))
    combined.merge(sbom.run(root))
    combined.merge(provenance.run(root, metadata_path))

    decision = decide(combined.findings, combined.facts)
    return decision, combined
