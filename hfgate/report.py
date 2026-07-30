"""Report emission: one machine-readable record, one human summary.

The JSON record is the interface to everything downstream (the registry in
Part B ingests it verbatim). The human summary is for the researcher whose
model just got quarantined, so it leads with the verdict and what to do about
it, not with a wall of findings.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .findings import Finding, Severity
from .policy import Decision

SCHEMA_VERSION = "1"
_TIER_LABEL = {
    "trusted": "Tier 1 - Trusted",
    "standard": "Tier 2 - Standard",
    "elevated": "Tier 3 - Elevated",
    "blocked": "Tier 4 - Blocked",
}


def machine_record(
    target: Path,
    decision: Decision,
    findings: list[Finding],
    facts: dict,
) -> dict:
    provenance = facts.get("provenance", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "hfgate", "version": "0.1.0"},
        "scanned_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "target": {
            "path": str(target),
            "repo_id": provenance.get("repo_id"),
            "revision": provenance.get("revision"),
        },
        "verdict": decision.verdict.value,
        "tier": decision.tier.value,
        "reasons": decision.reasons,
        "findings": [f.to_dict() for f in findings],
        "facts": facts,
    }


def _fmt_finding(f: Finding) -> str:
    lines = [f"  [{f.severity}] {f.title}"]
    if f.where:
        lines.append(f"      where: {f.where}")
    lines.append(f"      {f.detail}")
    if f.remediation:
        lines.append(f"      fix: {f.remediation}")
    return "\n".join(lines)


def human_summary(target: Path, decision: Decision, findings: list[Finding], facts: dict) -> str:
    provenance = facts.get("provenance", {})
    repo_id = provenance.get("repo_id") or target.name

    banner = "ALLOW" if decision.allowed else "QUARANTINE"
    lines = [
        "=" * 72,
        f"  {banner}: {repo_id}",
        f"  {_TIER_LABEL[decision.tier.value]}",
        "=" * 72,
        "",
        "Why:",
    ]
    lines += [f"  - {r}" for r in decision.reasons]

    if not decision.allowed:
        lines += [
            "",
            "What this means: the model is held at the mirror, not deleted. The",
            "findings below are the exit checklist -- resolve them (or request a",
            "reviewed exception) and rescan.",
        ]

    ordered = sorted(findings, key=lambda f: f.severity, reverse=True)
    notable = [f for f in ordered if f.severity >= Severity.LOW]
    info = [f for f in ordered if f.severity < Severity.LOW]

    if notable:
        lines += ["", f"Findings ({len(notable)}):"]
        lines += [_fmt_finding(f) for f in notable]
    if info:
        lines += ["", "For the record:"]
        lines += [f"  [{f.severity}] {f.title}: {f.detail}" for f in info]

    formats = facts.get("weight_formats", {})
    if formats:
        lines += ["", "Weight formats: " + ", ".join(
            f"{fmt} ({len(files)})" for fmt, files in formats.items()
        )]
    dep_count = facts.get("dependency_count")
    if dep_count is not None:
        lines += [f"Declared dependencies: {dep_count}"]

    lines += [""]
    return "\n".join(lines)


def write_record(record: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
