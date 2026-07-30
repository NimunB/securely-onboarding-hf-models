"""Findings are the common currency between checks and policy.

A check never decides anything. It reports what it saw, with a severity, and
policy.py turns the set of findings into a tier and a verdict. Keeping that
split means you can argue about the policy without touching the detection.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict


class Severity(enum.IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __str__(self) -> str:
        return self.name


@dataclass
class Finding:
    """One observation about a model repo.

    `id` is a stable dotted slug so downstream systems (and waivers) can refer
    to a finding class without string-matching on prose.
    """

    id: str
    severity: Severity
    title: str
    detail: str
    where: str | None = None
    remediation: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = str(self.severity)
        return d


@dataclass
class CheckResult:
    """What a single check produces: findings plus structured facts.

    Facts are the machine-readable observations policy may key on directly
    (e.g. "weight formats present"), as opposed to findings which are the
    human-facing narrative.
    """

    findings: list[Finding] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def add(self, *findings: Finding) -> None:
        self.findings.extend(findings)

    def merge(self, other: "CheckResult") -> None:
        self.findings.extend(other.findings)
        self.facts.update(other.facts)


def max_severity(findings: list[Finding]) -> Severity:
    if not findings:
        return Severity.INFO
    return max(f.severity for f in findings)
