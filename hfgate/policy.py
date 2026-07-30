"""Policy: findings + facts -> tier -> verdict.

This file is the tiering table from the design doc, executable. It is kept
separate from detection on purpose: the checks state what is true about the
artefact, and this file states what AISI has decided to do about it. Arguments
about risk appetite happen here, in one screen of code, without touching a
scanner.

Tiers:
  TRUSTED   Allowlisted publisher, safetensors only, no custom code, pinned
            revision. Mirrored with zero friction.
  STANDARD  Anyone-else's model that cannot execute code: safetensors/gguf
            only, no custom code. Scan is the control; publisher reputation
            is not required.
  ELEVATED  Something in the repo can execute code (pickle weights, custom
            code) but nothing looks hostile. Quarantined with a paved exit:
            conversion, or review-and-pin.
  BLOCKED   Positive evidence of hostility, or an artefact we cannot inspect.

Verdict mapping is deliberately blunt: TRUSTED/STANDARD -> allow, everything
else -> quarantine. Quarantine is a workflow state, not a "no" -- the reasons
list is the researcher's to-do list for getting out of it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .findings import Finding, Severity


class Tier(str, enum.Enum):
    TRUSTED = "trusted"
    STANDARD = "standard"
    ELEVATED = "elevated"
    BLOCKED = "blocked"


class Verdict(str, enum.Enum):
    ALLOW = "allow"
    QUARANTINE = "quarantine"


@dataclass
class Decision:
    tier: Tier
    verdict: Verdict
    reasons: list[str]

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW


# Finding ids that are positive evidence of hostility or uninspectability.
# Any one of these means BLOCKED regardless of anything else in the repo.
_BLOCKING_IDS = {
    "weights.pickle_dangerous_global",
    "weights.pickle_unparseable",
    "weights.bad_archive",
    "remote_code.dangerous_call",
    "sbom.possible_typosquat",
}

# Finding ids that mean "code can run at load time" without implying intent.
_CODE_EXECUTION_SURFACE_IDS = {
    "weights.pickle_only",
    "weights.keras_format",
    "weights.pickle_unexpected_global",
    "remote_code.auto_map_present",
}


def decide(findings: list[Finding], facts: dict) -> Decision:
    reasons: list[str] = []

    # --- BLOCKED: evidence of hostility, or inability to inspect -----------
    blocking = [f for f in findings if f.id in _BLOCKING_IDS]
    if blocking:
        for f in blocking:
            reasons.append(f.title)
        return Decision(Tier.BLOCKED, Verdict.QUARANTINE, reasons)

    # --- Establish the artefact's code-execution surface --------------------
    exec_surface = [f for f in findings if f.id in _CODE_EXECUTION_SURFACE_IDS]
    formats = facts.get("weight_formats", {})
    requires_trc = bool(facts.get("requires_trust_remote_code"))
    # Only Python the loader actually imports counts as a code path. A repo
    # shipping train_script.py for reproducibility is not executing anything at
    # from_pretrained time, and tiering on mere presence quarantined some of the
    # most-used models on the Hub for documenting themselves well.
    ships_python = bool(facts.get("loader_invoked_python"))

    provenance = facts.get("provenance", {})
    trusted_publisher = bool(provenance.get("trusted_publisher"))
    pinned = bool(provenance.get("revision_pinned"))
    provenance_known = bool(provenance.get("available"))

    # High-severity SBOM findings (VCS/URL deps) keep a model out of the
    # auto-allow tiers even when the artefact itself is inert: the declared
    # environment is part of what we would be approving.
    sbom_high = [
        f for f in findings
        if f.id == "sbom.vcs_or_url_dependency"
    ]

    # --- ELEVATED: a code path exists, nothing hostile found ---------------
    if exec_surface or ships_python or sbom_high:
        for f in exec_surface + sbom_high:
            reasons.append(f.title)
        if ships_python and not requires_trc and not exec_surface:
            reasons.append(
                "Repo ships Python that loading does not invoke; review before use"
            )
        return Decision(Tier.ELEVATED, Verdict.QUARANTINE, reasons)

    # --- Nothing to clear is not the same as nothing wrong ------------------
    # A repo with no recognisable weights is almost always an incomplete clone
    # (git-lfs pointers not fetched) or the wrong directory. Allowing it would
    # mean the gate returns "fine" for input it never actually inspected, which
    # is the one failure mode a gate must not have.
    if not formats:
        reasons.append(
            "No weight artefacts found. This is usually an incomplete clone "
            "(git-lfs objects not fetched) rather than a clean repo, and a gate "
            "must not pass input it did not inspect."
        )
        return Decision(Tier.ELEVATED, Verdict.QUARANTINE, reasons)

    # --- No code-execution surface: TRUSTED vs STANDARD ---------------------
    # Reaching here means no risky format raised anything actionable: the checks
    # already downgrade pickle and Keras files to informational when safetensors
    # is present, because that is what loaders select. So the question is not
    # "are all formats safe" -- an earlier version asked that, and quarantined
    # real repos merely for shipping a TensorFlow or OpenVINO export alongside
    # their safetensors. The question is whether a data-only format is present
    # to load at all.
    safe_formats = bool({"safetensors", "gguf"} & set(formats))

    if trusted_publisher and pinned and safe_formats:
        reasons.append(
            f"Allowlisted publisher `{provenance.get('owner')}` at pinned revision, "
            f"data-only weight format, no custom code"
        )
        return Decision(Tier.TRUSTED, Verdict.ALLOW, reasons)

    if safe_formats:
        reasons.append(
            "Data-only weight format and no custom code: no code-execution "
            "path at load time"
        )
        if not provenance_known:
            reasons.append("Provenance unknown; allowed on artefact evidence alone")
        elif not trusted_publisher:
            reasons.append(
                f"Publisher `{provenance.get('owner', 'unknown')}` not allowlisted; "
                f"the artefact checks carry the decision"
            )
        if not pinned and provenance_known:
            reasons.append(
                "Revision not pinned to a commit SHA; verdict applies to the "
                "scanned snapshot only"
            )
        return Decision(Tier.STANDARD, Verdict.ALLOW, reasons)

    # No data-only format to load: we cannot identify what would actually be
    # loaded here, so fail safe rather than guess.
    reasons.append(
        f"No safetensors or GGUF weights present; formats found: "
        f"{', '.join(sorted(formats)) or 'none'}. We cannot identify what would "
        f"be loaded, so this needs a look."
    )
    return Decision(Tier.ELEVATED, Verdict.QUARANTINE, reasons)
