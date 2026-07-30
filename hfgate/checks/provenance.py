"""Provenance and metadata.

Provenance answers a different question from the other checks. Those ask "does
this artefact do something dangerous". This one asks "if it turns out to be
dangerous later, what do we know, and can we find every job that used it".

Two things carry almost all the weight:

  Commit pinning. A HF repo is a git repo, and `main` moves. A model that
  scanned clean on Monday is a different artefact on Friday, under the same
  name, in code that has not changed. Pinning the revision is what makes a scan
  verdict mean anything at all, and it is what makes recall possible.

  Publisher identity. Not a popularity contest -- an allowlisted org is one
  where a compromise is a newsworthy event with a disclosure process attached,
  which is materially different from an anonymous account with a finetune.

Everything else here (card, license, download counts) is weak evidence. We
report it, we weight it lightly, and we say so rather than dressing it up.

Live Hub metadata is read from a sidecar JSON file so the tool runs offline
against a local clone. In the real system this is the mirror's Hub API response,
captured at intake and stored with the verdict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..findings import CheckResult, Finding, Severity

# Publishers whose releases we treat as attributable: a named legal entity, a
# security contact, and enough public scrutiny that a compromised release is
# discovered and disclosed rather than sitting unnoticed.
TRUSTED_PUBLISHERS = {
    "meta-llama", "mistralai", "google", "qwen", "deepseek-ai", "microsoft",
    "allenai", "eleutherai", "openai", "huggingfacetb", "huggingface",
    "bigscience", "bigcode", "tiiuae", "nvidia", "ibm-granite", "stabilityai",
    "facebook", "cohereforai", "ai21labs", "databricks", "sentence-transformers",
    "laion", "apple", "amazon", "aisi",
}

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_METADATA_NAMES = ("hf_metadata.json", ".hfgate/metadata.json")


def _load_metadata(root: Path, explicit: Path | None) -> tuple[dict | None, str | None]:
    if explicit is not None:
        return json.loads(explicit.read_text(encoding="utf-8")), str(explicit)
    for name in _METADATA_NAMES:
        candidate = root / name
        if candidate.is_file():
            try:
                return (
                    json.loads(candidate.read_text(encoding="utf-8")),
                    name,
                )
            except json.JSONDecodeError:
                return None, name
    return None, None


def _has_model_card(root: Path) -> bool:
    card = root / "README.md"
    return card.is_file() and len(card.read_text(encoding="utf-8", errors="replace").strip()) > 200


def run(root: Path, metadata_path: Path | None = None) -> CheckResult:
    result = CheckResult()
    metadata, source = _load_metadata(root, metadata_path)

    if metadata is None:
        result.facts["provenance"] = {"available": False}
        result.add(
            Finding(
                id="provenance.metadata_unavailable",
                severity=Severity.MEDIUM,
                title="No Hub metadata available",
                detail=(
                    "No sidecar metadata was found, so publisher, revision and gating "
                    "status are unknown. We can say what this artefact contains but "
                    "not where it came from, and an unattributed artefact cannot be "
                    "recalled if it is later found to be malicious."
                ),
                where=source,
                remediation="Capture the Hub API response at intake and rescan.",
            )
        )
        return result

    repo_id = str(metadata.get("id") or metadata.get("modelId") or "")
    owner = repo_id.split("/")[0] if "/" in repo_id else ""
    revision = str(metadata.get("sha") or metadata.get("revision") or "")
    downloads = metadata.get("downloads")
    likes = metadata.get("likes")
    gated = metadata.get("gated", False)
    private = metadata.get("private", False)
    tags = metadata.get("tags") or []
    license_id = metadata.get("license") or next(
        (t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("license:")),
        None,
    )
    created = metadata.get("createdAt") or metadata.get("created_at")
    trusted = _normalise_owner(owner) in TRUSTED_PUBLISHERS

    result.facts["provenance"] = {
        "available": True,
        "repo_id": repo_id,
        "owner": owner,
        "trusted_publisher": trusted,
        "revision": revision,
        "revision_pinned": bool(_FULL_SHA.match(revision)),
        "downloads": downloads,
        "likes": likes,
        "gated": bool(gated),
        "private": bool(private),
        "license": license_id,
        "created_at": created,
        "metadata_source": source,
    }

    if not repo_id:
        result.add(
            Finding(
                id="provenance.no_repo_id",
                severity=Severity.MEDIUM,
                title="Metadata does not identify the repo",
                detail="The metadata carries no `id`, so the artefact cannot be tied "
                       "back to a Hub repo.",
                where=source,
            )
        )

    if not _FULL_SHA.match(revision):
        result.add(
            Finding(
                id="provenance.revision_not_pinned",
                severity=Severity.MEDIUM,
                title="Revision is not a full commit SHA",
                detail=(
                    f"Revision recorded as {revision or '(none)'}. A Hub repo is a git "
                    f"repo and branch refs move, so a verdict against a branch name "
                    f"expires the moment the publisher pushes. Without a pinned commit "
                    f"we cannot say this scan describes what will actually be loaded."
                ),
                where=source,
                remediation="Resolve the ref to a 40-character commit SHA at intake and "
                            "record the verdict against that SHA.",
            )
        )

    if trusted:
        result.add(
            Finding(
                id="provenance.trusted_publisher",
                severity=Severity.INFO,
                title=f"Published by allowlisted org `{owner}`",
                detail=(
                    f"`{owner}` is on the publisher allowlist: an attributable "
                    f"organisation with a disclosure process, where a compromised "
                    f"release would be a public event."
                ),
            )
        )
    elif owner:
        weak_signal = (
            isinstance(downloads, int) and downloads < 1000
            and isinstance(likes, int) and likes < 10
        )
        result.add(
            Finding(
                id="provenance.unknown_publisher",
                severity=Severity.LOW if not weak_signal else Severity.MEDIUM,
                title=f"Publisher `{owner}` is not on the allowlist",
                detail=(
                    f"`{owner}` is an unattributed account "
                    f"(downloads: {downloads if downloads is not None else 'unknown'}, "
                    f"likes: {likes if likes is not None else 'unknown'}). "
                    + (
                        "Low engagement means few other people have looked at this "
                        "artefact, so we should not lean on community scrutiny."
                        if weak_signal
                        else "Engagement is non-trivial, but popularity is weak "
                             "evidence of safety and we weight it lightly."
                    )
                ),
                remediation="Fine for most research use; the artefact checks carry the "
                            "weight here rather than the publisher's reputation.",
            )
        )

    if not license_id:
        result.add(
            Finding(
                id="provenance.no_license",
                severity=Severity.LOW,
                title="No license declared",
                detail="No license in tags or metadata. This is a usage-rights and "
                       "publication question rather than a security one, but it does "
                       "need answering before results go into a public write-up.",
                remediation="Confirm licensing before the model is used in published work.",
            )
        )

    if not _has_model_card(root):
        result.add(
            Finding(
                id="provenance.thin_model_card",
                severity=Severity.LOW,
                title="Model card is missing or minimal",
                detail="Little documentation of training data, intended use or known "
                       "limitations. Weak evidence on its own, and it does make the "
                       "artefact harder to reason about after the fact.",
            )
        )

    if private or gated:
        result.add(
            Finding(
                id="provenance.gated_repo",
                severity=Severity.INFO,
                title="Repo is gated or private",
                detail="Access requires accepted terms or a credentialed token, which "
                       "means the publisher knows who pulled it and we have a named "
                       "agreement in place.",
            )
        )

    return result


def _normalise_owner(owner: str) -> str:
    return owner.strip().lower()
