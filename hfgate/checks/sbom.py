"""Declared-dependency SBOM.

Scope note, because this is where scanners usually sprawl: we build an SBOM of
what the *model repo declares*, not of the researcher's environment. A model
repo asking for extra packages is unusual and interesting; a full transitive
resolution of the research image is a different problem with a different owner.

The output is CycloneDX-shaped so it can feed anything that already speaks that
format, but we emit it ourselves rather than pulling in a toolchain -- a tool
that warns about dependencies should not arrive with a pile of its own.

What we actually care about, in order:
  1. Direct references to a VCS or URL. That bypasses the index entirely and
     points at mutable content; it is the strongest supply-chain signal here.
  2. Unpinned versions. Not dangerous today, but it means what we scanned is
     not what will install tomorrow, which undermines every other check.
  3. Packages whose capability is out of shape for a model repo (network,
     process control, cloud credentials).
  4. Names that are one edit away from a popular package.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ..findings import CheckResult, Finding, Severity

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - exercised on 3.9/3.10 only
    tomllib = None

# Capability-based concern, not a blocklist of "bad" packages. Each of these is
# perfectly legitimate in general; the point is that a model repo declaring
# them is asking for a capability it should not need.
OUT_OF_SHAPE = {
    "requests": "outbound HTTP",
    "httpx": "outbound HTTP",
    "urllib3": "outbound HTTP",
    "aiohttp": "outbound HTTP",
    "pycurl": "outbound HTTP",
    "boto3": "cloud credentials and API access",
    "botocore": "cloud credentials and API access",
    "google-cloud-storage": "cloud credentials and API access",
    "azure-storage-blob": "cloud credentials and API access",
    "paramiko": "outbound SSH",
    "fabric": "remote command execution",
    "pexpect": "process control",
    "psutil": "process and system introspection",
    "cryptography": "cryptographic operations (common in ransomware staging)",
    "pycryptodome": "cryptographic operations",
    "python-telegram-bot": "outbound messaging, a common exfiltration channel",
    "discord.py": "outbound messaging, a common exfiltration channel",
}

# Names close to popular packages. Short and honest rather than exhaustive; a
# real deployment would diff against the top-N index names automatically.
TYPOSQUATS = {
    "torchvison": "torchvision",
    "torchvsion": "torchvision",
    "trasformers": "transformers",
    "transfomers": "transformers",
    "tranformers": "transformers",
    "hugginface-hub": "huggingface-hub",
    "huggingface": "huggingface-hub",
    "numpu": "numpy",
    "nunpy": "numpy",
    "sklearn": "scikit-learn",
    "python-dateutil2": "python-dateutil",
    "requsts": "requests",
    "beautifulsoup": "beautifulsoup4",
    "accelerte": "accelerate",
}

_REQ_LINE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9._-]+)\s*(?P<spec>.*)$"
)
_PIN = re.compile(r"==\s*[0-9]")
_VCS_OR_URL = re.compile(r"(^|\s|@\s*)(git\+|hg\+|svn\+|bzr\+|https?://|file://)", re.I)


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirements(text: str, source: str) -> list[dict]:
    components: list[dict] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        components.append(_component_from_spec(line, source))
    return components


def _component_from_spec(spec: str, source: str) -> dict:
    is_url = bool(_VCS_OR_URL.search(spec))
    name, version = spec, None

    if is_url:
        # "pkg @ git+https://..." or a bare URL
        if "@" in spec:
            name = spec.split("@", 1)[0].strip() or spec
    else:
        match = _REQ_LINE.match(spec)
        if match:
            name = match.group("name")
            rest = match.group("spec").strip()
            if _PIN.search(rest):
                version = rest.split("==", 1)[1].strip().split(",")[0].strip()

    return {
        "name": name,
        "normalised": _normalise(name),
        "version": version,
        "pinned": version is not None,
        "vcs_or_url": is_url,
        "raw": spec,
        "source": source,
    }


# Fallback extractor for interpreters without tomllib (< 3.11).
#
# This exists because of a bug worth remembering: when the pyproject branch was
# simply skipped on older interpreters, the scanner silently reported zero
# dependencies, and a repo declaring a typosquat plus a git+https dependency
# was Blocked on 3.11 and Allowed on 3.9. A verdict that depends on the
# interpreter is worse than no verdict, because it looks like an answer.
#
# The rule this encodes: a check may be less precise on an old interpreter, but
# it may never silently not run. Where this parser is less sure than tomllib, it
# says so in a finding rather than returning a confident empty list.
_ARRAY_RE = re.compile(
    r"^\s*dependencies\s*=\s*\[(?P<body>.*?)\]", re.S | re.M
)
_OPT_TABLE_RE = re.compile(
    r"^\s*\[project\.optional-dependencies\](?P<body>.*?)(?=^\s*\[|\Z)", re.S | re.M
)
_STRING_RE = re.compile(r"['\"]([^'\"]+)['\"]")


def _parse_pyproject_fallback(text: str) -> tuple[list[tuple[str, str]], bool]:
    """Best-effort (spec, source) extraction without tomllib.

    Returns the specs found and whether parsing was complete enough to trust.
    Handles the ordinary `[project] dependencies = [...]` and
    `[project.optional-dependencies]` layouts; multi-line and inline both work.
    """
    found: list[tuple[str, str]] = []

    match = _ARRAY_RE.search(text)
    if match:
        for spec in _STRING_RE.findall(match.group("body")):
            found.append((spec, "pyproject.toml"))

    opt = _OPT_TABLE_RE.search(text)
    if opt:
        for line in opt.group("body").splitlines():
            key, _, rest = line.partition("=")
            group = key.strip()
            if not group or "[" not in rest:
                continue
            for spec in _STRING_RE.findall(rest):
                found.append((spec, f"pyproject.toml[{group}]"))

    # If the file declares dependencies but we extracted none, we are not
    # confident -- say so rather than reporting a clean bill of health.
    declares = "dependencies" in text
    complete = not declares or bool(found)
    return found, complete


def _collect(root: Path) -> tuple[list[dict], bool]:
    """Returns (components, parsing_was_complete)."""
    components: list[dict] = []
    complete = True

    for req in sorted(root.rglob("requirements*.txt")):
        if req.is_file():
            components.extend(
                _parse_requirements(
                    req.read_text(encoding="utf-8", errors="replace"),
                    str(req.relative_to(root)),
                )
            )

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        if tomllib is not None:
            try:
                data = tomllib.loads(text)
            except (tomllib.TOMLDecodeError, OSError):
                data, complete = {}, False
            for spec in data.get("project", {}).get("dependencies", []) or []:
                components.append(_component_from_spec(str(spec), "pyproject.toml"))
            optional = data.get("project", {}).get("optional-dependencies", {}) or {}
            for group, specs in optional.items():
                for spec in specs:
                    components.append(
                        _component_from_spec(str(spec), f"pyproject.toml[{group}]")
                    )
        else:
            specs, parsed_ok = _parse_pyproject_fallback(text)
            complete = complete and parsed_ok
            for spec, source in specs:
                components.append(_component_from_spec(spec, source))

    setup_py = root / "setup.py"
    if setup_py.is_file():
        text = setup_py.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.S)
        if match:
            for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
                components.append(_component_from_spec(item, "setup.py"))

    return components, complete


def _cyclonedx(components: list[dict]) -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"tools": [{"name": "hfgate", "version": "0.1.0"}]},
        "components": [
            {
                "type": "library",
                "name": c["name"],
                "version": c["version"] or "unspecified",
                "purl": f"pkg:pypi/{c['normalised']}@{c['version']}"
                if c["version"]
                else f"pkg:pypi/{c['normalised']}",
                "properties": [
                    {"name": "hfgate:source", "value": c["source"]},
                    {"name": "hfgate:pinned", "value": str(c["pinned"]).lower()},
                    {"name": "hfgate:vcs_or_url", "value": str(c["vcs_or_url"]).lower()},
                ],
            }
            for c in components
        ],
    }


def run(root: Path) -> CheckResult:
    result = CheckResult()
    components, complete = _collect(root)

    result.facts["sbom"] = _cyclonedx(components)
    result.facts["dependency_count"] = len(components)
    result.facts["sbom_parsing_complete"] = complete

    if not complete:
        # Never let a parsing gap read as a clean result.
        result.add(
            Finding(
                id="sbom.incomplete_parse",
                severity=Severity.MEDIUM,
                title="Dependency manifest could not be fully parsed",
                detail=(
                    "A manifest in this repo declares dependencies that could not be "
                    "extracted"
                    + (
                        " (running on Python < 3.11, where the fallback TOML parser is "
                        "less capable than tomllib)."
                        if tomllib is None
                        else " (the file is not valid TOML)."
                    )
                    + " The dependency findings below are therefore incomplete, and "
                    "their absence is not evidence of absence."
                ),
                remediation="Rescan on Python 3.11+ for full manifest coverage.",
            )
        )

    if not components and complete:
        result.add(
            Finding(
                id="sbom.no_declared_dependencies",
                severity=Severity.INFO,
                title="No declared dependencies",
                detail="The repo declares no Python dependencies of its own, so it "
                       "loads with whatever the research image already provides.",
            )
        )
        return result

    vcs = [c for c in components if c["vcs_or_url"]]
    if vcs:
        result.add(
            Finding(
                id="sbom.vcs_or_url_dependency",
                severity=Severity.HIGH,
                title=f"{len(vcs)} dependency declared by URL or VCS reference",
                detail=(
                    "These bypass the package index and its integrity guarantees, and "
                    "point at content that can change under the same reference: "
                    + "; ".join(f"{c['raw']} ({c['source']})" for c in vcs[:5])
                    + (" and others" if len(vcs) > 5 else "")
                    + "."
                ),
                remediation="Require index-published, version-pinned releases, or "
                            "vendor the dependency into the internal index.",
            )
        )

    unpinned = [c for c in components if not c["pinned"] and not c["vcs_or_url"]]
    if unpinned:
        names = ", ".join(c["name"] for c in unpinned[:8])
        result.add(
            Finding(
                id="sbom.unpinned_dependencies",
                severity=Severity.MEDIUM,
                title=f"{len(unpinned)} of {len(components)} dependencies are unpinned",
                detail=(
                    f"Unpinned: {names}"
                    f"{' and others' if len(unpinned) > 8 else ''}. "
                    f"What we scanned is not necessarily what will install later, "
                    f"which weakens every other finding in this report and makes the "
                    f"environment non-reproducible across runs."
                ),
                remediation="Pin with == and a hash, or resolve through a lockfile.",
            )
        )

    for component in components:
        squat_target = TYPOSQUATS.get(component["normalised"])
        if squat_target:
            result.add(
                Finding(
                    id="sbom.possible_typosquat",
                    severity=Severity.HIGH,
                    title=f"Dependency `{component['name']}` resembles `{squat_target}`",
                    detail=(
                        f"Declared in {component['source']}. The name is a small edit "
                        f"away from a widely used package, which is the standard shape "
                        f"of an index-squatting attack."
                    ),
                    where=component["source"],
                    remediation=f"Confirm the intended package. If it is "
                                f"`{squat_target}`, correct the spelling.",
                )
            )

        concern = OUT_OF_SHAPE.get(component["normalised"])
        if concern:
            result.add(
                Finding(
                    id="sbom.capability_out_of_shape",
                    severity=Severity.MEDIUM,
                    title=f"Dependency `{component['name']}` provides {concern}",
                    detail=(
                        f"Declared in {component['source']}. The package is legitimate, "
                        f"but a model repo should not need {concern} to define an "
                        f"architecture or load weights."
                    ),
                    where=component["source"],
                )
            )

    return result
