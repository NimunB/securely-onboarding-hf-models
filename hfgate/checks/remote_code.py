"""Remote / custom code shipped with the repo.

`trust_remote_code=True` is the second code-execution primitive in
`from_pretrained`, and unlike pickle it is fully documented and widely used, so
researchers reach for it without alarm. When `config.json` carries an
`auto_map`, transformers imports the named module *from the model repo* and
instantiates the class. That code runs with the full privileges of the training
job: the pod's service account, its mounted secrets, its network position.

This check is deliberately not a linter. We care about two questions:
does loading this repo import third-party Python, and does that Python do
things a modelling file has no business doing.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ..findings import CheckResult, Finding, Severity

# Call targets that have no place in a modelling file. Matched on the dotted
# path of the call, so `os.system(...)` and `subprocess.run(...)` both hit.
DANGEROUS_CALLS = {
    "os.system": "executes a shell command",
    "os.popen": "executes a shell command",
    "os.execv": "replaces the process image",
    "os.execve": "replaces the process image",
    "os.spawnl": "spawns a process",
    "os.remove": "deletes files",
    "os.rmdir": "deletes directories",
    "shutil.rmtree": "recursively deletes a directory tree",
    "subprocess.run": "spawns a subprocess",
    "subprocess.call": "spawns a subprocess",
    "subprocess.Popen": "spawns a subprocess",
    "subprocess.check_output": "spawns a subprocess",
    "subprocess.getoutput": "spawns a shell",
    "eval": "evaluates a runtime-constructed expression",
    "exec": "executes runtime-constructed code",
    "compile": "compiles runtime-constructed code",
    "__import__": "performs a dynamic import",
    "importlib.import_module": "performs a dynamic import",
    "marshal.loads": "deserialises Python bytecode",
    "pickle.loads": "deserialises a pickle stream",
    "pickle.load": "deserialises a pickle stream",
    "base64.b64decode": "decodes base64, commonly used to obscure a payload",
    "codecs.decode": "decodes an encoded string, commonly used to obscure a payload",
    "requests.get": "makes an outbound HTTP request",
    "requests.post": "makes an outbound HTTP request",
    "urllib.request.urlopen": "makes an outbound HTTP request",
    "socket.socket": "opens a network socket",
    "ctypes.CDLL": "loads a native library",
    "setattr": "mutates attributes dynamically",
}

DANGEROUS_IMPORTS = {
    "subprocess": Severity.HIGH,
    "socket": Severity.HIGH,
    "ctypes": Severity.HIGH,
    "marshal": Severity.HIGH,
    "pty": Severity.HIGH,
    "telnetlib": Severity.HIGH,
    "paramiko": Severity.HIGH,
    "requests": Severity.MEDIUM,
    "urllib": Severity.MEDIUM,
    "httpx": Severity.MEDIUM,
    "boto3": Severity.MEDIUM,
    "os": Severity.LOW,
}

# Files transformers/diffusers will import as part of loading.
_LOADER_INVOKED = ("configuration_", "modeling_", "tokenization_",
                   "processing_", "image_processing_", "feature_extraction_",
                   "pipeline_")


def _local_imports(path: Path, local_stems: set[str]) -> set[str]:
    """Module stems this file imports that resolve to another file in the repo.

    Both `import utils_helper` and `from .utils_helper import x` resolve to the
    same local file, so relative and absolute forms are treated alike.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in local_stems:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in local_stems:
                    found.add(root)
            else:
                # `from . import utils_helper`
                for alias in node.names:
                    if alias.name in local_stems:
                        found.add(alias.name)
    return found


# Calls that make the import graph unknowable by reading the source. If an
# invoked file uses one of these, our closure is a lower bound and we should
# stop claiming any file is unreachable.
_DYNAMIC_IMPORT_CALLS = {
    "importlib.import_module", "importlib.__import__", "__import__",
    "exec", "eval", "runpy.run_module", "runpy.run_path",
}


def _uses_dynamic_import(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return True  # can't read it, can't bound it
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = _dotted(node.func)
            if target and target in _DYNAMIC_IMPORT_CALLS:
                return True
    return False


def _invoked_closure(
    seeds: list[str], rels: list[str], paths: list[Path]
) -> tuple[list[str], bool]:
    """Files reachable from the loader entrypoints, and whether that is complete.

    Naming convention alone is not enough: a `modeling_x.py` that transformers
    imports may itself `import utils_helper`, and that helper executes just as
    surely. Scoping by filename left a hole big enough to walk a payload
    through -- modelling code in the file that gets scanned, behaviour in the
    one that doesn't. So we follow imports from the entrypoints.

    But static analysis only bounds the graph while the imports are static. One
    `importlib.import_module(name)` and we can no longer prove any file is
    unreachable. When that happens we return every file and say so, rather than
    keep asserting "this one doesn't run" on evidence we no longer have.
    """
    by_stem = {Path(r).stem: r for r in rels}
    path_of = dict(zip(rels, paths))
    local_stems = set(by_stem)

    invoked: set[str] = set(seeds)
    queue = list(seeds)
    bounded = True
    while queue:
        rel = queue.pop()
        target = path_of.get(rel)
        if target is None:
            continue
        if _uses_dynamic_import(target):
            bounded = False
        for stem in _local_imports(target, local_stems):
            nxt = by_stem.get(stem)
            if nxt and nxt not in invoked:
                invoked.add(nxt)
                queue.append(nxt)

    if not bounded:
        return sorted(rels), False
    return sorted(invoked), True


def _dotted(node: ast.AST) -> str | None:
    """Render an ast call target as a dotted string, or None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _analyse_python(path: Path, rel: str, invoked: bool) -> CheckResult:
    """Analyse one shipped .py file.

    `invoked` is whether loading the model actually imports this file, i.e. it
    is named in auto_map or follows the naming convention transformers imports.
    Severity depends on it, because the threat model does: code that
    `from_pretrained` never imports is not a load-time execution path. Training
    scripts shipped for reproducibility routinely call subprocess, and treating
    them as backdoors punishes the repos that document themselves best.
    """
    result = CheckResult()
    source = path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        result.add(
            Finding(
                id="remote_code.unparseable",
                severity=Severity.MEDIUM,
                title=f"Python file could not be parsed: {rel}",
                detail=f"{rel} is not valid Python for this interpreter ({exc.msg} "
                       f"at line {exc.lineno}); it was not analysed.",
                where=rel,
            )
        )
        return result

    seen_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            seen_imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            target = _dotted(node.func)
            if target and target in DANGEROUS_CALLS:
                result.add(
                    Finding(
                        id="remote_code.dangerous_call" if invoked
                           else "remote_code.dangerous_call_uninvoked",
                        severity=Severity.CRITICAL if invoked else Severity.LOW,
                        title=f"Shipped code calls `{target}`"
                              + ("" if invoked else " (not imported on load)"),
                        detail=(
                            f"{rel}:{node.lineno} calls `{target}`, which "
                            f"{DANGEROUS_CALLS[target]}."
                            + (
                                " This runs inside the training or inference process "
                                "when the model is loaded with trust_remote_code=True, "
                                "with that process's credentials and network access."
                                if invoked
                                else " This file is not named in auto_map and does not "
                                     "follow the naming convention transformers imports, "
                                     "so loading the model does not execute it. Typically "
                                     "a training or example script shipped for "
                                     "reproducibility."
                            )
                        ),
                        where=f"{rel}:{node.lineno}",
                        remediation=(
                            "Do not load with trust_remote_code=True. Review the file "
                            "by hand before considering an exception."
                            if invoked
                            else "No action needed unless you intend to run this script "
                                 "yourself, in which case read it first."
                        ),
                    )
                )

    for module, severity in DANGEROUS_IMPORTS.items():
        if module in seen_imports:
            result.add(
                Finding(
                    id="remote_code.notable_import" if invoked
                       else "remote_code.notable_import_uninvoked",
                    severity=severity if invoked else Severity.INFO,
                    title=f"Shipped code imports `{module}`"
                          + ("" if invoked else " (not imported on load)"),
                    detail=(
                        f"{rel} imports `{module}`. A modelling file normally needs "
                        f"only torch, transformers and numpy; process control, "
                        f"sockets and HTTP clients are outside that shape."
                        + ("" if invoked else " Loading the model does not import "
                                              "this file.")
                    ),
                    where=rel,
                )
            )

    return result


def run(root: Path) -> CheckResult:
    result = CheckResult()

    py_files = sorted(
        p for p in root.rglob("*.py")
        if p.is_file() and ".git" not in p.parts
    )
    rels = [str(p.relative_to(root)) for p in py_files]
    result.facts["python_files"] = rels

    # auto_map is the explicit declaration that loading imports repo code.
    auto_map_entries: dict[str, str] = {}
    for config_name in ("config.json", "tokenizer_config.json", "processor_config.json"):
        config_path = root / config_name
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        auto_map = config.get("auto_map")
        if isinstance(auto_map, dict):
            for key, value in auto_map.items():
                auto_map_entries[f"{config_name}:{key}"] = str(value)

    result.facts["auto_map"] = auto_map_entries
    result.facts["requires_trust_remote_code"] = bool(auto_map_entries)

    # An auto_map target may point at *another repo* using the documented
    # "owner/repo--module.Class" syntax. Found by scanning real Hub models:
    # nomic-ai/nomic-embed-text-v1.5 sources all seven of its classes from
    # nomic-ai/nomic-bert-2048. This matters more than local custom code,
    # because pinning *this* repo's commit does not pin the other repo -- the
    # code that executes can change with our revision still fully pinned.
    external_repos: set[str] = set()
    local_modules: set[str] = set()
    for value in auto_map_entries.values():
        if "--" in value:
            external_repos.add(value.split("--", 1)[0])
        else:
            local_modules.add(value.split(".", 1)[0])

    result.facts["auto_map_external_repos"] = sorted(external_repos)

    if auto_map_entries:
        # Summarise rather than dumping every entry: real repos declare seven or
        # more mappings that all resolve to the same one or two modules.
        n = len(auto_map_entries)
        sources = sorted(external_repos) + sorted(local_modules)
        result.add(
            Finding(
                id="remote_code.auto_map_present",
                severity=Severity.HIGH,
                title="Repo requires trust_remote_code=True",
                detail=(
                    f"config.json declares {n} auto_map "
                    f"{'entry' if n == 1 else 'entries'}, sourced from: "
                    f"{', '.join(sources)}. Loading this model imports and executes "
                    f"that Python. It is not reviewed by Hugging Face."
                ),
                where="config.json",
                remediation=(
                    "Prefer an architecture supported natively by transformers. If the "
                    "custom code is genuinely needed, it runs in the Elevated tier: "
                    "isolated namespace, no egress, no mounted credentials."
                ),
            )
        )

    if external_repos:
        result.add(
            Finding(
                id="remote_code.auto_map_external_repo",
                severity=Severity.HIGH,
                title=f"Executable code is sourced from {len(external_repos)} other repo(s)",
                detail=(
                    f"auto_map points at {', '.join(sorted(external_repos))}, not at "
                    f"files in this repo. Two consequences. First, scanning this repo "
                    f"tells us almost nothing about the code that will actually run. "
                    f"Second, and more important: pinning this repo to a commit does "
                    f"NOT pin that code. The other repo can change at any time while "
                    f"our revision stays fully pinned, so a clean verdict here has no "
                    f"shelf life."
                ),
                where="config.json",
                remediation=(
                    "Scan and pin each source repo as a dependency in its own right, "
                    "or vendor the modelling code into a repo we control."
                ),
            )
        )

    if not py_files:
        if not auto_map_entries:
            result.add(
                Finding(
                    id="remote_code.none",
                    severity=Severity.INFO,
                    title="No custom code in repo",
                    detail="No Python files and no auto_map. Loading this model uses "
                           "only library code we already trust.",
                )
            )
        return result

    # A file is "invoked" if loading the model imports it: either auto_map names
    # its module, or it follows the convention transformers imports. Anything
    # else is inert as far as from_pretrained is concerned.
    automap_modules = {
        v.split("--", 1)[-1].split(".", 1)[0] for v in auto_map_entries.values()
    }
    seeds = [
        r for r in rels
        if Path(r).name.startswith(_LOADER_INVOKED) or Path(r).stem in automap_modules
    ]
    # ...and everything those files import, transitively.
    loader_invoked, bounded = _invoked_closure(seeds, rels, py_files)
    result.facts["loader_invoked_python"] = loader_invoked
    result.facts["loader_entrypoint_python"] = seeds
    result.facts["import_graph_bounded"] = bounded

    if not bounded:
        result.add(
            Finding(
                id="remote_code.dynamic_import",
                severity=Severity.HIGH,
                title="Loaded code imports dynamically; import graph cannot be bounded",
                detail=(
                    "A file the loader imports resolves modules at runtime (via "
                    "importlib, __import__, exec or eval), so reading the source does "
                    "not tell us which files execute. Every Python file in the repo is "
                    "therefore treated as reachable."
                ),
                remediation="Review all shipped Python by hand, or use a build that "
                            "imports statically.",
            )
        )

    result.add(
        Finding(
            id="remote_code.python_present" if loader_invoked
               else "remote_code.python_present_uninvoked",
            severity=Severity.HIGH if loader_invoked else Severity.LOW,
            title=f"Repo ships {len(rels)} Python file(s)"
                  + ("" if loader_invoked else ", none imported on load"),
            detail=(
                f"Present: {', '.join(rels[:8])}"
                f"{' and others' if len(rels) > 8 else ''}."
                + (
                    f" {len(loader_invoked)} of these are imported when the model "
                    f"loads: {', '.join(loader_invoked[:5])}."
                    if loader_invoked
                    else " None are named in auto_map or follow the naming convention "
                         "transformers imports, so loading the model does not execute "
                         "them. Usually training or example scripts shipped for "
                         "reproducibility."
                )
            ),
            remediation="Review shipped code before use, or pull the model without it."
            if loader_invoked
            else "No action needed; these are inert unless you run them yourself.",
        )
    )

    invoked_set = set(loader_invoked)
    for path, rel in zip(py_files, rels):
        result.merge(_analyse_python(path, rel, rel in invoked_set))

    return result
