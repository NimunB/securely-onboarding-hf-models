"""Weight format inventory.

The question this check answers is narrow: *what does loading this repo's
weights actually do?* Formats differ enormously on that axis, and the
difference is the single strongest signal we have.

  safetensors  Length-prefixed header plus raw tensor bytes. The parser reads
               numbers. There is no callable to resolve, so there is no code
               execution primitive to abuse. This is why it is the paved road.
  gguf         Also data-only. Risk collapses to memory-safety bugs in the
               loader, which is a real but far smaller surface.
  pickle-based Runs the pickle VM on load. See pickle_opcodes.py.
  keras/h5     Can carry Lambda layers containing marshalled Python bytecode,
               which Keras executes on load. Distinct mechanism from pickle,
               same outcome.
"""

from __future__ import annotations

from pathlib import Path

from ..findings import CheckResult, Finding, Severity
from .pickle_opcodes import PICKLE_WEIGHT_SUFFIXES, scan_weight_file

SAFE_SUFFIXES = {".safetensors"}
GGUF_SUFFIXES = {".gguf", ".ggml"}
KERAS_SUFFIXES = {".h5", ".hdf5", ".keras", ".pb"}
NUMPY_SUFFIXES = {".npy", ".npz"}

# Files that use a pickle suffix but are small metadata sidecars rather than
# weights. We still scan them -- a malicious pickle is a malicious pickle -- but
# we do not treat their mere presence as "this repo ships pickle weights".
_SIDECAR_STEMS = {"scheduler", "optimizer", "training_args", "rng_state"}


def _format_of(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in SAFE_SUFFIXES:
        return "safetensors"
    if suffix in GGUF_SUFFIXES:
        return "gguf"
    if suffix in KERAS_SUFFIXES:
        return "keras"
    if suffix in NUMPY_SUFFIXES:
        return "numpy"
    if suffix in PICKLE_WEIGHT_SUFFIXES:
        return "pickle"
    return None


def run(root: Path) -> CheckResult:
    result = CheckResult()
    by_format: dict[str, list[str]] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        fmt = _format_of(path)
        if fmt is None:
            continue
        rel = str(path.relative_to(root))
        by_format.setdefault(fmt, []).append(rel)

        if fmt == "pickle":
            result.merge(scan_weight_file(path, rel))
        elif fmt == "numpy" and _looks_pickled(path):
            # .npy is data-only unless it holds an object array, in which case
            # the payload is a pickle stream and np.load(allow_pickle=True)
            # runs it.
            result.merge(scan_weight_file(path, rel))

    result.facts["weight_formats"] = {k: sorted(v) for k, v in sorted(by_format.items())}

    if not by_format:
        result.add(
            Finding(
                id="weights.none_found",
                severity=Severity.LOW,
                title="No weight files found",
                detail="No recognised weight artefacts in the repo. This may be a "
                       "config-only or adapter-only repo, or the clone may be "
                       "incomplete (git-lfs pointers not fetched).",
                remediation="Confirm the clone included LFS objects before relying "
                            "on this verdict.",
            )
        )
        return result

    pickle_files = [
        f for f in by_format.get("pickle", [])
        if Path(f).stem not in _SIDECAR_STEMS
    ]
    has_safetensors = "safetensors" in by_format

    if pickle_files and has_safetensors:
        result.add(
            Finding(
                id="weights.pickle_alongside_safetensors",
                severity=Severity.LOW,
                title="Pickle weights present alongside safetensors",
                detail=(
                    f"The repo ships both formats ({len(pickle_files)} pickle-based "
                    f"file(s) and {len(by_format['safetensors'])} safetensors file(s)). "
                    f"Loaders prefer safetensors when available, so the practical risk "
                    f"is low, but the pickle files remain loadable by anything that "
                    f"names them explicitly."
                ),
                remediation="Serve only the safetensors files from the internal mirror.",
            )
        )
    elif pickle_files:
        # Only claim the opcode scan came back clean if it actually did --
        # otherwise this finding contradicts the CRITICAL sitting above it.
        opcode_clean = not any(
            f.id.startswith("weights.pickle_") and f.severity >= Severity.HIGH
            for f in result.findings
        )
        assessment = (
            "Opcode inspection found nothing dangerous, but a clean opcode scan is "
            "weaker evidence than a format that cannot execute code at all."
            if opcode_clean
            else "See the opcode findings above: this file resolves callables it "
                 "should not."
        )
        result.add(
            Finding(
                id="weights.pickle_only",
                severity=Severity.MEDIUM,
                title="Weights are pickle-based with no safetensors alternative",
                detail=(
                    f"Loading these weights runs the pickle virtual machine: "
                    f"{', '.join(pickle_files[:5])}"
                    f"{' and others' if len(pickle_files) > 5 else ''}. "
                    f"{assessment}"
                ),
                remediation=(
                    "Convert to safetensors. The mirror can do this on intake and "
                    "serve the converted artefact -- no change to researcher code."
                ),
            )
        )

    if "keras" in by_format:
        # Same reasoning as pickle-alongside-safetensors, and it must be applied
        # consistently or we produce absurd results: a TensorFlow export sitting
        # next to safetensors is not what any loader picks up. Found by scanning
        # sentence-transformers/all-MiniLM-L6-v2, which ships safetensors, a
        # torch .bin, an OpenVINO export and tf_model.h5 -- and was being held
        # over the one file nobody in a PyTorch shop will ever load.
        result.add(
            Finding(
                id="weights.keras_alongside_safetensors" if has_safetensors
                   else "weights.keras_format",
                severity=Severity.LOW if has_safetensors else Severity.HIGH,
                title="Keras/HDF5 weights present"
                      + (" alongside safetensors" if has_safetensors else ""),
                detail=(
                    f"{', '.join(by_format['keras'])}: Keras archives can embed "
                    f"Lambda layers carrying marshalled Python bytecode, which the "
                    f"framework executes on load. Opcode scanning does not cover this "
                    f"path, so we cannot clear it the way we clear a pickle."
                    + (
                        " Safetensors is present and is what loaders select, so these "
                        "files are inert in practice."
                        if has_safetensors
                        else ""
                    )
                ),
                remediation="Serve only the safetensors files from the internal mirror."
                if has_safetensors
                else "Obtain a safetensors or GGUF build, or convert on a "
                     "disposable, network-isolated host.",
            )
        )

    if "gguf" in by_format and not has_safetensors:
        result.add(
            Finding(
                id="weights.gguf_only",
                severity=Severity.LOW,
                title="GGUF weights",
                detail="GGUF is data-only, so there is no deserialisation code "
                       "execution path. Residual risk is memory-safety bugs in the "
                       "loader, which the runtime sandbox is the right control for.",
            )
        )

    if has_safetensors and not pickle_files and "keras" not in by_format:
        result.add(
            Finding(
                id="weights.safetensors_only",
                severity=Severity.INFO,
                title="Weights are safetensors only",
                detail=(
                    f"{len(by_format['safetensors'])} safetensors file(s) and no "
                    f"executable weight format. Loading these parses a length-prefixed "
                    f"header and maps raw tensor bytes; there is no callable to resolve."
                ),
            )
        )

    return result


def _looks_pickled(path: Path) -> bool:
    """Cheap check for a pickled object inside a .npy container."""
    try:
        with path.open("rb") as fh:
            return b"OBJECT" in fh.read(128).upper()
    except OSError:
        return False
