"""Pickle opcode scanning.

This is the heart of the gate, so it is worth being precise about why it
exists and what it does.

`torch.load` on a .bin/.pt/.ckpt file runs the pickle virtual machine. Two
opcodes matter: GLOBAL/STACK_GLOBAL, which resolve `module.attr` to a live
Python object, and REDUCE, which calls it. A malicious checkpoint is usually
nothing more exotic than `os.system` pushed by GLOBAL and invoked by REDUCE.
That is the entire trick, and it fires during load, before a single tensor is
read and long before any "is this model safe" evaluation could run.

We inspect with `pickletools.genops`, which parses the opcode stream and never
executes it. That distinction is the whole reason this file is safe to point
at hostile input: we are reading the bytecode, not running it.

Limits, stated honestly: opcode inspection tells you which callables a stream
*resolves*, not what happens afterwards. An attacker who can reach code
execution through a module we consider benign will not show up here. This is a
high-value, cheap check, not a proof of safety -- which is exactly why policy
still pushes pickle formats toward conversion rather than blessing a clean
scan as equivalent to safetensors.
"""

from __future__ import annotations

import io
import pickletools
import zipfile
from pathlib import Path

from ..findings import CheckResult, Finding, Severity

# Modules whose presence in a checkpoint means code execution, full stop.
# There is no legitimate reason for a tensor archive to resolve `os.system`.
DANGEROUS_MODULES = {
    "os",
    "nt",
    "posix",
    "subprocess",
    "sys",
    "shutil",
    "socket",
    "pty",
    "runpy",
    "importlib",
    "commands",
    "webbrowser",
    "ctypes",
    "multiprocessing",
    "asyncio",
    "http",
    "urllib",
    "urllib2",
    "requests",
    "ftplib",
    "telnetlib",
    "smtplib",
    "pickle",
    "dill",
    "cloudpickle",
    "shelve",
    "code",
    "codeop",
    "timeit",
    "bdb",
    "pdb",
    "venv",
    "platform",
    "tempfile",
    "glob",
}

# Specific callables that are dangerous even from otherwise-ordinary modules.
DANGEROUS_CALLABLES = {
    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "compile"),
    ("builtins", "open"),
    ("builtins", "__import__"),
    ("builtins", "getattr"),
    ("builtins", "setattr"),
    ("builtins", "breakpoint"),
    ("builtins", "input"),
    ("__builtin__", "eval"),
    ("__builtin__", "exec"),
    ("__builtin__", "compile"),
    ("__builtin__", "open"),
    ("__builtin__", "__import__"),
    ("operator", "attrgetter"),
    ("operator", "methodcaller"),
    ("functools", "partial"),
    ("base64", "b64decode"),
    ("codecs", "decode"),
    ("numpy", "load"),
    ("numpy.lib.npyio", "load"),
    ("torch", "load"),
    ("torch.serialization", "load"),
    ("torch.hub", "load"),
    ("torch.jit", "load"),
    ("pandas", "read_pickle"),
}

# Modules a genuine PyTorch state dict is expected to touch. Anything outside
# this set is not automatically malicious, but it is unexpected, and unexpected
# is worth a human look.
EXPECTED_MODULES = {
    "collections",
    "torch",
    "torch._utils",
    "torch.storage",
    "torch.nn",
    "torch.nn.modules",
    "torch.nn.parameter",
    "torch._tensor",
    "numpy",
    "numpy.core.multiarray",
    "numpy._core.multiarray",
    "numpy.core.numeric",
    "numpy.dtype",
    "_codecs",
    "argparse",
    "builtins",
    "__builtin__",
    "copy_reg",
    "copyreg",
    "fractions",
    "transformers",
    "tokenizers",
    "sentencepiece",
    "peft",
    "safetensors",
}

# Opcodes that invoke or construct. A resolved global is inert until one of
# these runs, so we report them as corroborating evidence.
INVOKING_OPCODES = {"REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD"}

# Pickle members inside a torch zip archive.
_PICKLE_MEMBER_SUFFIXES = (".pkl", ".pickle")

# Raw-pickle weight suffixes (legacy, non-zip torch saves and friends).
PICKLE_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".model"}


def _module_root(module: str) -> str:
    return module.split(".")[0]


def _classify(module: str, name: str) -> tuple[Severity, str] | None:
    """Return (severity, why) for a resolved global, or None if unremarkable."""
    root = _module_root(module)

    if (module, name) in DANGEROUS_CALLABLES:
        return (
            Severity.CRITICAL,
            f"resolves `{module}.{name}`, which grants code execution or "
            f"arbitrary file/network access during load",
        )

    if root in DANGEROUS_MODULES or module in DANGEROUS_MODULES:
        return (
            Severity.CRITICAL,
            f"resolves `{module}.{name}`; module `{root}` has no legitimate "
            f"role in a tensor archive and provides code execution, process "
            f"control, or network access",
        )

    if module in EXPECTED_MODULES or root in EXPECTED_MODULES:
        return None

    return (
        Severity.HIGH,
        f"resolves `{module}.{name}`, which is outside the set of modules a "
        f"PyTorch state dict is expected to touch",
    )


def scan_pickle_stream(data: bytes, where: str) -> CheckResult:
    """Walk one pickle stream's opcodes without executing it."""
    result = CheckResult()
    globals_found: list[tuple[str, str]] = []
    invoking: set[str] = set()
    # STACK_GLOBAL takes its module and name from the two most recent string
    # pushes rather than from its own argument, so we track recent constants.
    recent_strings: list[str] = []

    # Counts how many opcodes parsed before any failure. This distinguishes
    # "a pickle that broke" from "not a pickle at all", which turns out to
    # matter a great deal -- see the except block.
    opcodes_read = 0

    try:
        for opcode, arg, _pos in pickletools.genops(io.BytesIO(data)):
            opcodes_read += 1
            name = opcode.name

            if name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "STRING",
                        "BINSTRING", "SHORT_BINSTRING", "BINBYTES",
                        "SHORT_BINBYTES"):
                if isinstance(arg, (str, bytes)):
                    text = arg.decode("utf-8", "replace") if isinstance(arg, bytes) else arg
                    recent_strings.append(text)
                    del recent_strings[:-8]

            elif name in ("GLOBAL", "INST"):
                # Argument is "module attr" (INST carries it before its args).
                if isinstance(arg, str) and " " in arg:
                    module, _, attr = arg.partition(" ")
                    globals_found.append((module, attr))

            elif name == "STACK_GLOBAL":
                if len(recent_strings) >= 2:
                    globals_found.append((recent_strings[-2], recent_strings[-1]))
                else:
                    globals_found.append(("<unresolved>", "<unresolved>"))

            if name in INVOKING_OPCODES:
                invoking.add(name)

    except Exception as exc:  # noqa: BLE001 - malformed input is itself a signal
        if opcodes_read == 0:
            # Nothing parsed at all, so this was never a pickle stream. The .bin
            # extension is not owned by PyTorch: OpenVINO ships
            # `openvino_model.bin` (IR weights), and other toolchains reuse it
            # too. Found by scanning sentence-transformers/all-MiniLM-L6-v2,
            # one of the most-downloaded models on the Hub, which this check
            # originally BLOCKED outright.
            #
            # The safety argument for downgrading: risk follows whatever will
            # actually load the file. `torch.load` requires a valid pickle (or
            # zip) from byte zero, so a file that yields no opcodes cannot reach
            # the pickle VM at all. It is not a deserialisation threat, and
            # calling it one produces exactly the false positive that teaches
            # researchers to route around the gate.
            result.add(
                Finding(
                    id="weights.unrecognised_binary",
                    severity=Severity.LOW,
                    title=f"Binary file in an unrecognised format: {where}",
                    detail=(
                        f"{where} uses a weight-file extension but is not a pickle "
                        f"stream -- no opcode parsed at byte 0 ({exc.__class__.__name__}). "
                        f"Commonly this is another toolchain reusing `.bin` (OpenVINO IR "
                        f"weights, for example). It cannot be loaded by torch.load, so "
                        f"it presents no pickle deserialisation risk, but we also cannot "
                        f"say what it does contain."
                    ),
                    where=where,
                    remediation="No action needed if the format is expected for this repo "
                                "(e.g. an OpenVINO export).",
                )
            )
        else:
            # Parsing began and then failed: this really is a pickle, and it is
            # truncated or malformed. That we cannot finish inspecting it is
            # the whole reason to hold it.
            result.add(
                Finding(
                    id="weights.pickle_unparseable",
                    severity=Severity.HIGH,
                    title="Pickle stream could not be parsed",
                    detail=(
                        f"{where} began parsing as a pickle ({opcodes_read} opcode(s) "
                        f"read) and then failed ({exc.__class__.__name__}: {exc}). A "
                        f"checkpoint we cannot fully inspect is one we cannot clear; "
                        f"truncation and deliberate malformation both land here."
                    ),
                    where=where,
                    remediation="Re-download the file and rescan. If it still fails to "
                                "parse, treat the artefact as untrusted.",
                )
            )
        return result

    for module, attr in globals_found:
        verdict = _classify(module, attr)
        if verdict is None:
            continue
        severity, why = verdict
        corroboration = (
            f" The stream also contains {'/'.join(sorted(invoking))}, which invokes "
            f"or constructs from resolved globals."
            if invoking
            else ""
        )
        result.add(
            Finding(
                id="weights.pickle_dangerous_global"
                if severity is Severity.CRITICAL
                else "weights.pickle_unexpected_global",
                severity=severity,
                title=f"Pickle resolves `{module}.{attr}`",
                detail=(
                    f"{where} {why}. This executes when the checkpoint is loaded "
                    f"(torch.load / from_pretrained), before any tensor is read."
                    f"{corroboration}"
                ),
                where=where,
                remediation=(
                    "Do not load this checkpoint. Obtain a safetensors build from "
                    "the publisher, or reproduce the weights from a trusted source."
                ),
            )
        )

    result.facts[f"pickle_globals::{where}"] = [f"{m}.{a}" for m, a in globals_found]
    return result


def scan_weight_file(path: Path, rel: str) -> CheckResult:
    """Scan a pickle-bearing weight file, handling both zip and raw layouts.

    Modern torch.save writes a zip archive containing data.pkl; the legacy
    format is a bare pickle stream. Both reach the same VM.
    """
    result = CheckResult()

    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as zf:
                members = [
                    n for n in zf.namelist()
                    if n.endswith(_PICKLE_MEMBER_SUFFIXES)
                ]
                if not members:
                    result.add(
                        Finding(
                            id="weights.zip_no_pickle",
                            severity=Severity.LOW,
                            title="Zip archive contains no pickle member",
                            detail=f"{rel} is a zip archive with no .pkl member; "
                                   f"nothing to inspect at the opcode level.",
                            where=rel,
                        )
                    )
                for member in members:
                    with zf.open(member) as fh:
                        result.merge(
                            scan_pickle_stream(fh.read(), f"{rel}!{member}")
                        )
        except zipfile.BadZipFile as exc:
            result.add(
                Finding(
                    id="weights.bad_archive",
                    severity=Severity.HIGH,
                    title="Weight archive is malformed",
                    detail=f"{rel} looked like a zip but failed to open ({exc}).",
                    where=rel,
                )
            )
    else:
        result.merge(scan_pickle_stream(path.read_bytes(), rel))

    return result
