# hfgate

Loading an untrusted model from Hugging Face is arbitrary code execution. Pickle-format
weights and repository-supplied Python both run in the loading process, with that
process's credentials, mounted data and network access, so a malicious or tampered model
repository can exfiltrate secrets and data or move laterally into the surrounding
infrastructure before the model produces any output.

hfgate is a static analyser that detects this before the model is loaded. It determines
which load-time code-execution mechanisms a repository exposes, assigns a risk tier, and
returns allow or quarantine with the evidence for that decision. It parses the
repository's files and never deserialises or imports them.

The repository contains a design document ([Part-1-Design.md](Part-1-Design.md)) and an
implementation of two components of that design: the scan gate and the model registry.

## Threat model

The setting is an organisation bringing open-weights models from Hugging Face into its
compute environment. `from_pretrained()` is not a pure data read: two documented
mechanisms let a model repository execute code in the loading process.

1. **Deserialisation.** PyTorch checkpoints (`.bin`, `.pt`, `.ckpt`) are pickle streams.
   Pickle is a stack-machine bytecode: `GLOBAL`/`STACK_GLOBAL` resolve any importable
   `module.attr` to a live object and `REDUCE` calls it. A stream that resolves `os.system`
   and reduces it runs a shell command during `torch.load`, before any tensor is read.
2. **Dynamic import.** If `config.json` contains an `auto_map` and the caller passes
   `trust_remote_code=True`, `transformers` imports Python modules from the repository
   (or from another repository that `auto_map` names) and instantiates classes from them.

| | |
|---|---|
| **Adversary** | A model publisher, or anyone able to tamper with a legitimate repository. |
| **Assets** | What the loading job can reach: cloud credentials, mounted datasets, other workloads on the cluster, outbound network access. |
| **Threats covered** | Malicious weights; malicious repository code; a repository that changes after it was checked (a Hub repository is a git repository and `main` moves). |
| **Property checked** | Loading the repository at a given commit does not execute adversary-supplied code. |
| **Out of scope** | Model behaviour: harmful outputs, jailbreaks, and backdoors trained into the weights. None of it executes at load, and none of it is detectable by the analysis used here. |

The full threat model is [section A of the design document](Part-1-Design.md#a-threat-model).

## What I did

1. **Threat model** for model intake, summarised above.
   ([design doc, section A](Part-1-Design.md#a-threat-model))
2. **System design** in five stages: mirror, scan gate, registry, runtime isolation,
   observability. The gate and the registry are implemented here; the other three are
   specified only. ([design doc, section B](Part-1-Design.md#b-system-design);
   orchestration in [`scanner.py`](hfgate/scanner.py))
3. **Pickle opcode scanner.** Parses the stream with `pickletools.genops`, extracts every
   resolved global, and classifies each against a denylist and against the set of modules
   a PyTorch state dict normally references. Handles both zip-archive and raw checkpoints.
   ([`checks/pickle_opcodes.py`](hfgate/checks/pickle_opcodes.py), called from
   [`checks/weights.py`](hfgate/checks/weights.py))
4. **Repository code check.** Resolves `auto_map`, computes the transitive closure of
   imports from the modules the loader enters, and matches denylisted calls in the AST of
   each reachable file. Falls back to the whole repository when an import is dynamic.
   ([`checks/remote_code.py`](hfgate/checks/remote_code.py))
5. **Dependency and provenance checks.** An SBOM of declared requirements (unpinned,
   VCS/URL, known misspellings), and publisher and revision-pinning metadata from the Hub.
   ([`checks/sbom.py`](hfgate/checks/sbom.py),
   [`checks/provenance.py`](hfgate/checks/provenance.py))
6. **Tiering policy.** Four tiers, decided by which execution mechanisms are present.
   Checks report findings; a separate file of about 170 lines maps findings to a tier.
   ([`policy.py`](hfgate/policy.py);
   [design doc, section D](Part-1-Design.md#d-risk-tiering))
7. **Registry.** SQLite, keyed on repository and commit. A state machine with an
   append-only audit trail; approving an Elevated model requires a recorded justification,
   Blocked cannot be overridden, and a rescan withdraws an existing approval.
   ([`registry.py`](hfgate/registry.py); [usage](#the-registry))
8. **Evaluation** on six synthetic and six real repositories.
   ([results](#evaluation); [`tests/test_hfgate.py`](tests/test_hfgate.py);
   [`fixtures/build_fixtures.py`](fixtures/build_fixtures.py))

The implementation has no third-party dependencies and runs on Python 3.9 or newer.

## Risk by artefact type

The tier is determined by which load-time execution mechanisms a repository exposes, not
by who published it.

| Artefact | Execution mechanism at load | What the scan establishes | Tier |
|---|---|---|---|
| **safetensors** | None. A length-prefixed JSON header followed by raw tensor bytes; the parser resolves no callables. | That the weights are in this format. | Standard or Trusted |
| **GGUF** | None through deserialisation. Residual risk is memory-safety defects in the native parser. | Format only. | Standard |
| **Pickle** (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`) | Arbitrary callable invocation by the pickle VM. | Which globals the stream resolves, not what they do when called. | Elevated or Blocked |
| **NumPy** (`.npy`, `.npz`) | None for numeric arrays. Object arrays embed a pickle stream. | Object arrays are detected and scanned as pickle. | As for pickle |
| **Keras / HDF5** (`.h5`, `.keras`) | `Lambda` layers can embed marshalled Python bytecode, executed when Keras rebuilds the model. | Presence only. The bytecode is not analysed. | Elevated |
| **Repository Python** (`auto_map` with `trust_remote_code=True`) | Module import: top-level code and class constructors execute. | The reachable import graph, and denylisted calls within it. | Elevated or Blocked |
| **Declared dependencies** | Not at load. Package installation executes build and install hooks. | Requirement names and specifiers. | Elevated or Blocked |

**Pickle.** A stream is Blocked if it resolves a global in a denylisted module (`os`,
`subprocess`, `socket`, …) or a denylisted callable (`eval`, `exec`, `getattr`, …), or if
it begins parsing and then fails, since it cannot be fully inspected. A global outside
the set a PyTorch state dict normally uses is Elevated. A stream with no findings is
still Elevated: the absence of a denylisted global is weaker evidence than a format with
no invocation mechanism.

**Repository Python.** A denylisted call (`subprocess.*`, `eval`, `socket.socket`, …) in
any file reachable from the modules `auto_map` names is Blocked. Everything else is
Elevated regardless of how the code reads, because AST matching does not survive
obfuscation. Dynamic imports (`importlib.import_module`, `__import__`) are themselves on
the denylist, and they also make the import graph unbounded, so when a reachable module
uses one, every Python file in the repository is treated as reachable. `auto_map` entries that point to
another repository are reported separately: that code is not scanned, and pinning this
repository's commit does not pin it.

**Dependencies.** A name on a fixed list of known misspellings of popular packages is
Blocked. A VCS or URL requirement is Elevated.

**Unused formats.** A pickle or Keras file beside safetensors weights is recorded at low
severity and does not raise the tier, on the assumption that the loader selects
safetensors. That holds for `transformers` defaults, not for a caller who names the file
explicitly. A denylisted global in such a pickle still blocks.

**No recognised weights.** Elevated. This is usually an incomplete clone (git-lfs
pointers not fetched), and the gate does not pass input it has not inspected.

| Tier | Condition | Effect |
|---|---|---|
| **1 Trusted** | No execution mechanism; allowlisted publisher; pinned commit | Allowed |
| **2 Standard** | No execution mechanism | Allowed |
| **3 Elevated** | An execution mechanism exists; no evidence of hostile use | Quarantined pending a recorded approval; intended to run without network access or credentials |
| **4 Blocked** | Evidence of hostile use, or an artefact that could not be inspected | Not run. Terminal for that commit |

## Evaluation

### The test models

`fixtures/build_fixtures.py` generates six fake model repos. Each exists to pin one
behaviour, so if you change the logic these tell you what you broke.

```bash
for f in fixtures/*/; do python3 -m hfgate scan "$f" | sed -n '2,3p'; done
```

| Fixture | Tier | What it demonstrates |
|---|---|---|
| `allow-tinyllama-safetensors` | 1 Trusted | The normal case |
| `standard-community-finetune` | 2 Standard | Unknown author still allowed — nothing can execute |
| `elevated-custom-arch` | 3 Elevated | Custom code that is genuinely fine is still contained |
| `elevated-legacy-pickle-clean` | 3 Elevated | Old pickle format → offer conversion, don't refuse |
| `quarantine-backdoored-finetune` | 4 Blocked | Malicious weights *and* malicious shipped code |
| `quarantine-trojaned-checkpoint` | 4 Blocked | **See below** |

That last one's (`quarantine-trojaned-checkpoint`) verdict should be noted. It has a well-known publisher, a pinned commit, 2.9
million downloads, clean pinned dependencies, a detailed model card, and no custom code —
every reputational signal is good. It's still blocked, because the checkpoint itself contains
an instruction to call `eval`. It exists to prove the gate isn't just a popularity check, and
a test fails if that ever stops being true.

### Running it against real Hugging Face models

```bash
python3 fixtures/fetch_real_model.py sentence-transformers/all-MiniLM-L6-v2 --max-mb 120
python3 -m hfgate scan fixtures/real/sentence-transformers__all-MiniLM-L6-v2
```

The fetcher uses `urllib` rather than `huggingface_hub`, so it stays dependency-free, and it
writes the Hub's raw API response to `hf_metadata.json` — which is the same sidecar file the
offline fixtures use, so that format is genuinely what the Hub returns rather than something
invented. Files over the size cap are skipped and recorded in `hfgate_fetch_note.json`, since
a scan that quietly missed the weights would report a verdict it hadn't earned.

What we tested:

| Repo | Tier | Why |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 1 Trusted | Allowlisted org, pinned, safetensors |
| `hf-internal-testing/tiny-random-gpt2` | 2 Standard | Safetensors present, publisher unknown |
| `nomic-ai/nomic-embed-text-v1.5` | 3 Elevated | Genuinely requires `trust_remote_code` |
| `jinaai/jina-embeddings-v2-small-en` | 3 Elevated | Same |
| `prajjwal1/bert-tiny` | 3 Elevated | Pickle-only, pre-safetensors era |
| `sshleifer/tiny-gpt2` | 3 Elevated | Pickle-only plus a TF export |

**This found four real bugs that the 30 passing tests did not.** Each now has a regression
test, and the suite has since grown to 44:

1. **`auto_map` can point at a *different repo*.** `nomic-ai/nomic-embed-text-v1.5` sources
   all seven classes from `nomic-ai/nomic-bert-2048` via the `repo--module.Class` syntax. The
   check called this "code from the model repo itself" — wrong — and advised pinning a
   commit, which *doesn't work here*: pinning this repo doesn't pin the other one. Now flagged
   as its own finding, because a clean verdict on a repo whose code lives elsewhere has no
   shelf life.
2. **`.bin` is not owned by PyTorch.** OpenVINO ships `openvino_model.bin`. Treating every
   `.bin` as a pickle **blocked all-MiniLM-L6-v2**, one of the most-downloaded models on the
   Hub. Now distinguished by whether *any* opcode parses at byte 0 — nothing parsed means it
   was never a pickle and `torch.load` can't load it either. A pickle that starts valid and
   breaks later is still held.
3. **A shipped training script is not a load path.** all-MiniLM ships `train_script.py`,
   which `from_pretrained` never imports. Tiering on its presence penalised a repo for
   documenting itself well. Severity now depends on whether the loader actually imports the
   file — see the note below, because the first version of this fix was wrong.
4. **Unused formats were treated inconsistently.** A pickle next to safetensors was already
   downgraded; a Keras export next to safetensors wasn't. Both are inert — the loader picks
   safetensors either way.

Three of those four were **false positives that would have blocked legitimate, popular
models.**

#### The fix that was wrong, and what replaced it

Fix 3 above scoped severity to files whose *names* match the convention transformers imports.
That is pattern-matching on filenames, not on the threat, and it left an obvious hole — so I
built the attack against my own fix:

```
modeling_evil.py     ← matches the convention, gets scanned, looks clean
  └── imports utils_helper.py   ← doesn't match, so "not imported on load"
                                  ...and that is where the subprocess call lives
```

The helper executes. The tool called it LOW and printed *"loading the model does not execute
it"*, which was simply false. Fixing one false positive had created a false negative.

**Invocation is now the transitive closure of imports** from the files the loader actually
enters, not a filename test — so a payload one hop away still blocks, and `train_script.py`
is still correctly ignored.

That raised a second question: static analysis only bounds the import graph while the imports
are *static*. One `importlib.import_module(name)` and we can't prove any file is unreachable.
So when loaded code imports dynamically, every Python file in the repo is treated as reachable
and the scan says why, rather than continuing to assert "this one doesn't run" on evidence it
no longer has. Four tests cover this: the bypass, the relative-import form, an import cycle,
and the dynamic-import fallback.

## Limitations

- **The pickle analysis is a denylist over resolved globals.** It does not model what a
  permitted callable does with attacker-controlled arguments, so a gadget chain built from
  modules outside the denylist is not detected. This is why a clean pickle is Elevated and
  not allowed.
- **The code analysis is syntactic.** It matches call names in the AST. Aliasing,
  string-built attribute access and encoded payloads can evade it. Containment at run time
  is the control for Elevated models; the static result only distinguishes Elevated from
  Blocked.
- **Code in other repositories is not followed.**
- **Keras bytecode and native parser defects are not analysed.**
- **Backdoors trained into the weights are not detected.** A model with one can be tiered
  Trusted. The tier is a statement about load-time code execution only.
- **The mirror, runtime isolation and observability components of the design are not
  implemented.** The Elevated tier's isolation is specified, not enforced, by this code.
- **The publisher allowlist and the typosquat list are illustrative**, not maintained.

The design document discusses the accepted residual risk in
[section E](Part-1-Design.md#e-what-we-deliberately-chose-not-to-do).

## Running it

No installation is needed.

```bash
python3 fixtures/build_fixtures.py                              # generate the six fixtures
python3 -m hfgate scan fixtures/allow-tinyllama-safetensors     # Tier 1, allowed
python3 -m hfgate scan fixtures/quarantine-backdoored-finetune  # Tier 4, quarantined
python3 -m unittest discover -s tests -v                        # 44 tests
```

Output for the backdoored fixture, followed by each finding with its location and
remediation:

```
========================================================================
  QUARANTINE: anon-research-42/custom-llama-finetune
  Tier 4 - Blocked
========================================================================

Why:
  - Pickle resolves `os.system`
  - Shipped code calls `subprocess.run`
  - Dependency `trasformers` resembles `transformers`
```

Each scan also writes `<model>.hfgate.json` (verdict, tier, reasons, all findings, an
SBOM and provenance) and sets an exit code: `0` allowed, `2` quarantined, `3` tool error.

## The registry

The scanner decides — [`scanner.py`](hfgate/scanner.py) runs the checks,
[`policy.py`](hfgate/policy.py) turns the results into a tier. The registry remembers —
[`registry.py`](hfgate/registry.py) stores what models exist, their scan results, their tier,
and who approved what.

```bash
# scan and record in one step
python3 -m hfgate scan fixtures/elevated-custom-arch --register --db demo.db --actor ci-bot

python3 -m hfgate registry --db demo.db list
python3 -m hfgate registry --db demo.db show "<repo_id>@<commit>"
```

Approving an Elevated model requires you to say why, and that gets stored:

```bash
python3 -m hfgate registry --db demo.db promote \
  "allenai/mamba-hybrid-370m-fixture@b7e2c91d4a68f3057e1b8c2d9f4a6053e8c7b1d2" \
  --to approved --actor n.bajwa --reason "SSM eval workstream" \
  --justification "Code reviewed at this commit; runs with no egress. Expires 2026-10-01."
```

Without `--justification` that command is refused.

**States.** `registered → scanned → approved`, with `quarantined`, `blocked`, `deprecated` and
`revoked` alongside. `blocked` and `revoked` are terminal. Invalid moves are refused with the
list of what's allowed from where you are.

**Every command:**

| Command | What it does |
|---|---|
| `scan <path> --register --db D --actor A` | Scan and file the verdict in one step |
| `registry --db D register <repo> --revision <sha> --actor A` | Record a model without scanning it |
| `registry --db D ingest <record.json> --actor A` | File a verdict from a previous `scan --out` |
| `registry --db D promote <repo>@<sha> --to <state> --actor A --reason R [--justification J]` | Move a model through the state machine |
| `registry --db D list [--state S] [--tier T] [--json]` | List models, optionally filtered |
| `registry --db D show <repo>@<sha>` | Full detail: every scan, and the audit trail |

Revoking an approved model during an incident:

```bash
python3 -m hfgate registry --db demo.db promote "<repo>@<sha>" \
  --to revoked --actor sec-oncall --reason "upstream compromise disclosed"
```

Four rules that were implemented:

- **Models are identified by repo *and commit*, never the name alone.** Hugging Face repos are
  git repos and `main` moves, so a name on its own identifies something that can change.
- **Blocked is final.** There is no `unblock` command, and `promote --to approved` refuses on
  a blocked record no matter what justification you give it. Blocked means we found something
  hostile in *these exact bytes* — a checkpoint that calls `os.system`, code that shells out —
  and no amount of business need changes what's in the file.

  This is the one place the system won't negotiate, and that's deliberate. Every other
  restriction here has an escape hatch, because escape hatches are what stop people building
  their own. But a "just this once" override on a known-malicious artefact is a pressure valve
  that only ever gets used under deadline, which is exactly when you least want it.

  **The way out is a different artefact, not a different decision.** If the publisher fixes
  the model, that's a new commit; a new commit is a new registry record, gets its own scan,
  and can be approved on its own merits. The blocked record stays in the database as history.
  If you think a block is a false positive, the fix is a change to
  [`policy.py`](hfgate/policy.py) with a test — a bug report, not an exception request.
- **Re-scanning an approved model drops it back to "scanned".** A model that changed shouldn't
  silently keep its approval.
- **Revoking works from any state**, because incident response must never be blocked by
  process.

## Running it automatically (CI)

**What this is for.** The design's end state is that the mirror runs the gate invisibly, so
nobody thinks about it. Until that exists, the practical way to make the check automatic is to
run it in CI — the pipeline that already runs on your repo when you push. If a project commits
a model, or a script that pulls a specific model, CI can scan it on every change, so a bad
model is caught at review time rather than in a training run at 3am.

This is the fallback, not the paved road. It only helps teams whose model use goes through a
repo. See [Part-1-Design.md](Part-1-Design.md#c-the-paved-road) for why the mirror is the real
answer.

**How it plugs in.** The tool is designed for this: it prints a human summary, writes a JSON
record, and — the part CI actually keys on — sets an exit code.

| Exit code | Meaning | CI behaviour |
|---|---|---|
| `0` | Allowed (Trusted or Standard) | Step passes, pipeline continues |
| `2` | Quarantined (Elevated or Blocked) | Step fails, pipeline stops |
| `3` | Tool error — bad path, unreadable metadata | Step fails; this is a broken job, not a bad model |

Because non-zero exit codes fail a step by default, you don't need to write any logic to make
the gate enforce itself.

**GitHub Actions:**

```yaml
- name: Model intake gate
  run: |
    python3 -m hfgate scan ./model-clone --out scan.json
    python3 -m hfgate registry ingest scan.json --actor "${{ github.actor }}"
```

The first line scans and writes the machine-readable record. The second files that verdict in
the registry, tagged with whoever pushed the change, so there's a durable answer to "who
brought this model in". If the scan quarantines, the job fails on exit code 2 and the second
line never runs.

Note there's nothing to install first — no `pip install` step, because the tool has no
dependencies. That matters more in CI than anywhere else: it's one less thing to cache,
version, or have break on a runner image update.

**Keeping the record but not blocking**, useful when introducing the gate to a team that
hasn't adopted it yet — you get the data and the warnings without failing anyone's build on
day one:

```yaml
- name: Model intake gate (report only)
  continue-on-error: true
  run: python3 -m hfgate scan ./model-clone --register --db registry.db --actor "${{ github.actor }}"
```

**Scanning several models**, failing the job if any is quarantined:

```bash
fail=0
for m in models/*/; do
  python3 -m hfgate scan "$m" || fail=1
done
exit $fail
```

## Repository layout

```
Part-1-Design.md       design document
hfgate/
  __main__.py          CLI: `scan` and `registry`
  scanner.py           runs the checks in order
  checks/              weights, pickle_opcodes, remote_code, sbom, provenance
  policy.py            maps findings to a tier
  report.py            terminal and JSON output
  registry.py          SQLite registry and its state machine
fixtures/
  build_fixtures.py    generates the six synthetic repositories
  fetch_real_model.py  fetches a real repository from the Hub
tests/test_hfgate.py   44 tests
```
