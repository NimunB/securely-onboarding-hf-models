# hfgate

**Checks a Hugging Face model before it's allowed into our environment, and says allow or
quarantine with reasons.**

Loading a model can run code that the model's author wrote —-See
[Part-1-Design.md](Part-1-Design.md) for the design document.

---

## What's in here

```
Part-1-Design.md       the design document — start here
hfgate/
  __main__.py          the CLI: `scan` and `registry` commands
  scanner.py           runs the four checks in order
  checks/              those four, plus pickle_opcodes.py which weights.py calls
  policy.py            turns findings into a tier          ← risk appetite lives here
  report.py            renders the verdict for humans and machines
  registry.py          the database and its promotion rules
fixtures/
  build_fixtures.py    generates six offline test models
  fetch_real_model.py  pulls a real repo from the Hub to scan
tests/                 44 tests
```

Two files carry most of the thinking: [`policy.py`](hfgate/policy.py), which is the entire
risk appetite in one screen, and [`checks/pickle_opcodes.py`](hfgate/checks/pickle_opcodes.py),
which is the core detection.

---

## Try it in 30 seconds

No installation, no virtualenv, no dependencies. Python 3.9 or newer.

```bash
python3 fixtures/build_fixtures.py                              # make some test models
python3 -m hfgate scan fixtures/allow-tinyllama-safetensors     # a good one
python3 -m hfgate scan fixtures/quarantine-backdoored-finetune  # a bad one
```

Run the tests with `python3 -m unittest discover -s tests -v` (44 tests, well under a second).

---

## What you get back

A model that's fine:

```
========================================================================
  ALLOW: meta-llama/TinyLlama-1.1B-Chat-fixture
  Tier 1 - Trusted
========================================================================

Why:
  - Allowlisted publisher `meta-llama` at pinned revision, data-only weight
    format, no custom code
```

A model that isn't:

```
========================================================================
  QUARANTINE: anon-research-42/custom-llama-finetune
  Tier 4 - Blocked
========================================================================

Why:
  - Pickle resolves `os.system`
  - Shipped code calls `subprocess.run`
  - Dependency `trasformers` resembles `transformers`

What this means: the model is held at the mirror, not deleted. The
findings below are the exit checklist -- resolve them (or request a
reviewed exception) and rescan.
```

Each finding below the banner comes with where it was found and how to fix it.

### The other outputs

Every scan produces the same verdict in three more forms:

| Output | What it's for |
|---|---|
| `<model>.hfgate.json`, written automatically | Machine-readable: verdict, tier, reasons, all findings, an SBOM, provenance. Feeds the registry and any pipeline. |
| Exit code: `0` allowed, `2` quarantined, `3` tool error | CI |
| A registry row, if you pass `--register` | Permanent record of the verdict and who approved it |

---

## Reading a verdict

4 tiers. The full reasoning is in [Part-1-Design.md](Part-1-Design.md#d-risk-tiering). The summary:

| Tier | Meaning | What happens |
|---|---|---|
| **1 Trusted** | Known publisher, pinned commit, safetensors, no custom code | Allowed automatically |
| **2 Standard** | Nothing can execute; publisher unknown | Allowed automatically |
| **3 Elevated** | Something *could* execute, but nothing hostile found | Runs with no network access and no credentials; needs a recorded approval |
| **4 Blocked** | Something hostile, or a file we couldn't inspect | Doesn't run |

The important idea: **we judge what a model can do, not who published it.** An unknown
author's safetensors model is allowed; a famous lab's tampered checkpoint is blocked.

---

## The test models

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
test:

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
models**

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

---

## What it checks

| Check | Question it answers |
|---|---|
| [weights.py](hfgate/checks/weights.py) | What format are the weights, and can loading that format run code? |
| [pickle_opcodes.py](hfgate/checks/pickle_opcodes.py) | If they're pickle format, do they contain instructions that call dangerous things? |
| [remote_code.py](hfgate/checks/remote_code.py) | Does the repo ship Python that gets imported on load, and what does it do? |
| [sbom.py](hfgate/checks/sbom.py) | What does the repo declare as dependencies — unpinned, typosquatted, or from a git URL? |
| [provenance.py](hfgate/checks/provenance.py) | Who published it, and is the version pinned to a commit? |

The publisher allowlist and the typosquat list are illustrative rather than maintained; a real
deployment would generate the latter from popular index names automatically.

Checks only *report* what they find. A separate file, [policy.py](hfgate/policy.py), decides
what those findings mean. 

---

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

---

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

---

What this gate does *not* cover, and the residual risk we accept, is in
[Part-1-Design.md](Part-1-Design.md#e-what-we-deliberately-chose-not-to-do).
