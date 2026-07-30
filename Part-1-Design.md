# Securely onboarding Hugging Face models

**Who this is for:** researchers who pull open-weights models from Hugging Face, and the
engineers who support them. No security background assumed.

**Motivation:** when you call `from_pretrained()`, you may be running code written by whoever
uploaded the model. We want that checked before it runs, without changing how you work. In the
normal case you will not notice this system exists.

---

## A. Threat model

Downloading a model feels like downloading data. It isn't — two documented, everyday
mechanisms make a model repo run code on your machine.

**1. The weights file can be a program.** Most PyTorch checkpoints (`.bin`, `.pt`, `.ckpt`)
use Python's `pickle`, which is not a data format but a small program Python executes to
rebuild the saved object — and it can contain any instruction, including "run this shell
command". `torch.load()` on an untrusted `.bin` is closer to piping a downloaded script into
your shell than to opening JSON, and **it runs during loading**, before the model emits a
token, so anything you planned to do afterwards happens too late. `safetensors` fixes exactly
this: a header and raw numbers, with no mechanism for calling code.

**2. The repo can ship its own Python.** Architectures not yet in `transformers` include `.py`
files and point at them from `config.json`; `trust_remote_code=True` imports and runs them.
Nobody at Hugging Face reviews that code.

Either way someone else's code runs inside your job, with your job's access: the pod's AWS
credentials, mounted datasets, other work on the cluster, the open internet. Concretely:
**a tampered checkpoint on a legitimate-looking repo**; **a repo changing after someone checked
it**, since a Hugging Face repo is a git repo and `main` moves; **credentials and data leaving
the cluster**; and **agents with real tools**, where "the weights look odd" becomes "something
took actions".

**What we are not worried about.** Intake asks one question — does *loading* this compromise
the platform — so a lot is deliberately out of scope. **How the model behaves** is the biggest:
bias, jailbreaks and harmful output are what AISI exists to measure, and blocking models for
being unsafe to talk to would break the mission. Also out: **licensing**, which we record but do
not gate on; **hidden behaviour trained into the weights**, which no static check can find (see
section E); and **model quality**, which is a research question, not a security one. This gate
is not an endorsement of a model — only a statement that loading it won't hand someone our
cluster.

---

## B. System design

```
 RESEARCHER ─▶ 1. MIRROR ─▶ 2. SCAN GATE ─▶ 3. REGISTRY ─▶ 4. RUNTIME ─▶ 5. OBSERVABILITY
  unchanged     serve from    tier + reasons   source of      tier sets      which model ran
  code; base    cache, or     out: summary,    truth; audit   pod egress     where, under
  image sets    fetch + ask   JSON record,     trail of who   + mounted      whose identity
  HF_ENDPOINT   the gate      CI exit code     approved what  credentials
                [design]      [BUILT · 2A]     [BUILT · 2B]   [design]       [design]
```

Pointing `HF_ENDPOINT` at the **mirror** in our base images means no research code changes, and
it is the natural place to convert pickle weights to `safetensors` so the safer format costs
nothing. The **gate** reads headers and metadata, not tensor data, so it takes seconds. The
**registry** is what makes the gate a control rather than a script someone must remember to
run. Pieces 2 and 3 are runnable in this repo; 1, 4 and 5 are designed here only.

---

## C. The paved road

**Nothing changes for you.** You write the line you write today:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

Popular models are pre-warmed and approved, so this is a cache hit from inside our network —
**faster than the public internet.** Security showing up as a speed improvement is the goal.

First to use a model? The gate runs while it downloads: seconds for a safetensors repo. Clean
means you never see a message. Otherwise the reasons arrive in the error rather than a ticket
queue, and most are self-service — pick the safetensors branch, pin a commit. If you genuinely
need a flagged model, you say why, it is logged, you carry on. That exception path is official
on purpose, because if it isn't, people build an unofficial one.

**The constraint behind all of it:** a tool people must remember to run is a tool that gets
skipped, so the control lives in the mirror everyone already goes through.

---

## D. Risk tiering

We tier on **what a model can do, not who published it.** An unknown student's `safetensors`
model is allowed, because that format cannot execute code no matter who uploaded it; a famous
lab's tampered `.bin` is blocked, because reputation does not disarm a payload.

**How a tier is decided.** Four checks (weight format, shipped code, dependencies, provenance)
report observations with severities. Policy then asks four questions in order, first "yes"
wins — that ordering is the whole policy, in one file ([`hfgate/policy.py`](hfgate/policy.py)):

1. Hostile evidence, or a file we couldn't inspect — malicious pickle instructions, shipped code
   calling `subprocess`, a typosquatted dependency, a corrupt archive? → **Blocked**
2. Could code run at load — pickle weights, `trust_remote_code`, git-URL dependencies — with
   nothing hostile found? → **Elevated**
3. No weights found at all, usually an incomplete download? → **Elevated**
4. Otherwise nothing can execute → **Trusted** if an allowlisted publisher at a pinned commit,
   else **Standard**

| Tier | What you get |
|---|---|
| **Trusted** / **Standard** | Normal namespace and egress; approved automatically |
| **Elevated** | Same model, smaller blast radius: no egress, no mounted credentials, no shared storage. Recorded, time-boxed approval |
| **Blocked** | Does not run. Terminal — a publisher's fix is a new commit, so a new decision |

**Elevated is not a refusal** — most genuinely novel architectures need custom code and land
here; they run, just fenced. **Quarantine is a queue, not a bin:** only Blocked is terminal.

**Where decisions come out.** Every scan emits the verdict four ways: a terminal summary with
reasons and fixes, for the researcher; a JSON record with every finding, an SBOM and provenance,
for the mirror, registry or any pipeline; an exit code (`0` allow, `2` quarantine, `3` error)
for CI; and a registry row recording who approved it and why, for audit. Registry verdicts are
append-only, so "what did we believe then, and who acted" stays answerable.

---

## E. What we deliberately chose not to do

Scope boundaries of the system, not a backlog. Each traces to the threat model in section A,
and each leaves real risk on the table — naming that risk is the point.

**We don't judge how a model behaves.** Measuring behaviour is AISI's actual work and belongs
downstream in evaluation. *Residual risk:* a biased or jailbreakable model passes unremarked,
and "Trusted" may be misread as a claim about outputs. It is not one.

**We don't detect backdoors trained into the weights.** Both mechanisms in section A execute at
load; a trigger-phrase backdoor is neither, and finding it statically would mean already knowing
the trigger. *Residual risk:* a poisoned model scans clean and reaches Trusted. The alternative
is a check that pretends to work, and a control nobody trusts is worse than none.

**We don't try to out-analyse obfuscated code.** *Residual risk:* hostile remote code that hides
its intent reads as clean — which is why `auto_map` alone sends a model to Elevated however good
the code looks. Containment is the control; analysis is a bonus.

**We don't follow code into other repositories.** `auto_map` can source classes from a different
repo, and real models do. *Residual risk:* the code that actually runs is unscanned, and pinning
our revision doesn't pin it, so a clean verdict has no shelf life. The gap I'd close first.

**We don't put a human in the loop by default.** A review queue becomes the thing people design
around. *Residual risk:* most models are tiered with no human judgement — but automated tiering
plus a recorded self-service exception is enforceable, and a review board nobody uses is not.

**We don't govern what a model does once running.** *Residual risk:* everything after load,
including an agent that is prompt-injected rather than backdoored — a runtime concern with
different controls and a different owner.

**Taken together, the risk we knowingly accept:** trigger-phrase backdoors in weights;
obfuscated remote code that clears the static check; linked-repo code that changes under our
pinned revision; a compromised allowlisted publisher shipping clean-scanning safetensors; and
the runtime behaviour of any approved model.
