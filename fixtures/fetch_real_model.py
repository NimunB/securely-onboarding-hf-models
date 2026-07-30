#!/usr/bin/env python3
"""Fetch a real Hugging Face repo so the gate can be run against it.

Why this exists
---------------
The take-home does not require live Hub access ("you do not need live
HuggingFace access... We care about the decision logic"). But the fixtures and
the scanner were written by the same person, which is a closed loop: fixtures
can be unconsciously shaped to pass the checks they are meant to test, and a
green test suite would never reveal it. Running against repos we did not author
is the only thing that breaks that circularity.

It also closes a second loop. `provenance.py` reads Hub metadata from an
`hf_metadata.json` sidecar so the tool works offline. That file format is only
honest if it is genuinely what the Hub returns -- so this script writes the raw
API response to it, unmodified.

Deliberately uses urllib rather than `huggingface_hub`, to keep the repo
dependency-free.

Usage:
    python3 fixtures/fetch_real_model.py nomic-ai/nomic-embed-text-v1.5
    python3 fixtures/fetch_real_model.py hf-internal-testing/tiny-random-gpt2 --max-mb 50
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"
UA = {"User-Agent": "hfgate-fetch/0.1"}
OUT_ROOT = Path(__file__).parent / "real"

# Files we always want: they drive every check except the weights themselves.
ALWAYS = (".json", ".py", ".txt", ".md", ".toml", ".cfg", ".yaml", ".yml")
WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".gguf", ".msgpack")


def api(path: str) -> dict:
    req = urllib.request.Request(HUB + path, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def head_size(repo: str, revision: str, name: str) -> int | None:
    url = f"{HUB}/{repo}/resolve/{revision}/{urllib.request.quote(name)}"
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            # HF redirects LFS objects to a CDN; the header we want survives it.
            size = r.headers.get("x-linked-size") or r.headers.get("Content-Length")
            return int(size) if size else None
    except urllib.error.HTTPError:
        return None


def download(repo: str, revision: str, name: str, dest: Path) -> None:
    url = f"{HUB}/{repo}/resolve/{revision}/{urllib.request.quote(name)}"
    req = urllib.request.Request(url, headers=UA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as fh:
        while chunk := r.read(1 << 16):
            fh.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch a real HF repo for scanning.")
    ap.add_argument("repo_id", help="e.g. nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--revision", default=None,
                    help="commit SHA (default: resolve the current one)")
    ap.add_argument("--max-mb", type=float, default=40.0,
                    help="skip individual files larger than this (default 40)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    try:
        meta = api(f"/api/models/{args.repo_id}")
    except urllib.error.HTTPError as e:
        print(f"error: could not fetch {args.repo_id}: HTTP {e.code}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: no network access ({e}). This script needs the internet; "
              f"the offline fixtures do not.", file=sys.stderr)
        return 1

    revision = args.revision or meta.get("sha")
    if not revision:
        print("error: could not resolve a commit SHA", file=sys.stderr)
        return 1

    out = args.out or OUT_ROOT / args.repo_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)

    # The sidecar provenance.py reads: the raw API response, pinned to the
    # commit we actually fetched.
    meta["sha"] = revision
    (out / "hf_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    files = [s["rfilename"] for s in meta.get("siblings", [])]
    cap = int(args.max_mb * 1024 * 1024)
    got, skipped = [], []

    print(f"{args.repo_id} @ {revision[:12]}  ({len(files)} files)")
    for name in files:
        lower = name.lower()
        wanted = lower.endswith(ALWAYS) or lower.endswith(WEIGHTS)
        if not wanted:
            continue
        size = head_size(args.repo_id, revision, name)
        if size is not None and size > cap:
            skipped.append((name, size))
            print(f"  skip  {name}  ({size/1e6:.0f} MB > {args.max_mb:.0f} MB cap)")
            continue
        try:
            download(args.repo_id, revision, name, out / name)
            got.append(name)
            print(f"  get   {name}" + (f"  ({size/1e6:.1f} MB)" if size else ""))
        except urllib.error.HTTPError as e:
            print(f"  fail  {name}: HTTP {e.code}")

    # Being explicit about partial fetches matters: a scan that silently missed
    # the weights would report a verdict it has not earned.
    if skipped:
        note = {
            "partial_fetch": True,
            "skipped_files": [{"name": n, "bytes": s} for n, s in skipped],
            "warning": (
                "Weight files were skipped because of the size cap. Any scan of this "
                "directory is PARTIAL: the weight-format and pickle checks did not see "
                "these files. Re-fetch with a higher --max-mb for a complete verdict."
            ),
        }
        (out / "hfgate_fetch_note.json").write_text(
            json.dumps(note, indent=2), encoding="utf-8")
        print(f"\n  !! PARTIAL: {len(skipped)} weight file(s) skipped. "
              f"See hfgate_fetch_note.json")

    print(f"\nFetched {len(got)} file(s) to {out}")
    print(f"Scan it:  python3 -m hfgate scan {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
