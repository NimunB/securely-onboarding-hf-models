"""CLI: python3 -m hfgate <command>

  scan      inspect a model repo and emit an allow/quarantine verdict
  registry  record models, ingest verdicts, move them through promotion states

Exit codes are the CI contract:
  0  allow / success
  2  quarantine
  3  usage, IO, or invalid-transition error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import DEFAULT_DB, Registry, RegistryError, State
from .report import human_summary, machine_record, write_record
from .scanner import scan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hfgate",
        description="Intake gate and registry for Hugging Face models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- scan --------------------------------------------------------------
    scan_p = sub.add_parser("scan", help="Scan a local model repo clone")
    scan_p.add_argument("path", type=Path, help="Path to the model repo directory")
    scan_p.add_argument(
        "--metadata", type=Path, default=None,
        help="Hub metadata JSON sidecar (defaults to <path>/hf_metadata.json)")
    scan_p.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the machine-readable record "
             "(default: <path>.hfgate.json next to the repo)")
    scan_p.add_argument(
        "--json", action="store_true",
        help="Print the machine-readable record to stdout instead of the summary")
    scan_p.add_argument(
        "--register", action="store_true",
        help="Ingest the resulting verdict into the registry")
    scan_p.add_argument("--db", type=Path, default=DEFAULT_DB, help="Registry database path")
    scan_p.add_argument("--actor", default="hfgate-ci", help="Who is running this scan")

    # -- registry ----------------------------------------------------------
    reg_p = sub.add_parser("registry", help="Model registry operations")
    reg_p.add_argument("--db", type=Path, default=DEFAULT_DB, help="Registry database path")
    reg_sub = reg_p.add_subparsers(dest="reg_command", required=True)

    r_register = reg_sub.add_parser("register", help="Record a model without scanning it")
    r_register.add_argument("repo_id", help="e.g. meta-llama/Llama-3-8B")
    r_register.add_argument("--revision", required=True, help="Commit SHA")
    r_register.add_argument("--actor", required=True)

    r_ingest = reg_sub.add_parser("ingest", help="Ingest a Part A scan record")
    r_ingest.add_argument("record", type=Path, help="Path to a .hfgate.json record")
    r_ingest.add_argument("--actor", default="hfgate-ci")

    r_promote = reg_sub.add_parser("promote", help="Move a model to a new state")
    r_promote.add_argument("ref", help="repo_id@revision")
    r_promote.add_argument(
        "--to", required=True,
        choices=[s.value for s in State], help="Target state")
    r_promote.add_argument("--actor", required=True)
    r_promote.add_argument("--reason", required=True)
    r_promote.add_argument(
        "--justification", default=None,
        help="Required to approve an elevated-tier model: why the risk is "
             "accepted and what compensating controls apply")

    r_list = reg_sub.add_parser("list", help="List registered models")
    r_list.add_argument("--state", default=None, choices=[s.value for s in State])
    r_list.add_argument("--tier", default=None)
    r_list.add_argument("--json", action="store_true")

    r_show = reg_sub.add_parser("show", help="Show one model with scan and audit history")
    r_show.add_argument("ref", help="repo_id@revision")

    return parser


def _split_ref(ref: str) -> tuple[str, str]:
    if "@" not in ref:
        raise RegistryError(
            f"'{ref}' must be repo_id@revision. The registry keys on the commit "
            f"because branch refs move."
        )
    repo_id, _, revision = ref.rpartition("@")
    return repo_id, revision


def _cmd_scan(args: argparse.Namespace) -> int:
    root = args.path
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 3
    if args.metadata is not None and not args.metadata.is_file():
        print(f"error: metadata file {args.metadata} not found", file=sys.stderr)
        return 3

    decision, result = scan(root, args.metadata)
    record = machine_record(root, decision, result.findings, result.facts)

    out_path = args.out or root.parent / f"{root.name}.hfgate.json"
    write_record(record, out_path)

    if args.json:
        print(json.dumps(record, indent=2, default=str))
    else:
        print(human_summary(root, decision, result.findings, result.facts))
        print(f"Machine-readable record: {out_path}")

    if args.register:
        with Registry(args.db) as reg:
            model = reg.ingest(record, args.actor)
        print(f"Registry: {model.repo_id}@{model.revision} -> "
              f"{model.state.value} (tier: {model.tier})")

    return 0 if decision.allowed else 2


def _cmd_registry(args: argparse.Namespace) -> int:
    with Registry(args.db) as reg:
        if args.reg_command == "register":
            model = reg.register(args.repo_id, args.revision, args.actor)
            print(f"{model.repo_id}@{model.revision} -> {model.state.value}")

        elif args.reg_command == "ingest":
            if not args.record.is_file():
                print(f"error: {args.record} not found", file=sys.stderr)
                return 3
            record = json.loads(args.record.read_text(encoding="utf-8"))
            model = reg.ingest(record, args.actor)
            print(f"{model.repo_id}@{model.revision} -> {model.state.value} "
                  f"(tier: {model.tier})")

        elif args.reg_command == "promote":
            repo_id, revision = _split_ref(args.ref)
            model = reg.promote(
                repo_id, revision, State(args.to), args.actor,
                args.reason, args.justification,
            )
            print(f"{model.repo_id}@{model.revision} -> {model.state.value}")

        elif args.reg_command == "list":
            models = reg.list_models(args.state, args.tier)
            if args.json:
                print(json.dumps([
                    {"repo_id": m.repo_id, "revision": m.revision,
                     "state": m.state.value, "tier": m.tier,
                     "updated_at": m.updated_at}
                    for m in models
                ], indent=2))
            elif not models:
                print("No models registered.")
            else:
                print(f"{'STATE':<12} {'TIER':<10} {'REVISION':<10} REPO")
                print("-" * 72)
                for m in models:
                    print(f"{m.state.value:<12} {(m.tier or '-'):<10} "
                          f"{m.revision[:8]:<10} {m.repo_id}")

        elif args.reg_command == "show":
            repo_id, revision = _split_ref(args.ref)
            model = reg._require(repo_id, revision)
            print(f"{model.repo_id}@{model.revision}")
            print(f"  state:      {model.state.value}")
            print(f"  tier:       {model.tier or '-'}")
            print(f"  registered: {model.registered_at}")

            scans = reg.scans_for(model.id)
            print(f"\n  Scans ({len(scans)}):")
            for s in scans:
                counts = json.loads(s["finding_counts"])
                summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none"
                print(f"    {s['scanned_at']}  {s['verdict']:<10} tier={s['tier']:<9} "
                      f"findings[{summary}]")
                print(f"      sha256={s['record_sha256'][:16]}...")
                for reason in json.loads(s["reasons"]):
                    print(f"      - {reason}")

            print("\n  Audit trail:")
            for t in reg.history(model.id):
                print(f"    {t['at']}  {t['from_state'] or '(new)'} -> {t['to_state']}  "
                      f"by {t['actor']}")
                print(f"      {t['reason']}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "scan":
            return _cmd_scan(args)
        if args.command == "registry":
            return _cmd_registry(args)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 3


if __name__ == "__main__":
    sys.exit(main())
