"""Model registry: the source of truth the mirror answers from.

Why this and not the runtime sandbox (Part C)? Because the mirror needs
something to ask at request time, and without it the scan gate is a tool
someone has to remember to run. The registry is what turns a scan into a
control. It is also where the answer to "which jobs used the model we just
found out is bad" lives, and that question is the one that actually gets asked
during an incident.

Design commitments:

  Identity is (repo_id, revision), never repo_id alone. A HF repo is a git
  repo; `main` moves. Registering a name without a commit would record a fact
  that expires silently, which is worse than recording nothing.

  Verdicts are immutable and append-only. A rescan adds a row, it does not
  update one. When a model is found bad six months from now we need the
  history, including what we believed at the time and who acted on it.

  Promotion is a state machine with explicit transitions, and every transition
  records an actor and a reason. "Who approved this and why" is the first
  question in any post-incident review.

  Human overrides are first-class, not a back door. A researcher with a real
  need for an Elevated model gets there through `approve --justification`,
  which is recorded. Making the exception path official is what stops people
  building an unofficial one.
"""

from __future__ import annotations

import datetime as _dt
import enum
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB = Path("registry.db")


class State(str, enum.Enum):
    REGISTERED = "registered"      # known to exist, not yet scanned
    SCANNED = "scanned"            # verdict recorded, no decision taken
    APPROVED = "approved"          # cleared for use at its tier
    QUARANTINED = "quarantined"    # held, exit path available
    BLOCKED = "blocked"            # terminal deny
    DEPRECATED = "deprecated"      # was approved, superseded; existing jobs may drain
    REVOKED = "revoked"            # was approved, now must not run anywhere


# Transitions we allow, and nothing else. Notably:
#   BLOCKED is terminal. Getting out requires a new scan of a new revision,
#   which is a new registry row -- there is no "unblock" verb by design.
#   APPROVED -> REVOKED is always available, because incident response must
#   never be blocked by a workflow rule.
TRANSITIONS: dict[State, set[State]] = {
    State.REGISTERED: {State.SCANNED, State.BLOCKED},
    State.SCANNED: {State.APPROVED, State.QUARANTINED, State.BLOCKED},
    State.QUARANTINED: {State.APPROVED, State.BLOCKED, State.SCANNED},
    State.APPROVED: {State.DEPRECATED, State.REVOKED, State.SCANNED},
    State.DEPRECATED: {State.REVOKED, State.APPROVED},
    State.BLOCKED: set(),
    State.REVOKED: set(),
}

# Tiers that may be approved without a recorded human justification. Elevated
# always needs a name against it; that is the whole point of the tier.
AUTO_APPROVABLE_TIERS = {"trusted", "standard"}


SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id           INTEGER PRIMARY KEY,
    repo_id      TEXT NOT NULL,
    revision     TEXT NOT NULL,
    state        TEXT NOT NULL,
    tier         TEXT,
    registered_at TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE (repo_id, revision)
);

CREATE TABLE IF NOT EXISTS scans (
    id          INTEGER PRIMARY KEY,
    model_id    INTEGER NOT NULL REFERENCES models(id),
    verdict     TEXT NOT NULL,
    tier        TEXT NOT NULL,
    reasons     TEXT NOT NULL,
    finding_counts TEXT NOT NULL,
    record_sha256 TEXT NOT NULL,
    record      TEXT NOT NULL,
    scanned_at  TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transitions (
    id          INTEGER PRIMARY KEY,
    model_id    INTEGER NOT NULL REFERENCES models(id),
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    actor       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scans_model ON scans(model_id);
CREATE INDEX IF NOT EXISTS idx_transitions_model ON transitions(model_id);
"""


class RegistryError(Exception):
    """Raised for invalid transitions and missing records."""


@dataclass
class ModelRow:
    id: int
    repo_id: str
    revision: str
    state: State
    tier: str | None
    registered_at: str
    updated_at: str


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Registry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- lookup ------------------------------------------------------------
    def get(self, repo_id: str, revision: str) -> ModelRow | None:
        row = self.conn.execute(
            "SELECT * FROM models WHERE repo_id = ? AND revision = ?",
            (repo_id, revision),
        ).fetchone()
        return self._row_to_model(row) if row else None

    def _require(self, repo_id: str, revision: str) -> ModelRow:
        model = self.get(repo_id, revision)
        if model is None:
            raise RegistryError(
                f"{repo_id}@{revision} is not registered. Register it or ingest a "
                f"scan record for it first."
            )
        return model

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ModelRow:
        return ModelRow(
            id=row["id"],
            repo_id=row["repo_id"],
            revision=row["revision"],
            state=State(row["state"]),
            tier=row["tier"],
            registered_at=row["registered_at"],
            updated_at=row["updated_at"],
        )

    def list_models(self, state: str | None = None, tier: str | None = None) -> list[ModelRow]:
        query = "SELECT * FROM models"
        clauses, params = [], []
        if state:
            clauses.append("state = ?")
            params.append(state)
        if tier:
            clauses.append("tier = ?")
            params.append(tier)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY repo_id, revision"
        return [self._row_to_model(r) for r in self.conn.execute(query, params)]

    def scans_for(self, model_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM scans WHERE model_id = ? ORDER BY ingested_at DESC, id DESC",
            (model_id,),
        ))

    def latest_scan(self, model_id: int) -> sqlite3.Row | None:
        scans = self.scans_for(model_id)
        return scans[0] if scans else None

    def history(self, model_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM transitions WHERE model_id = ? ORDER BY id",
            (model_id,),
        ))

    # -- mutation ----------------------------------------------------------
    def register(self, repo_id: str, revision: str, actor: str,
                 reason: str = "initial registration") -> ModelRow:
        existing = self.get(repo_id, revision)
        if existing:
            return existing

        now = _now()
        cur = self.conn.execute(
            "INSERT INTO models (repo_id, revision, state, tier, registered_at, updated_at) "
            "VALUES (?, ?, ?, NULL, ?, ?)",
            (repo_id, revision, State.REGISTERED.value, now, now),
        )
        self.conn.execute(
            "INSERT INTO transitions (model_id, from_state, to_state, actor, reason, at) "
            "VALUES (?, NULL, ?, ?, ?, ?)",
            (cur.lastrowid, State.REGISTERED.value, actor, reason, now),
        )
        self.conn.commit()
        return self._require(repo_id, revision)

    def ingest(self, record: dict, actor: str = "hfgate-ci") -> ModelRow:
        """Ingest a Part A scan record, registering the model if needed."""
        target = record.get("target", {})
        repo_id = target.get("repo_id")
        revision = target.get("revision")

        if not repo_id:
            raise RegistryError(
                "Scan record has no target.repo_id. A verdict we cannot attribute to "
                "a repo cannot be recorded; rescan with Hub metadata present."
            )
        if not revision:
            raise RegistryError(
                f"Scan record for {repo_id} has no target.revision. The registry keys "
                f"on (repo_id, revision) because branch refs move."
            )

        model = self.get(repo_id, revision)
        if model is None:
            model = self.register(repo_id, revision, actor, "auto-registered on scan ingest")

        blob = json.dumps(record, sort_keys=True, default=str)
        counts: dict[str, int] = {}
        for finding in record.get("findings", []):
            counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1

        self.conn.execute(
            "INSERT INTO scans (model_id, verdict, tier, reasons, finding_counts, "
            "record_sha256, record, scanned_at, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                model.id,
                record["verdict"],
                record["tier"],
                json.dumps(record.get("reasons", [])),
                json.dumps(counts),
                hashlib.sha256(blob.encode()).hexdigest(),
                blob,
                record.get("scanned_at", _now()),
                _now(),
            ),
        )
        self.conn.commit()

        # A scan moves the model to SCANNED and stamps the tier. From APPROVED
        # this is a rescan, which correctly drops it back for re-decision --
        # a model that changes verdict should not silently keep its approval.
        target_state = State.BLOCKED if record["tier"] == "blocked" else State.SCANNED
        self._transition(
            model,
            target_state,
            actor,
            f"scan verdict {record['verdict']} at tier {record['tier']}",
            tier=record["tier"],
        )
        return self._require(repo_id, revision)

    def _transition(self, model: ModelRow, to_state: State, actor: str,
                    reason: str, tier: str | None = None) -> None:
        allowed = TRANSITIONS[model.state]
        if to_state not in allowed and to_state is not model.state:
            allowed_list = ", ".join(sorted(s.value for s in allowed)) or "(none - terminal)"
            raise RegistryError(
                f"{model.repo_id}@{model.revision}: cannot move from "
                f"{model.state.value} to {to_state.value}. Allowed: {allowed_list}."
            )

        now = _now()
        self.conn.execute(
            "UPDATE models SET state = ?, updated_at = ?"
            + (", tier = ?" if tier else "")
            + " WHERE id = ?",
            ((to_state.value, now, tier, model.id) if tier
             else (to_state.value, now, model.id)),
        )
        self.conn.execute(
            "INSERT INTO transitions (model_id, from_state, to_state, actor, reason, at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (model.id, model.state.value, to_state.value, actor, reason, now),
        )
        self.conn.commit()

    def promote(self, repo_id: str, revision: str, to_state: State, actor: str,
                reason: str, justification: str | None = None) -> ModelRow:
        model = self._require(repo_id, revision)

        if to_state is State.APPROVED:
            scan = self.latest_scan(model.id)
            if scan is None:
                raise RegistryError(
                    f"{repo_id}@{revision} has no scan on record. Approving an "
                    f"unscanned model defeats the point of the gate."
                )
            if scan["tier"] == "blocked":
                raise RegistryError(
                    f"{repo_id}@{revision} is blocked: "
                    f"{'; '.join(json.loads(scan['reasons']))}. Blocked is terminal. "
                    f"If the publisher ships a fixed revision, register that revision."
                )
            if scan["tier"] not in AUTO_APPROVABLE_TIERS and not justification:
                raise RegistryError(
                    f"{repo_id}@{revision} scanned as tier '{scan['tier']}', which "
                    f"requires --justification recording why the risk is accepted "
                    f"and what compensating controls apply."
                )
            if justification:
                reason = f"{reason} | justification: {justification}"

        self._transition(model, to_state, actor, reason)
        return self._require(repo_id, revision)
