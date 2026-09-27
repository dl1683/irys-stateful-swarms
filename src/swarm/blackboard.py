from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .models import (
    DocumentStatus, Entry, EntrySource, EpistemicStatus, Signal, WorkerRecord,
    advance_id_counters, gen_entry_id, gen_signal_id, id_counters,
)


def _priority_rank(p: str) -> int:
    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(p, 1)


def signals_similar(a: Signal, b: Signal) -> bool:
    if a.type != b.type:
        return False

    def trigrams(text):
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        if len(words) < 3:
            return {text.lower().strip()}
        return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}

    tri_a, tri_b = trigrams(a.content), trigrams(b.content)
    if not tri_a or not tri_b:
        return a.content.strip().lower() == b.content.strip().lower()
    return len(tri_a & tri_b) / len(tri_a | tri_b) >= 0.5


@dataclass
class Blackboard:
    task_instruction: str = ""
    documents: list[DocumentStatus] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    iteration: int = 0
    total_tokens_used: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    cost_by_model: dict = field(default_factory=dict)
    token_budget: int = 3_000_000
    started_at: str = ""
    output_dir: str = ""
    entity_overview_state: dict = field(default_factory=dict)

    @staticmethod
    def source_fingerprints(documents: list[DocumentStatus]) -> list[dict]:
        """Hash original task files, preserving order even when document IDs repeat."""
        sources = []
        for doc in documents:
            if doc.source_path:
                with open(doc.source_path, "rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
            else:
                digest = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
            sources.append({"id": doc.id, "name": doc.name,
                            "source_path": doc.source_path, "sha256": digest})
        return sources

    def save_checkpoint(self, seed_plan: dict, entries_per_iteration: list[int],
                        *, loop_ended: bool) -> Path | None:
        """Persist the complete durable state after an iteration has finished."""
        if not self.output_dir:
            return None
        if self.iteration < 1:
            raise ValueError("checkpoint needs a completed iteration")
        directory = Path(self.output_dir) / "swarm" / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        sources = self.source_fingerprints(self.documents)
        board = {
            key: getattr(self, key) for key in (
                "task_instruction", "iteration", "total_tokens_used", "tokens_input",
                "tokens_output", "cost_by_model", "token_budget", "started_at",
                "output_dir", "entity_overview_state",
            )
        }
        board["documents"] = [{
            key: getattr(doc, key) for key in (
                "id", "name", "size_bytes", "source_path", "headings",
                "structural_profile", "read_status", "sections_read", "sections_unread",
            )
        } | {"was_loaded": doc.is_loaded} for doc in self.documents]
        board["entries"] = [asdict(entry) for entry in self.entries]
        board["signals"] = [asdict(signal) for signal in self.signals]
        counters = id_counters()
        for kind, items in (("entry", self.entries), ("signal", self.signals)):
            prefix = "e" if kind == "entry" else "s"
            counters[kind] = max(counters[kind], *(
                int(item.id[1:]) for item in items
                if item.id.startswith(prefix) and item.id[1:].isdigit()
            ), 0)
        payload = {
            "schema_version": 1, "completed_iteration": self.iteration,
            "loop_ended": loop_ended, "seed_plan": seed_plan,
            "entries_per_iteration": entries_per_iteration[-2:],
            "source_fingerprints": sources, "id_counters": counters,
            "blackboard": board,
        }
        path = directory / f"checkpoint_iter_{self.iteration}.json"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                             prefix=".checkpoint_", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load_checkpoint(cls, path: Path) -> tuple[Blackboard, dict, list[int], bool, dict]:
        """Read a complete v1 checkpoint; document text is reattached by the runner."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            required = {"schema_version", "completed_iteration", "loop_ended",
                        "seed_plan", "entries_per_iteration", "source_fingerprints",
                        "id_counters", "blackboard"}
            if not isinstance(data, dict) or not required <= data.keys():
                raise ValueError("incomplete checkpoint")
            if type(data["schema_version"]) is not int or data["schema_version"] != 1:
                raise ValueError("unsupported checkpoint version")
            board_data = data["blackboard"]
            board_fields = {item.name for item in fields(cls)}
            if not isinstance(board_data, dict) or not board_fields <= board_data.keys():
                raise ValueError("incomplete blackboard state")
            if (type(data["completed_iteration"]) is not int
                    or data["completed_iteration"] < 1
                    or board_data["iteration"] != data["completed_iteration"]
                    or type(data["loop_ended"]) is not bool
                    or not isinstance(data["seed_plan"], dict)
                    or not isinstance(data["entries_per_iteration"], list)
                    or not isinstance(data["source_fingerprints"], list)):
                raise ValueError("invalid checkpoint metadata")
            if (len(data["entries_per_iteration"]) > 2
                    or any(type(count) is not int or count < 0
                           for count in data["entries_per_iteration"])
                    or any(type(board_data[key]) is not int or board_data[key] < 0
                           for key in ("total_tokens_used", "tokens_input",
                                       "tokens_output", "token_budget"))
                    or not isinstance(board_data["cost_by_model"], dict)
                    or any(not isinstance(usage, dict)
                           or any(type(usage.get(key)) is not int or usage[key] < 0
                                  for key in ("input", "output", "total", "calls"))
                           for usage in board_data["cost_by_model"].values())):
                raise ValueError("invalid checkpoint metering")
            counters = data["id_counters"]
            if (not isinstance(counters, dict)
                    or any(type(counters.get(key)) is not int or counters[key] < 0
                           for key in ("entry", "signal"))):
                raise ValueError("invalid checkpoint ID counters")
            if (not isinstance(board_data["documents"], list)
                    or not isinstance(board_data["entries"], list)
                    or not isinstance(board_data["signals"], list)):
                raise ValueError("invalid checkpoint collections")
            document_fields = {"id", "name", "size_bytes", "source_path", "headings",
                               "structural_profile", "read_status", "sections_read",
                               "sections_unread", "was_loaded"}
            entry_fields = {item.name for item in fields(Entry)}
            signal_fields = {item.name for item in fields(Signal)}
            if (any(not isinstance(doc, dict) or not document_fields <= doc.keys()
                    or type(doc["was_loaded"]) is not bool
                    for doc in board_data["documents"])
                    or any(not isinstance(entry, dict)
                           or not entry_fields <= entry.keys()
                           for entry in board_data["entries"])
                    or any(not isinstance(signal, dict)
                           or not signal_fields <= signal.keys()
                           for signal in board_data["signals"])):
                raise ValueError("incomplete checkpoint records")
            board = cls(**{key: board_data[key] for key in board_fields
                           if key not in ("documents", "entries", "signals")})
            board.documents = []
            for doc in board_data["documents"]:
                restored = DocumentStatus(**{key: value for key, value in doc.items()
                                             if key != "was_loaded"})
                restored._checkpoint_was_loaded = doc["was_loaded"]
                board.documents.append(restored)
            board.entries = [Entry(
                **{**entry,
                   "source": EntrySource(**entry["source"]) if entry["source"] else None,
                   "epistemic": EpistemicStatus(**entry["epistemic"]) if entry["epistemic"] else None,
                   "created_by": WorkerRecord(**entry["created_by"])}
            ) for entry in board_data["entries"]]
            board.signals = [Signal(**signal) for signal in board_data["signals"]]
            board._entry_index = {entry.id: entry for entry in board.entries if entry.id}
            advance_id_counters(counters["entry"], counters["signal"])
            return (board, data["seed_plan"], data["entries_per_iteration"],
                    data["loop_ended"], data["source_fingerprints"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"invalid checkpoint: {error}") from error

    def add_tokens_from_last_call(self, tokens: int) -> None:
        """Add tokens and grab model info from the last call_model invocation."""
        if tokens <= 0:
            return
        from .worker_dispatch import get_last_call_usage
        by_model, model, t_in, t_out = get_last_call_usage()
        if isinstance(by_model, dict) and by_model:
            self.total_tokens_used += tokens
            self.tokens_input += sum(v.get("input", 0) for v in by_model.values())
            self.tokens_output += sum(v.get("output", 0) for v in by_model.values())
            for model, usage in by_model.items():
                if model not in self.cost_by_model:
                    self.cost_by_model[model] = {"input": 0, "output": 0, "total": 0, "calls": 0}
                self.cost_by_model[model]["input"] += usage.get("input", 0)
                self.cost_by_model[model]["output"] += usage.get("output", 0)
                self.cost_by_model[model]["total"] += usage.get("total", 0)
                self.cost_by_model[model]["calls"] += usage.get("calls", 0)
            return
        self.add_tokens(tokens, t_in, t_out, model)

    def add_tokens(self, tokens: int, tokens_in: int = 0, tokens_out: int = 0,
                   model: str = "") -> None:
        self.total_tokens_used += tokens
        self.tokens_input += tokens_in
        self.tokens_output += tokens_out
        if model:
            if model not in self.cost_by_model:
                self.cost_by_model[model] = {"input": 0, "output": 0, "total": 0, "calls": 0}
            self.cost_by_model[model]["input"] += tokens_in
            self.cost_by_model[model]["output"] += tokens_out
            self.cost_by_model[model]["total"] += tokens
            self.cost_by_model[model]["calls"] += 1

    def budget_used_pct(self) -> float:
        return round(self.total_tokens_used / max(self.token_budget, 1) * 100, 1)

    def _index_entry(self, entry: Entry) -> None:
        if not hasattr(self, '_entry_index'):
            self._entry_index: dict[str, Entry] = {}
        if entry.id:
            self._entry_index[entry.id] = entry

    def _ensure_unique_id(self, entry: Entry) -> None:
        if not hasattr(self, '_entry_index'):
            self._entry_index = {}
        if entry.id and entry.id in self._entry_index:
            from .models import gen_entry_id
            for _ in range(100):
                entry.id = gen_entry_id()
                if entry.id not in self._entry_index:
                    break

    def add_entry(self, entry: Entry) -> None:
        self._ensure_unique_id(entry)
        self.entries.append(entry)
        self._index_entry(entry)
        self._extract_signals(entry)
        self._propagate_effects(entry)

    def add_entries_batch(self, entries: list[Entry]) -> None:
        for e in entries:
            self._ensure_unique_id(e)
            self.entries.append(e)
            self._index_entry(e)
        for e in entries:
            self._extract_signals(e)
        for e in entries:
            self._propagate_effects(e)

    def find_entry(self, entry_id: str) -> Entry | None:
        if hasattr(self, '_entry_index'):
            found = self._entry_index.get(entry_id)
            if found is not None:
                return found
        for e in self.entries:
            if e.id == entry_id:
                self._index_entry(e)
                return e
        return None

    def get_entries_by_ids(self, ids: list[str]) -> list[Entry]:
        id_set = set(ids)
        return [e for e in self.entries if e.id in id_set and e.status == "active"]

    def get_summary(self) -> dict:
        active = [e for e in self.entries if e.status == "active"]
        type_counts: dict[str, int] = {}
        for e in active:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1
        open_sigs = [s for s in self.signals if s.status == "open"]
        return {
            "iteration": self.iteration,
            "entry_counts": type_counts,
            "total_active_entries": len(active),
            "open_signals": open_sigs,
            "critical_signals": [s for s in open_sigs if s.priority == "critical"],
            "high_signals": [s for s in open_sigs if s.priority == "high"],
            "documents": [d.to_dict() for d in self.documents],
            "budget_used_pct": self.budget_used_pct(),
            "entries_this_iteration": [
                e for e in active if e.created_by.iteration == self.iteration
            ],
            "disputed_entries": [
                e for e in self.entries
                if e.status == "disputed"
                or (e.status == "active" and e.confidence < 0.4)
            ],
        }

    def add_signal(self, signal: Signal) -> None:
        for existing in self.signals:
            if existing.status == "open" and signals_similar(existing, signal):
                if _priority_rank(signal.priority) > _priority_rank(existing.priority):
                    existing.priority = signal.priority
                return
        self.signals.append(signal)

    def expire_old_signals(self, expiry_iterations: int = 3) -> None:
        for s in self.signals:
            if (s.status == "open"
                    and s.priority in ("medium", "low")
                    and self.iteration - s.iteration_created >= expiry_iterations):
                s.status = "expired"

    def _extract_signals(self, entry: Entry) -> None:
        for q in entry.opens_questions[:5]:
            if isinstance(q, str) and q.strip():
                self.add_signal(Signal(
                    id=gen_signal_id(), type="question", content=q.strip(),
                    origin_entry=entry.id, priority="medium",
                    status="open", iteration_created=self.iteration,
                ))
        for sig_id in entry.addresses_signals:
            for s in self.signals:
                if s.id == sig_id and s.status == "open":
                    if s.type == "artifact_requirement" and entry.type not in (
                        "analysis", "calculation", "strategy",
                    ):
                        continue
                    s.status = "addressed"
                    s.addressed_by = entry.id

    def _propagate_effects(self, entry: Entry) -> None:
        for sid in entry.supports_entries:
            target = self.find_entry(sid)
            if not target:
                continue
            same_src = sum(
                1 for e in self.entries
                if sid in e.supports_entries
                and e.source and target.source
                and e.source.document == target.source.document
            )
            boost = 0.02 if same_src > 2 else 0.05
            target.confidence = min(0.98, target.confidence + boost)

        # Stage new contradiction entries to avoid mutating self.entries during iteration
        staged_entries: list[Entry] = []
        contradiction_penalized = False
        for cid in entry.contradicts_entries:
            target = self.find_entry(cid)
            if not target:
                continue
            target.confidence = max(0.1, target.confidence - 0.12)
            if not contradiction_penalized:
                entry.confidence = max(0.1, entry.confidence - 0.12)
                contradiction_penalized = True
            target.status = "disputed"
            entry.status = "disputed"
            staged_entries.append(Entry(
                id=gen_entry_id(), type="contradiction",
                content=(
                    f"CONFLICT: [{entry.id}] {entry.content[:150]}... "
                    f"vs [{cid}] {target.content[:150]}..."
                ),
                created_by=WorkerRecord("system", "contradiction_detection", self.iteration),
                confidence=1.0, status="active",
            ))
            self.add_signal(Signal(
                id=gen_signal_id(), type="contradiction_resolution",
                content=f"Resolve conflict between {entry.id} and {cid}",
                origin_entry=entry.id, priority="critical",
                status="open", iteration_created=self.iteration,
            ))
            # Snapshot entries to avoid iterating over staged additions
            for other in list(self.entries):
                if cid in other.supports_entries:
                    other.confidence = max(0.1, other.confidence - 0.05)

        for staged in staged_entries:
            self.entries.append(staged)

        for sid in entry.supersedes_entries:
            target = self.find_entry(sid)
            if not target:
                continue
            target.status = "superseded"
            for s in self.signals:
                if s.addressed_by == sid:
                    s.status = "open"
                    s.addressed_by = None
                    s.iteration_created = self.iteration

    def save_snapshot(self, label: str = "") -> None:
        if not self.output_dir:
            return
        snapshot_dir = Path(self.output_dir) / "swarm"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{label}" if label else ""
        path = snapshot_dir / f"blackboard_iter_{self.iteration}{suffix}.json"
        data = {
            "task_instruction": self.task_instruction,
            "documents": [d.to_dict() for d in self.documents],
            "entries": [e.to_dict() for e in self.entries],
            "signals": [s.to_dict() for s in self.signals],
            "iteration": self.iteration,
            "total_tokens_used": self.total_tokens_used,
            "token_budget": self.token_budget,
            "entity_overview_state": self.entity_overview_state,
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        # Keep a stable, small summary beside snapshots so capped runs are inspectable.
        entity_state = self.entity_overview_state
        if entity_state:
            from .entity_overview import overview_inventory
            inventory = overview_inventory(entity_state, self.entries, self.iteration)
            variants = entity_state.get("variants", {})
            overviews = entity_state.get("overviews", {})
            report = {
                "catalogue_groups": len(variants),
                "catalogue_variants": sum(len(names) for names in variants.values()),
                "processed_card_count": len(entity_state.get("discovered_card_ids", [])),
                "overview_count": len({card["id"] for record in overviews.values()
                                       for card in record.get("cards", [])}) or len(overviews),
                "overviews": overviews,
                "usage": entity_state.get("usage", {}),
                "jobs": entity_state.get("jobs", []),
                "document_rounds": entity_state.get("document_rounds", 0),
                "delta_backlog": entity_state.get("delta_backlog", []),
                "delta_errors": entity_state.get("delta_errors", []),
                "reference_errors": entity_state.get("reference_errors", []),
                "overview_review_requests": entity_state.get("overview_review_requests", []),
                "identity_review_requests": entity_state.get("identity_review_requests", []),
                "identity_request_triage": entity_state.get("identity_request_triage", []),
                "identity_reviews": entity_state.get("identity_reviews", {}),
                "identity_review_backlog": entity_state.get("identity_review_backlog", []),
                "identity_history": entity_state.get("identity_history", []),
                "caller_fallbacks": entity_state.get("caller_fallbacks", []),
                "recipients": entity_state.get("recipients", []),
                "recipient_count": len(entity_state.get("recipients", [])),
                "referencing_recipient_count": sum(
                    bool(item.get("referenced_overview_ids"))
                    for item in entity_state.get("recipients", [])
                ),
                "failed": entity_state.get("failed", []),
                "initial_backlog": entity_state.get("initial_queue", []),
                "eligible_initial_backlog": [item["entity_id"] for item in inventory
                                             if item["eligibility"] == "initial"],
                "eligible_refresh_backlog": [item["entity_id"] for item in inventory
                                             if item["eligibility"] == "refresh"],
                "scheduled": entity_state.get("pending", []),
                "budget_limited": entity_state.get("budget_limited", []),
                "freshness_failures": entity_state.get("freshness_failures", []),
                "final_candidates": entity_state.get("final_candidates", []),
                "final_attempted": entity_state.get("final_attempted", []),
                "selection_log": entity_state.get("selection_log", []),
                "deferred": entity_state.get("deferred", []),
                "rejected_requests": entity_state.get("rejected_requests", []),
                "discovery_failures": entity_state.get("discovery_failures", []),
            }
            (snapshot_dir / "entity_overview_report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8",
            )
