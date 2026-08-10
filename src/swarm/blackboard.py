from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .entity_maintenance_store import EntityMaintenanceState
from .models import (
    DocumentStatus, Entry, Signal, WorkerRecord,
    gen_entry_id, gen_signal_id,
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
    entity_maintenance_state: EntityMaintenanceState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.entity_maintenance_state = (
            EntityMaintenanceState.load(self.output_dir)
            if self.output_dir else EntityMaintenanceState()
        )

    def add_tokens_from_last_call(self, tokens: int) -> None:
        """Add tokens and grab model info from the last call_model invocation."""
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
            "entity_maintenance": {
                "processed_cards": len(self.entity_maintenance_state.processed_card_fingerprints),
                "profiles": len(self.entity_maintenance_state.profiles),
                "decisions": len(self.entity_maintenance_state.decisions),
            },
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
