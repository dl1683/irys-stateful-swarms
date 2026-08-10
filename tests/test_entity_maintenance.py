import json

from src import swarm
from src.swarm.blackboard import Blackboard
from src.swarm.entity_maintenance import (
    maintenance_is_due,
    project_confirmed_decisions,
)
from src.swarm.entity_maintenance_store import (
    EntityMaintenanceConfig,
    EntityMaintenanceState,
)
from src.swarm.models import Entry, EntrySource, Task, WorkerOutput


class FakeCaller:
    pass


def task_with_direct_cards(metadata: dict[str, object] | None = None) -> Task:
    return Task(
        instruction="Resolve entity records.",
        documents=[],
        metadata=metadata or {},
    )


def patch_minimal_swarm_for_iterations(monkeypatch, *, worker_iterations: int) -> None:
    def worker_outputs(tasks, blackboard, caller):
        entry = Entry(
            id=f"direct-card-{blackboard.iteration}",
            content="Northwind Ltd is named in the task record.",
            source=EntrySource(document="task_instruction", evidence="task record"),
            direct_document_context=True,
        )
        return [WorkerOutput(
            entries=[entry], tokens_used=0, tokens_input=0, tokens_output=0,
            model="test", worker_id="test-worker", task={}, sections_read=[],
        )]

    monkeypatch.setattr(swarm, "_execute_initial_reading", lambda *args: ([], 0))
    monkeypatch.setattr(swarm, "run_orchestrator", lambda *args, **kwargs: (
        {"workers": [{}]} if args[0].iteration <= worker_iterations else {"workers": []}, 0,
    ))
    monkeypatch.setattr(swarm, "execute_workers_parallel", worker_outputs)
    monkeypatch.setattr(swarm, "passes_quality_gate", lambda entry: True)
    monkeypatch.setattr(swarm, "curate_entries", lambda *args: ([], 0))
    monkeypatch.setattr(swarm, "synthesize_deliverable", lambda *args: ("answer", 0))
    monkeypatch.setattr(swarm, "source_claim_verification_enabled", lambda: False)


def test_swarm_runs_maintenance_after_iteration_three_not_each_worker_batch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        swarm,
        "run_entity_maintenance",
        lambda bb, caller, config, *, trigger: calls.append((bb.iteration, trigger)),
        raising=False,
    )
    patch_minimal_swarm_for_iterations(monkeypatch, worker_iterations=4)

    swarm.run_swarm(task_with_direct_cards(), FakeCaller(), max_iterations=4, min_iterations=4)

    assert calls == [(3, "periodic"), (4, "final")]


def test_metadata_changes_interval_without_changing_orchestrator(monkeypatch):
    calls = []
    monkeypatch.setattr(
        swarm,
        "run_entity_maintenance",
        lambda bb, caller, config, *, trigger: calls.append(
            (bb.iteration, config.entity_resolution_interval_iterations, trigger)
        ),
        raising=False,
    )
    patch_minimal_swarm_for_iterations(monkeypatch, worker_iterations=4)

    swarm.run_swarm(
        task_with_direct_cards(metadata={"entity_resolution_interval_iterations": 2}),
        FakeCaller(),
        max_iterations=4,
        min_iterations=4,
    )

    assert calls == [(2, 2, "periodic"), (4, 2, "periodic"), (4, 2, "final")]


def _decision(outcome: str) -> dict[str, object]:
    return {
        "decision_id": f"decision-{outcome}",
        "semantic_key": f"company:{outcome}",
        "outcome": outcome,
        "evidence_fingerprint": f"candidate-fingerprint-{outcome}",
        "profile_ids": ["company:northwind-ltd", "company:northwind-limited"],
        "source_card_ids": ["e1", "e2"],
        "rationale": "Source-grounded outcome.",
        "conflicts": [],
    }


def same_entity_decision() -> dict[str, object]:
    return _decision("same_entity")


def same_name_distinct_entity_decision() -> dict[str, object]:
    return _decision("same_name_distinct_entity")


def uncertain_decision() -> dict[str, object]:
    return _decision("uncertain")


def blackboard_and_state_with_decisions(
    *decisions: dict[str, object],
) -> tuple[Blackboard, EntityMaintenanceState]:
    return Blackboard(iteration=3), EntityMaintenanceState(
        decisions={str(decision["decision_id"]): decision for decision in decisions}
    )


def test_only_confirmed_outcomes_create_blackboard_resolution_cards():
    blackboard, state = blackboard_and_state_with_decisions(
        same_entity_decision(), same_name_distinct_entity_decision(), uncertain_decision(),
    )

    project_confirmed_decisions(blackboard, state)

    cards = [entry for entry in blackboard.entries if entry.type == "duplicate_name_resolution"]
    assert {json.loads(card.content)["outcome"] for card in cards} == {
        "same_entity", "same_name_distinct_entity",
    }
    assert all("uncertain" not in card.content for card in cards)


def test_due_gate_runs_every_three_iterations_and_always_final():
    config = EntityMaintenanceConfig()

    assert [maintenance_is_due(i, config) for i in range(1, 7)] == [
        False, False, True, False, False, True,
    ]
    assert maintenance_is_due(1, config, final=True) is True
