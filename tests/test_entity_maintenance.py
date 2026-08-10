import json

from src.swarm.blackboard import Blackboard
from src.swarm.entity_maintenance import (
    maintenance_is_due,
    project_confirmed_decisions,
)
from src.swarm.entity_maintenance_store import (
    EntityMaintenanceConfig,
    EntityMaintenanceState,
)


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
