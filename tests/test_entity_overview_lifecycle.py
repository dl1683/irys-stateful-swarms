"""The original overview refresh lifecycle was replaced by pointer maintenance."""

from src.swarm import _prepare_entity_work
from src.swarm.blackboard import Blackboard
from src.swarm.models import Entry, EntrySource


def test_pending_initial_overview_never_defers_analysis():
    board = Blackboard(task_instruction="Assess identity", entries=[
        Entry(id="e1", content="Alex Rowan signed the application.",
              source=EntrySource("identity.txt", "s")),
        Entry(id="e2", content="Alex Rowan has a screening candidate.",
              source=EntrySource("identity.txt", "s")),
    ])
    board.entity_overview_state = {"pending": ["alex rowan"]}
    inventory = [{"entity_id": "alex rowan", "matched_card_ids": ["e1", "e2"]}]
    [task] = _prepare_entity_work(board, [{
        "description": "Compare identity evidence", "expected_output_type": "analysis",
        "reads_from_blackboard": ["e1", "e2"],
    }], 1, inventory)
    assert task["reads_from_blackboard"] == ["e1", "e2"]
