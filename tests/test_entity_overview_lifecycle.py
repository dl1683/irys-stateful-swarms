import json

from src.swarm import _commit_entity_worker_outputs, _prepare_entity_work, _run_final_overview_pass
from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import attach_relevant_overviews, entry_state, overview_inventory
from src.swarm.models import Entry, EntrySource, ModelResult, WorkerOutput, WorkerRecord
from src.swarm.synthesis_packet import build_synthesis_packet, project_overview_statements


class FakeCaller:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def complete(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return ModelResult(next(self.responses), 10, 5, 15, "fake", 1)


def _board(tmp_path, budget=100_000):
    first = Entry(id="e1", content="Petrov's date of birth is 1975.",
                  source=EntrySource("kyc.pdf", "identity"))
    second = Entry(id="e2", content="Petrov is a distinct screening candidate.",
                   source=EntrySource("screening.pdf", "candidate"))
    old = Entry(id="old", type="entity_overview", content="Prior Petrov profile with supported identity details.")
    board = Blackboard(task_instruction="Report on Petrov", entries=[first, second, old],
                       iteration=5, token_budget=budget, output_dir=str(tmp_path))
    board.entity_overview_state = {
        "variants": {"petrov": ["Petrov"]}, "discovered_card_ids": ["e1", "e2", "e3"],
        "overviews": {"petrov": {
            "entry_id": "old", "input_ids": ["e1", "e2"],
            "source_states": {"e1": entry_state(first), "e2": entry_state(second)},
            "structured": {"entity_profiles": [{"label": "Petrov", "statements": [{
                "text": "Petrov's date of birth is 1975.", "supports_entries": ["e1"], "kind": "fact",
            }]}]}, "last_success_iteration": 3,
        }},
    }
    return board


def _late_source_output():
    late = Entry(id="e3", content="Petrov is a Swiss permanent resident.",
                 source=EntrySource("kyc.pdf", "residency"),
                 created_by=WorkerRecord("reader", "read identity", 5))
    return WorkerOutput([late], 0, 0, 0, "", "reader", {
        "expected_output_type": "observation", "reads_from_blackboard": ["e1", "e2", "old"],
    }, [("kyc.pdf", "residency")])


def test_supervisor_attachment_then_final_late_source_reaches_labelled_packet(tmp_path):
    board = _board(tmp_path)
    board.save_snapshot("before_supervisor")
    inventory = overview_inventory(board.entity_overview_state, board.entries, board.iteration)
    [task] = _prepare_entity_work(board, [{"expected_output_type": "observation",
        "reads_from_blackboard": ["e1", "e2"]}], board.iteration, inventory)
    assert "old" in task["reads_from_blackboard"]

    _commit_entity_worker_outputs(board, [_late_source_output()], FakeCaller([]), board.iteration)
    board.save_snapshot("after_supervisor")
    assert board.entity_overview_state["recipients"][0]["overview_ids"] == ["old"]
    assert overview_inventory(board.entity_overview_state, board.entries, 5)[0]["suggestion"] == ""

    response = {"findings": [{"type": "entity_overview", "structured_overview": {
        "entity_profiles": [
            {"label": "Petrov (subject)", "statements": [
                {"text": "Petrov is a Swiss permanent resident.",
                 "supports_entries": ["e3"], "kind": "fact"}]},
            {"label": "Petrov (screening candidate)", "statements": [
                {"text": "The screening candidate remains distinct.",
                 "supports_entries": ["e2"], "kind": "unresolved"}]},
        ], "relationships_and_distinctions": [], "unresolved_or_conflicting_evidence": [],
    }}]}
    caller = FakeCaller([json.dumps(response)])
    _run_final_overview_pass(board, caller)
    packet = build_synthesis_packet(project_overview_statements(board), board)
    (tmp_path / "packet.json").write_text(json.dumps(packet, indent=2))

    assert board.entity_overview_state["final_attempted"] == ["petrov"]
    assert board.entity_overview_state["overviews"]["petrov"]["entry_id"] != "old"
    assert any(row["overview_group_label"] == "Petrov (subject)"
               and row["entry_ids"] == ["e3"] for row in packet)
    assert any(row["overview_group_label"] == "Petrov (screening candidate)"
               and row["entry_ids"] == ["e2"] for row in packet)
    board.save_snapshot("final_check")
    report = json.loads((tmp_path / "swarm" / "entity_overview_report.json").read_text())
    assert report["final_attempted"] == ["petrov"]


def test_failed_final_refresh_keeps_old_state_and_exposes_backlog(tmp_path):
    board = _board(tmp_path)
    _commit_entity_worker_outputs(board, [_late_source_output()], FakeCaller([]), board.iteration)
    _run_final_overview_pass(board, FakeCaller([json.dumps({"findings": []})]))

    assert board.entity_overview_state["overviews"]["petrov"]["entry_id"] == "old"
    assert board.entity_overview_state["failed"] == ["petrov"]
    assert board.entity_overview_state["freshness_failures"]
    board.save_snapshot("failed")
    report = json.loads((tmp_path / "swarm" / "entity_overview_report.json").read_text())
    assert report["eligible_refresh_backlog"] == []
    assert report["failed"] == ["petrov"]


def test_final_pass_respects_budget_and_inactive_support_is_not_projected(tmp_path):
    board = _board(tmp_path, budget=100)
    board.entity_overview_state["overviews"]["petrov"]["structured"]["entity_profiles"][0]["statements"][0]["supports_entries"] = ["e1", "e2"]
    board.entries[0].status = "inactive"
    board.entries.append(Entry(id="e3", content="Petrov is a Swiss permanent resident.",
                               source=EntrySource("kyc.pdf", "residency")))
    caller = FakeCaller([])
    [worker] = attach_relevant_overviews([{"expected_output_type": "analysis",
        "reads_from_blackboard": ["old", "e1", "e2"]}], board.entries,
        board.entity_overview_state)
    assert "old" not in worker["reads_from_blackboard"]
    _run_final_overview_pass(board, caller)

    assert caller.prompts == []
    assert board.entity_overview_state["budget_limited"] == ["petrov"]
    assert board.entity_overview_state["overviews"]["petrov"]["entry_id"] == "old"
    assert project_overview_statements(board) == []
