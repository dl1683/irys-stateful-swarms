"""Provider-free first checkpoint for pointer overviews."""

import json

import pytest

from src.swarm import _prepare_entity_work, _run_initial_overviews
from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import validate_pointer_cards
from src.swarm.models import DocumentStatus, Entry, EntrySource, ModelResult
from src.swarm.synthesis import _selected_evidence_text
from src.swarm.synthesis_packet import build_synthesis_packet


class Caller:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def complete(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return ModelResult(json.dumps(self.response), 20, 10, 30, "fixture", 1)


def test_initial_identity_cards_attach_and_writers_follow_originals(tmp_path):
    entries = [
        Entry(id=f"src{i}", content=text, source=EntrySource("identity.txt", "s"))
        for i, text in enumerate([
            "Alex Rowan was born 1 Jan 1970.",
            "Alex Rowan lives at 10 North Street.",
            "Alex Rowan signed the applicant form.",
            "Alex Rowan was born 2 Feb 1980.",
            "Alex Rowan lives at 20 South Street.",
            "Alex Rowan signed the screening response.",
            "Alex Rowan appears on an ambiguous search hit.",
        ], 1)
    ]
    board = Blackboard(task_instruction="Report on the applicant", entries=entries,
                       documents=[DocumentStatus(name="identity.txt")],
                       output_dir=str(tmp_path))
    board.entity_overview_state = {"variants": {"alex rowan": ["Alex Rowan"]}}
    ordinary = Caller({})
    smart = Caller({"cards": [
        {"name": "Alex Rowan (applicant)", "identity_source_ids": ["src1", "src2", "src3"],
         "candidate_source_ids": ["src7"], "facts": [
             {"text": "Born 1 Jan 1970", "source_ids": ["src1"]},
             {"text": "Lives at 10 North Street", "source_ids": ["src2"]}],
         "identity_clues": [{"type": "birth_date", "value": "1 Jan 1970", "source_ids": ["src1"]}]},
        {"name": "Alex Rowan (screening candidate)",
         "identity_source_ids": ["src4", "src5", "src6"],
         "candidate_source_ids": ["src7"], "facts": [
             {"text": "Born 2 Feb 1980", "source_ids": ["src4"]},
             {"text": "Lives at 20 South Street", "source_ids": ["src5"]}],
         "identity_clues": [{"type": "birth_date", "value": "2 Feb 1980", "source_ids": ["src4"]}]},
    ]})
    _run_initial_overviews(board, ordinary, smart)
    assert not ordinary.prompts
    assert len(smart.prompts) == 1
    assert board.entity_overview_state.get("document_rounds", 0) == 0
    cards = board.entity_overview_state["overviews"]["alex rowan"]["cards"]
    assert len(cards) == 2
    assert all("src7" not in card["identity_source_ids"] for card in cards)
    assert "20 South" not in next(e.content for e in board.entries if e.id == cards[0]["id"])
    assert all(fact["source_ids"] for card in cards for fact in card["facts"])

    work = [{"description": "Compare the applicant", "expected_output_type": "analysis",
             "reads_from_blackboard": ["src1", "src2"]},
            {"description": "Legacy request", "expected_output_type": "entity_overview",
             "entity_overview_id": "alex rowan"}]
    [task] = _prepare_entity_work(board, work, 1, [])
    assert cards[0]["id"] in task["reads_from_blackboard"]
    assert cards[1]["id"] not in task["reads_from_blackboard"]
    assert board.entity_overview_state["rejected_requests"]

    selected = [{"entry_ids": ["src1"], "summary": "Applicant birth date"}]
    evidence = _selected_evidence_text(selected, board.entries, 10000, include_remaining=False)
    assert "src2" in evidence and "10 North Street" in evidence
    assert "20 South Street" not in evidence
    packet = build_synthesis_packet(selected, board)
    assert len(packet) == 1 and packet[0]["entry_ids"] == ["src1"]
    assert "final_attempted" not in board.entity_overview_state


def test_pointer_card_rejects_candidate_fact_and_unknown_reference():
    base = {"name": "Alex Rowan", "identity_source_ids": ["src1"],
            "candidate_source_ids": ["src2"], "facts": [], "identity_clues": []}
    with pytest.raises(ValueError):
        validate_pointer_cards([dict(base, facts=[{"text": "Candidate date", "source_ids": ["src2"]}])],
                               {"src1", "src2"})
    with pytest.raises(ValueError):
        validate_pointer_cards([dict(base, identity_source_ids=["invented"])], {"src1", "src2"})


def test_run_swarm_smoke_saves_cards_and_original_writer_evidence(tmp_path, monkeypatch):
    import re
    import src.swarm as swarm
    from src.swarm.models import Document, Task

    class RoutingCaller:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, **_kwargs):
            self.prompts.append(prompt)
            if prompt.startswith("Examine this document's structure"):
                value = {"numbered_items": 0}
            elif prompt.startswith("Read this section"):
                value = {"findings": [{"type": "observation", "content": text, "confidence": 0.9}
                    for text in [
                        "Alex Rowan applicant born 1 Jan 1970.",
                        "Alex Rowan applicant lives at 10 North Street.",
                        "Alex Rowan applicant signed the form.",
                        "Alex Rowan screening person born 2 Feb 1980.",
                        "Alex Rowan screening person lives at 20 South Street.",
                        "Alex Rowan screening person signed the response.",
                    ]]}
            elif prompt.startswith("List every organization and person name"):
                value = {"names": ["Alex Rowan"]}
            elif prompt.startswith("You are the analytical orchestrator"):
                value = {"workers": []}
            else:
                raise AssertionError(prompt[:100])
            return ModelResult(json.dumps(value), 10, 5, 15, "fixture-ordinary", 1)

    class SmartCaller:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, **_kwargs):
            self.prompts.append(prompt)
            ids = re.findall(r"\[(e\d+)\] match=", prompt)
            assert len(ids) == 6
            value = {"cards": [
                {"name": "Alex Rowan (applicant)", "identity_source_ids": ids[:3],
                 "candidate_source_ids": [], "facts": [
                     {"text": "Born 1 Jan 1970", "source_ids": [ids[0]]}],
                 "identity_clues": []},
                {"name": "Alex Rowan (screening person)", "identity_source_ids": ids[3:],
                 "candidate_source_ids": [], "facts": [
                     {"text": "Born 2 Feb 1980", "source_ids": [ids[3]]}],
                 "identity_clues": []},
            ]}
            return ModelResult(json.dumps(value), 10, 5, 15, "fixture-smart", 1)

    ordinary, smart = RoutingCaller(), SmartCaller()
    writer = {}

    def curate(board, _caller):
        first = next(e for e in board.entries if e.type == "observation")
        return [{"entry_id": first.id, "summary": first.content, "section": "Identity"}], 0

    def synthesize(board, packet, _caller):
        writer["packet"] = packet
        writer["evidence"] = _selected_evidence_text(packet, board.entries, 10000,
                                                      include_remaining=False)
        return "Applicant identity report", 0

    monkeypatch.setattr(swarm, "curate_entries", curate)
    monkeypatch.setattr(swarm, "synthesize_deliverable", synthesize)
    text = "\n".join(["Alex Rowan identity matter."] * 5)
    result, board = swarm.run_swarm(
        Task("Report on applicant", [Document("d1", "identity.txt", text)], output_dir=str(tmp_path)),
        ordinary, smart_caller=smart, max_iterations=1, min_iterations=1,
    )
    assert result == "Applicant identity report"
    assert len(smart.prompts) == 1
    assert len(board.entity_overview_state["overviews"]["alex rowan"]["cards"]) == 2
    assert len(writer["packet"]) == 1
    assert "born 1 Jan 1970" in writer["evidence"]
    assert "2 Feb 1980" not in writer["evidence"]
    (tmp_path / "checkpoint_excerpt.json").write_text(json.dumps({
        "cards": board.entity_overview_state["overviews"]["alex rowan"]["cards"],
        "writer_packet": writer["packet"], "writer_evidence": writer["evidence"],
        "smart_calls": len(smart.prompts), "ordinary_calls": len(ordinary.prompts),
    }, indent=2))
