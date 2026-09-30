import json

import pytest

from src import swarm
from src.swarm.blackboard import Blackboard
from src.swarm.models import (
    Document, Entry, EntrySource, Signal, Task, WorkerRecord, gen_entry_id,
)
from src.swarm.worker_dispatch import set_last_call_usage


class NoProvider:
    def complete(self, *_args, **_kwargs):
        raise AssertionError("provider called during offline resume test")


def _checkpoint(tmp_path, *, ended=False):
    source = tmp_path / "source.txt"
    source.write_text("Evidence in source.", encoding="utf-8")
    task = Task(
        instruction="Review source", output_dir=str(tmp_path),
        documents=[Document("d1", "source.txt", source.read_text(),
                            metadata={"path": str(source)})],
    )
    board = Blackboard(
        task_instruction=task.instruction, output_dir=task.output_dir, iteration=1,
        documents=swarm._build_doc_statuses(task.documents),
        entries=[Entry(id="e20", content="Evidence", type="observation",
                       source=EntrySource("source.txt", evidence="Evidence"),
                       created_by=WorkerRecord("worker", "read", 1))],
        signals=[Signal(id="s12", content="Open question", iteration_created=1)],
        total_tokens_used=20, tokens_input=13, tokens_output=7,
        cost_by_model={"prior": {"input": 13, "output": 7, "total": 20, "calls": 2}},
        entity_overview_state={"usage": {"overview_calls": 1}},
    )
    board.documents[0].read_status = "fully_read"
    path = board.save_checkpoint({"key_questions": ["Open question"]}, [1],
                                 loop_ended=ended)
    return task, source, path


def test_resume_reaches_next_iteration_with_saved_state(tmp_path, monkeypatch):
    task, source, path = _checkpoint(tmp_path)
    observed = {}

    def inspect_orchestration(board, _caller, **_kwargs):
        observed["iteration"] = board.iteration
        observed["entry"] = board.entries[0].id
        observed["read_status"] = board.documents[0].read_status
        observed["text"] = board.documents[0].text
        observed["signals"] = [signal.id for signal in board.signals]
        observed["tokens"] = (board.total_tokens_used, board.tokens_input,
                              board.tokens_output, board.cost_by_model["prior"]["calls"])
        observed["new_id"] = gen_entry_id()
        raise RuntimeError("stop after restored orchestration")

    monkeypatch.setenv("SWARM_FABLE_MODEL", "")
    monkeypatch.setattr(swarm, "run_orchestrator", inspect_orchestration)
    monkeypatch.setattr(swarm, "generate_seed", lambda *_: pytest.fail("seed repeated"))
    monkeypatch.setattr(swarm, "_execute_initial_reading",
                        lambda *_: pytest.fail("initial reading repeated"))
    with pytest.raises(RuntimeError, match="restored orchestration"):
        swarm.run_swarm(task, NoProvider(), max_iterations=2, resume_checkpoint=path)
    assert int(observed.pop("new_id")[1:]) > 20
    assert observed == {
        "iteration": 2, "entry": "e20", "read_status": "fully_read",
        "text": "Evidence in source.", "signals": ["s12"],
        "tokens": (20, 13, 7, 2),
    }

    source.write_text("Changed source.", encoding="utf-8")
    with pytest.raises(ValueError, match="source files differ"):
        swarm.run_swarm(task, NoProvider(), max_iterations=2, resume_checkpoint=path)
    source.write_text("Evidence in source.", encoding="utf-8")
    changed_instruction = Task("Different instruction", task.documents, output_dir=task.output_dir)
    with pytest.raises(ValueError, match="task instruction differs"):
        swarm.run_swarm(changed_instruction, NoProvider(), max_iterations=2,
                        resume_checkpoint=path)
    payload = json.loads(path.read_text())
    payload["blackboard"]["entries"][0].pop("opens_questions")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete checkpoint records"):
        swarm.run_swarm(task, NoProvider(), max_iterations=2, resume_checkpoint=path)
    payload["blackboard"]["entries"][0]["opens_questions"] = []
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        swarm.run_swarm(task, NoProvider(), max_iterations=2, resume_checkpoint=path)


def test_final_iteration_resume_keeps_prior_and_later_usage(tmp_path, monkeypatch):
    task, _source, path = _checkpoint(tmp_path, ended=True)
    monkeypatch.setenv("SWARM_FABLE_MODEL", "")
    monkeypatch.setattr(swarm, "run_orchestrator", lambda *_a, **_k: pytest.fail("loop repeated"))
    monkeypatch.setattr(swarm, "_run_due_deltas", lambda *_a, **_k: None)
    monkeypatch.setattr(swarm, "curate_entries", lambda *_: ([], 0))
    monkeypatch.setattr(swarm, "build_synthesis_packet", lambda *_: {})
    monkeypatch.setattr(swarm, "write_synthesis_packet_report", lambda *_: None)
    monkeypatch.setattr(swarm, "source_claim_verification_enabled", lambda: False)

    def synthesize(*_args):
        set_last_call_usage({"later": {"input": 3, "output": 2, "total": 5, "calls": 1}})
        return "Finished", 5

    monkeypatch.setattr(swarm, "synthesize_deliverable", synthesize)
    deliverable, board = swarm.run_swarm(
        task, NoProvider(), max_iterations=2, resume_checkpoint=path,
    )
    assert deliverable == "Finished"
    assert (board.total_tokens_used, board.tokens_input, board.tokens_output) == (25, 16, 9)
    assert board.cost_by_model["prior"]["calls"] == 2
    assert board.cost_by_model["later"]["calls"] == 1
