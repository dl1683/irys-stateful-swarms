import json

import pytest

from src.swarm.blackboard import Blackboard
from src.swarm.models import (
    DocumentStatus, Entry, EntrySource, EpistemicStatus, Signal, WorkerRecord,
    gen_entry_id, gen_signal_id, reset_id_counters,
)


def test_checkpoint_round_trip_and_diagnostic_shape(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Agreement section one", encoding="utf-8")
    reset_id_counters()
    board = Blackboard(
        task_instruction="Review agreement", iteration=2, output_dir=str(tmp_path),
        documents=[DocumentStatus(
            id="doc1", name="source.txt", source_path=str(source),
            text="Agreement section one", read_status="partially_read",
            headings=["section one"], sections_read=["section one"],
        )],
        entries=[Entry(
            id="e41", content="An obligation", source=EntrySource("source.txt", "one", "quoted"),
            epistemic=EpistemicStatus(neutral_restatement="Neutral wording"),
            created_by=WorkerRecord("worker", "read", 1),
            opens_questions=["Who pays?"],
        )],
        signals=[Signal(id="s31", content="Who pays?", iteration_created=1)],
        total_tokens_used=19, tokens_input=12, tokens_output=7,
        cost_by_model={"fixture": {"input": 12, "output": 7, "total": 19, "calls": 2}},
        entity_overview_state={"overviews": {"Acme": {"summary": "Known"}}},
    )
    board.save_snapshot("post_2")
    path = board.save_checkpoint({"key_questions": ["Who pays?"]}, [3, 1], loop_ended=False)
    assert path is not None
    restored, seed, counts, ended, fingerprints = Blackboard.load_checkpoint(path)

    assert seed == {"key_questions": ["Who pays?"]}
    assert counts == [3, 1] and not ended
    assert [item["id"] for item in fingerprints] == ["doc1"]
    assert restored.entries[0].opens_questions == ["Who pays?"]
    assert restored.entries[0].epistemic.neutral_restatement == "Neutral wording"
    assert restored.signals[0].iteration_created == 1
    assert restored.documents[0].sections_read == ["section one"]
    assert restored.entity_overview_state == board.entity_overview_state
    assert (restored.total_tokens_used, restored.tokens_input, restored.tokens_output) == (19, 12, 7)
    assert restored.cost_by_model == board.cost_by_model
    assert int(gen_entry_id()[1:]) > 41
    assert int(gen_signal_id()[1:]) > 31

    diagnostic = json.loads((tmp_path / "swarm" / "blackboard_iter_2_post_2.json").read_text())
    assert set(diagnostic) == {
        "task_instruction", "documents", "entries", "signals", "iteration",
        "total_tokens_used", "token_budget", "entity_overview_state",
    }
    assert "opens_questions" not in diagnostic["entries"][0]

    path.write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        Blackboard.load_checkpoint(path)
    path.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid checkpoint"):
        Blackboard.load_checkpoint(path)
