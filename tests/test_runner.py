import json

from src.swarm.models import Document


def test_failed_swarm_writes_status_artifact(tmp_path, monkeypatch):
    from src import runner

    task_dir = tmp_path / "task"
    output_dir = tmp_path / "results"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(json.dumps({
        "instructions": "Assess the transaction.",
    }))
    monkeypatch.setattr(
        runner, "discover_documents",
        lambda *_: [Document("d1", "source.txt", "source")],
    )
    monkeypatch.setattr(runner, "GeminiCaller", lambda **_kwargs: object())
    monkeypatch.setattr(
        runner, "run_swarm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("request cap reached")),
    )

    result = runner.run_single_task(task_dir, output_dir, task_id="fixture/task")

    assert result.error == "swarm error: request cap reached"
    status = json.loads((output_dir / "fixture" / "task" / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == result.error
