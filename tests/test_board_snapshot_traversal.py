from pathlib import Path
import pytest
from src.loop.state import Board


def test_board_snapshot_valid_label(tmp_path: Path):
    board = Board(instruction="test")
    board.output_dir = str(tmp_path)
    board.iteration = 1

    board.snapshot(label="step_a")

    expected_file = tmp_path / "loop" / "board_iter_1_step_a.json"
    assert expected_file.exists()
    assert expected_file.is_file()


@pytest.mark.parametrize("traversal_label", [
    "../../outside",
    "../escaped",
    "sub/escape",
    "sub\\escape",
    "sub/../../escape",
])
def test_board_snapshot_rejects_traversal(tmp_path: Path, traversal_label: str):
    board = Board(instruction="test")
    board.output_dir = str(tmp_path)
    board.iteration = 1

    with pytest.raises(ValueError, match="Snapshot path traversal detected"):
        board.snapshot(label=traversal_label)

    assert not (tmp_path / "outside.json").exists()
    assert not (tmp_path.parent / "outside.json").exists()
