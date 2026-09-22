from pathlib import Path
from unittest.mock import patch
from src.ingestion.docx import read_docx


def test_read_docx_dash_prefixed_filename(tmp_path: Path):
    """Regression test: Ensure filenames starting with dashes are placed after '--' delimiter."""
    dash_file = tmp_path / "--output=injection.docx"
    dash_file.write_bytes(b"PK\x03\x04")

    with patch("shutil.which", return_value="/usr/bin/pandoc"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "# Test Markdown"

        text, meta = read_docx(dash_file)

        assert mock_run.called, "subprocess.run was not invoked"
        cmd = mock_run.call_args[0][0]

        assert "--" in cmd, "Command missing '--' delimiter before positional arguments"

        delimiter_idx = cmd.index("--")
        assert str(dash_file.resolve()) in cmd[delimiter_idx + 1:], (
            "Dash-prefixed path was not placed after the '--' delimiter"
        )
        assert text == "# Test Markdown"
