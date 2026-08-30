"""Unit tests for eml ingestion (S-10).

Builds a minimal .eml fixture in memory using stdlib email,
then verifies that read_eml() correctly extracts headers, body, and attachments.
"""
from __future__ import annotations

import email as email_lib
import email.policy
import io
import tempfile
from email.message import EmailMessage
from pathlib import Path

import pytest

from src.ingestion.eml import read_eml


@pytest.fixture()
def simple_eml_path(tmp_path: Path) -> Path:
    """Create a simple plain-text .eml file."""
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Test Subject"
    msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0000"
    msg.set_content("Hello, this is the email body.")
    p = tmp_path / "test.eml"
    p.write_bytes(msg.as_bytes(policy=email.policy.SMTP))
    return p


@pytest.fixture()
def eml_with_attachment_path(tmp_path: Path) -> Path:
    """Create an .eml file with a CC header and an attachment."""
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["To"] = "receiver@example.com"
    msg["Cc"] = "cc@example.com"
    msg["Subject"] = "With Attachment"
    msg["Date"] = "Tue, 02 Jan 2024 08:00:00 +0000"
    msg.set_content("Body text here.")
    msg.add_attachment(b"fake pdf bytes", maintype="application", subtype="pdf",
                       filename="report.pdf")
    p = tmp_path / "attachment.eml"
    p.write_bytes(msg.as_bytes(policy=email.policy.SMTP))
    return p


class TestReadEmlHeaders:
    def test_from_header_extracted(self, simple_eml_path):
        text, meta = read_eml(simple_eml_path)
        assert "alice@example.com" in text
        assert meta["headers"]["from"] == "alice@example.com"

    def test_to_header_extracted(self, simple_eml_path):
        text, meta = read_eml(simple_eml_path)
        assert "bob@example.com" in text
        assert meta["headers"]["to"] == "bob@example.com"

    def test_subject_header_extracted(self, simple_eml_path):
        text, meta = read_eml(simple_eml_path)
        assert "Test Subject" in text
        assert meta["headers"]["subject"] == "Test Subject"

    def test_cc_header_extracted_when_present(self, eml_with_attachment_path):
        text, meta = read_eml(eml_with_attachment_path)
        assert "cc@example.com" in text
        assert meta["headers"]["cc"] == "cc@example.com"

    def test_cc_header_absent_when_not_set(self, simple_eml_path):
        text, meta = read_eml(simple_eml_path)
        assert meta["headers"]["cc"] == ""


class TestReadEmlBody:
    def test_body_text_present_in_output(self, simple_eml_path):
        text, _ = read_eml(simple_eml_path)
        assert "Hello, this is the email body." in text


class TestReadEmlAttachments:
    def test_no_attachments_returns_empty_list(self, simple_eml_path):
        _, meta = read_eml(simple_eml_path)
        assert meta["attachments"] == []

    def test_attachment_filename_detected(self, eml_with_attachment_path):
        text, meta = read_eml(eml_with_attachment_path)
        assert "report.pdf" in meta["attachments"]
        assert "report.pdf" in text


class TestReadEmlErrorHandling:
    def test_nonexistent_file_returns_error_string(self, tmp_path):
        bad_path = tmp_path / "nonexistent.eml"
        text, meta = read_eml(bad_path)
        assert "error" in text.lower()
        assert meta == {}
