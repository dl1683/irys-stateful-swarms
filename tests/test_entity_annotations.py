from src import swarm
from src.swarm.blackboard import Blackboard
from src.swarm.models import DocumentStatus, SectionIndex, SectionRange, Task
from src.swarm.worker_dispatch import parse_worker_output


def test_direct_document_context_keeps_unquoted_worker_entity():
    entries = parse_worker_output(
        {"findings": [{
            "type": "observation",
            "content": "Northwind Limited signed the agreement.",
            "source_document": "agreement.pdf",
            "source_section": "Signature",
            "entities": [{"entity_type": "company", "name": "Northwind Limited", "attributes": []}],
        }]},
        1, "w1", "read", {"agreement.pdf"}, direct_document_context=True,
    )

    assert entries[0].direct_document_context is True
    assert entries[0].entities[0]["name"] == "Northwind Limited"
    assert entries[0].entity_annotation_provenance == [{"method": "worker"}]


def test_blackboard_only_worker_is_not_direct_document_context():
    entries = parse_worker_output(
        {"findings": [{"type": "analysis", "content": "Northwind Limited is material."}]},
        2, "w2", "analysis", set(), direct_document_context=False,
    )

    assert entries[0].direct_document_context is False


def test_invalid_entity_annotation_is_audited_without_dropping_card():
    entries = parse_worker_output(
        {"findings": [{
            "type": "observation", "content": "Northwind signed the agreement.",
            "source_document": "agreement.pdf", "source_section": "Signature",
            "evidence": "Northwind signed the agreement.",
            "entities": [{"entity_type": "company", "name": "Invented Co", "attributes": []}],
        }]},
        1, "w1", "read", {"agreement.pdf"}, direct_document_context=True,
    )

    assert len(entries) == 1
    assert entries[0].entities == []
    assert entries[0].entity_annotation_rejections == ["entity_name_not_in_evidence"]


def test_direct_document_context_validates_nonempty_evidence_without_source_document():
    entries = parse_worker_output(
        {"findings": [{
            "type": "observation", "content": "Northwind signed the agreement.",
            "evidence": "Northwind signed the agreement.",
            "entities": [{"entity_type": "company", "name": "Invented Co", "attributes": []}],
        }]},
        1, "w1", "read", {"agreement.pdf"}, direct_document_context=True,
    )

    assert entries[0].entities == []
    assert entries[0].entity_annotation_rejections == ["entity_name_not_in_evidence"]


def test_initial_reader_binds_known_source_without_quote_payload(monkeypatch):
    source_text = "Dimitri Volkov signed the agreement on 1 January 2026. " * 2
    blackboard = Blackboard(documents=[DocumentStatus(
        id="d1",
        name="agreement.pdf",
        text=source_text,
        section_index=SectionIndex([SectionRange("Signature", 0, len(source_text), 1)]),
        sections_unread=["Signature"],
    )])
    prompts = []

    def fake_call(caller, prompt, max_tokens):
        prompts.append(prompt)
        return {"findings": [
            {
                "content": "Dimitri Volkov signed the agreement on 1 January 2026.",
                "type": "observation",
            },
            {
                "content": "Dmitri K. Volkov signed the agreement on 1 January 2026.",
                "type": "observation",
                "source_document": "wrong.pdf",
                "source_section": "Wrong section",
            },
        ]}, 0

    monkeypatch.setattr(swarm, "call_model", fake_call)

    entries, _ = swarm._execute_initial_reading(
        blackboard, Task("Extract facts", []), object(),
    )

    assert 'source_document: exactly "agreement.pdf"' not in prompts[0]
    assert 'source_section: exactly "Signature"' not in prompts[0]
    assert "evidence: an exact contiguous quote" not in prompts[0]
    assert [(entry.source.document, entry.source.section) for entry in entries] == [
        ("agreement.pdf", "Signature"),
        ("agreement.pdf", "Signature"),
    ]
    assert [entry.source.evidence for entry in entries] == ["", ""]
