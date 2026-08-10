from __future__ import annotations

"""Build auditable prompts for ambiguous pairs; this module makes no LLM call."""

import json

from .models import CompanyCandidate, CompanyRecord


def generate_review_prompt(
    left: CompanyRecord,
    right: CompanyRecord,
    candidate: CompanyCandidate,
    max_snippets: int = 2,
) -> str:
    payload = {
        "left": {
            "record_id": left.record_id,
            "name": left.name,
            "metadata": left.metadata,
            "snippets": left.snippets[:max_snippets],
        },
        "right": {
            "record_id": right.record_id,
            "name": right.name,
            "metadata": right.metadata,
            "snippets": right.snippets[:max_snippets],
        },
        "score": candidate.score,
        "evidence": list(candidate.evidence),
        "conflicts": list(candidate.conflicts),
    }
    return (
        "Decide whether these company records refer to the same legal entity. "
        "Beware parent/subsidiary/group ambiguity and related but distinct companies. "
        "Return strict JSON with keys decision, confidence, and rationale. "
        "decision must be one of same, different, uncertain.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )
