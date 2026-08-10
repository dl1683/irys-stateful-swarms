from __future__ import annotations

"""Create auditable review prompts without calling an LLM."""

import json

from .person_models import UncertainConnection


def generate_person_review_prompt(connection: UncertainConnection) -> str:
    return (
        "Review the uncertain connection between these person profiles. "
        "This output is advisory only, a human reviewer must make the final decision, and the LLM is not an identity adjudicator. "
        "Do not assume that a similar name, birth date, address, workplace, nationality, gender, profession, or family context proves identity. "
        "Similarities may instead reflect relatives, marriage, a shared household, colleagues, coincidence, stale data, identity misuse, or source error. "
        "Use only the supplied sourced information and preserve uncertainty. "
        "Return strict JSON with keys decision, confidence, rationale, supporting_evidence, conflicts, and follow_up_needed. "
        "decision must be one of same_person, connected_people, unrelated, uncertain.\n\n"
        + json.dumps(connection.to_dict(), indent=2, sort_keys=True)
    )
