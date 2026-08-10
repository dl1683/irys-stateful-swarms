import json

from src.swarm.entity_maintenance_store import EntityMaintenanceState
from src.swarm.entity_specialist_review import review_pending_candidates
from src.swarm.models import ModelResult


class FakeCaller:
    def __init__(self, responses: list[dict[str, object]]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 8192,
                 temperature: float = 0.05, json_mode: bool = True) -> ModelResult:
        self.prompts.append(prompt)
        return ModelResult(
            text=json.dumps(self.responses.pop(0)),
            tokens_input=10,
            tokens_output=5,
            tokens_total=15,
            model="fake-model",
            latency_ms=1,
        )


class FailingCaller:
    def complete(self, prompt: str, **_: object) -> ModelResult:
        raise RuntimeError("provider unavailable")


def state_with_pending_candidate() -> EntityMaintenanceState:
    profiles = {
        "company:northwind-ltd": {
            "profile_id": "company:northwind-ltd",
            "entity_type": "company",
            "primary_name": "Northwind Ltd",
            "aliases": [],
            "source_card_ids": ["e1"],
            "facts": [{
                "field": "registration_number",
                "value": "CH-7788",
                "source_card_id": "e1",
                "quote": "Northwind Ltd registration CH-7788",
                "verified": True,
                "qualifiers": {},
            }],
        },
        "company:northwind-limited": {
            "profile_id": "company:northwind-limited",
            "entity_type": "company",
            "primary_name": "Northwind Limited",
            "aliases": [],
            "source_card_ids": ["e2"],
            "facts": [{
                "field": "registration_number",
                "value": "CH-7788",
                "source_card_id": "e2",
                "quote": "Northwind Limited registration CH-7788",
                "verified": True,
                "qualifiers": {},
            }],
        },
    }
    candidate = {
        "candidate_id": "candidate_northwind",
        "semantic_key": "company:company:northwind-limited|company:northwind-ltd",
        "evidence_fingerprint": "candidate_fp_northwind",
        "profile_ids": ["company:northwind-ltd", "company:northwind-limited"],
        "source_card_groups": [["e1"], ["e2"]],
        "score": 0.99,
        "evidence": ["exact_name", "same_registration_number"],
        "conflicts": [],
        "status": "pending_review",
    }
    return EntityMaintenanceState(profiles=profiles, candidates={"candidate_northwind": candidate})


def test_specialist_makes_one_call_and_accepts_same_entity_with_source_citations():
    state = state_with_pending_candidate()
    caller = FakeCaller([{
        "decision": "same_entity",
        "rationale": "The same registration appears on both source cards.",
        "citations": [
            {"source_card_id": "e1", "quote": "CH-7788"},
            {"source_card_id": "e2", "quote": "CH-7788"},
        ],
    }])

    summary = review_pending_candidates(state, tuple(state.candidates), caller)

    assert summary.model_calls == 1
    assert state.decisions[next(iter(state.decisions))]["outcome"] == "same_entity"


def test_uncertain_and_transport_failure_remain_sidecar_only_for_retry():
    state = state_with_pending_candidate()
    caller = FailingCaller()

    summary = review_pending_candidates(state, tuple(state.candidates), caller)

    assert summary.model_calls == 1
    assert state.candidates[next(iter(state.candidates))]["status"] == "review_retry"
    assert state.decisions == {}
