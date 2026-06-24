from src.swarm.blackboard import Blackboard
from src.swarm.models import ModelResult
from src.swarm.synthesis import _merge_usage, _sectioned_synthesis, synthesize_deliverable


class FakeCaller:
    def __init__(self, responses: list[str], model: str = "fake-model"):
        self.responses = list(responses)
        self.model = model
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 8192,
                 temperature: float = 0.05, json_mode: bool = True) -> ModelResult:
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else ""
        return ModelResult(
            text=text,
            tokens_input=10,
            tokens_output=5,
            tokens_total=15,
            model=self.model,
            latency_ms=1,
        )


def _large_must_include(n: int = 51) -> list[dict]:
    return [{"section": "Risk", "summary": f"Explain risk {i}."} for i in range(n)]


def test_sectioned_synthesis_returns_nonempty_usage_dict_matching_tokens():
    blackboard = Blackboard(task_instruction="Prepare a large memo.")
    must_include = _large_must_include()
    caller = FakeCaller(["RISK_CHUNK_1", "RISK_CHUNK_2", "RISK_CHUNK_3"])

    draft, tokens, usage_by_model = _sectioned_synthesis(
        blackboard, must_include, [], caller,
    )

    assert usage_by_model
    assert tokens == 45
    assert sum(v["total"] for v in usage_by_model.values()) == tokens


def test_synthesize_deliverable_populates_blackboard_cost_by_model():
    blackboard = Blackboard(task_instruction="Prepare a large memo.")
    must_include = _large_must_include()
    caller = FakeCaller([
        "RISK_CHUNK_1", "RISK_CHUNK_2", "RISK_CHUNK_3",
        '{"missing":[]}',
    ])

    draft, tokens = synthesize_deliverable(blackboard, must_include, caller)
    blackboard.add_tokens_from_last_call(tokens)

    assert blackboard.cost_by_model
    assert blackboard.cost_by_model["fake-model"]["total"] == tokens


def test_parallel_section_drafts_token_counts_summed_across_calls():
    blackboard = Blackboard(task_instruction="Prepare a large memo.")
    must_include = _large_must_include()
    # Force a verify+augment cycle on top of the parallel section drafts so
    # multiple model calls happen both on worker threads and the coordinator.
    caller = FakeCaller([
        "RISK_CHUNK_1", "RISK_CHUNK_2", "RISK_CHUNK_3",
        '{"missing":[{"item_number":1,"summary":"Explain risk 0.","entry_id":""}]}',
        "AUGMENTED_RISK_CHUNK",
    ])

    draft, tokens = synthesize_deliverable(blackboard, must_include, caller)
    blackboard.add_tokens_from_last_call(tokens)

    assert tokens == 75
    assert blackboard.total_tokens_used == 75
    assert blackboard.cost_by_model["fake-model"]["total"] == 75
    assert blackboard.cost_by_model["fake-model"]["calls"] == 5


def test_merge_usage_accumulates_across_multiple_calls():
    target: dict = {}
    _merge_usage(target, {"m1": {"input": 10, "output": 5, "total": 15, "calls": 1}})
    _merge_usage(target, {"m1": {"input": 20, "output": 10, "total": 30, "calls": 1}})
    _merge_usage(target, {"m2": {"input": 1, "output": 1, "total": 2, "calls": 1}})

    assert target["m1"] == {"input": 30, "output": 15, "total": 45, "calls": 2}
    assert target["m2"] == {"input": 1, "output": 1, "total": 2, "calls": 1}


def test_merge_usage_ignores_non_dict_source():
    target: dict = {"m1": {"input": 1, "output": 1, "total": 2, "calls": 1}}
    _merge_usage(target, None)
    _merge_usage(target, "not a dict")

    assert target == {"m1": {"input": 1, "output": 1, "total": 2, "calls": 1}}
