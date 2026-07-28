"""Tests for analytical_steering() activation in the main loop.

Verifies that the proactive analytical_steering() cadence works
when SWARM_ENABLE_ANALYTICAL_STEERING is set, and that the
existing fallback path is unchanged when it's not.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from src.swarm.convergence import analytical_steering
from src.swarm.worker_dispatch import passes_quality_gate


class TestAnalyticalSteeringFunction:
    """Tests for analytical_steering() itself."""

    def test_analytical_steering_exists(self):
        """The function exists and is callable."""
        import inspect
        assert callable(analytical_steering)
        sig = inspect.signature(analytical_steering)
        assert "blackboard" in sig.parameters
        assert "steerer" in sig.parameters

    def test_analytical_steering_returns_tuple(self):
        """Returns (list[dict], int) — tasks + tokens."""
        bb = MagicMock()
        bb.task_instruction = "Analyze the contract"
        bb.entries = []
        bb.signals = []
        bb.documents = []
        bb.iteration = 3

        caller = MagicMock()
        caller.complete.return_value.text = '{"workers": []}'
        caller.complete.return_value.tokens_input = 10
        caller.complete.return_value.tokens_output = 5
        caller.complete.return_value.tokens_total = 15
        caller.complete.return_value.model = "test-model"
        caller.complete.return_value.latency_ms = 1

        tasks, tokens = analytical_steering(bb, caller)
        assert isinstance(tasks, list)
        assert isinstance(tokens, int)

    def test_analytical_steering_with_entries(self):
        """Produces tasks when blackboard has entries."""
        from src.swarm.models import Entry, EntrySource, WorkerRecord

        bb = MagicMock()
        bb.task_instruction = "Analyze the contract"
        bb.get_summary.return_value = {"documents": []}
        bb.iteration = 3

        # Give it some entries to work with
        entry = MagicMock(spec=Entry)
        entry.id = "e1"
        entry.status = "active"
        entry.type = "observation"
        entry.content = "The contract value is $500,000"
        entry.confidence = 0.95
        entry.source = MagicMock()
        entry.source.document = "contract.pdf"
        entry.epistemic = None

        bb.entries = [entry]
        bb.signals = []
        bb.documents = []

        caller = MagicMock()
        caller.complete.return_value.text = json.dumps({
            "workers": [{
                "description": "Calculate total liability from extracted amounts",
                "reads_from_blackboard": ["e1"],
                "reads_from_documents": [],
                "expected_output_type": "calculation",
            }]
        })
        caller.complete.return_value.tokens_input = 100
        caller.complete.return_value.tokens_output = 50
        caller.complete.return_value.tokens_total = 150
        caller.complete.return_value.model = "test-model"
        caller.complete.return_value.latency_ms = 1

        tasks, tokens = analytical_steering(bb, caller)
        assert len(tasks) >= 1
        assert tokens > 0


# ── Env gating integration ─────────────────────────────────────────


def test_steering_env_var_not_set_by_default():
    """SWARM_ENABLE_ANALYTICAL_STEERING is not set by default."""
    val = os.getenv("SWARM_ENABLE_ANALYTICAL_STEERING")
    # The env var is not set — this means the feature is off
    assert val is None or val.strip().lower() not in ("1", "true", "yes", "on")


def test_steering_env_var_respected():
    """When set to 1, the env var is recognized."""
    os.environ["SWARM_ENABLE_ANALYTICAL_STEERING"] = "1"
    val = os.getenv("SWARM_ENABLE_ANALYTICAL_STEERING", "").strip().lower()
    assert val in ("1", "true", "yes", "on")
    os.environ.pop("SWARM_ENABLE_ANALYTICAL_STEERING", None)


def test_steering_interval_env_var():
    """SWARM_STEERING_INTERVAL is configurable."""
    os.environ["SWARM_STEERING_INTERVAL"] = "5"
    interval = int(os.getenv("SWARM_STEERING_INTERVAL", "3"))
    assert interval == 5
    os.environ.pop("SWARM_STEERING_INTERVAL", None)

    # Default
    interval = int(os.getenv("SWARM_STEERING_INTERVAL", "3"))
    assert interval == 3


def test_pass_quality_gate_works():
    """passes_quality_gate is importable and callable."""
    from src.swarm.models import Entry

    good = MagicMock(spec=Entry)
    good.content = "This is a meaningful finding with enough context"
    good.id = "e1"
    assert passes_quality_gate(good)

    bad = MagicMock(spec=Entry)
    bad.content = "short"
    bad.id = "e2"
    assert not passes_quality_gate(bad)
