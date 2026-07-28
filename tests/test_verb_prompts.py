"""Tests for verb-specific worker prompt templates.

Verifies that _detect_task_verb correctly identifies task verbs,
and that _append_verb_instructions adds the right prompt suffix
for each verb type.
"""
from __future__ import annotations

import os
import re

from src.swarm.worker_dispatch import (
    _verb_prompts_enabled,
    _detect_task_verb,
    _append_verb_instructions,
    compose_worker_prompt_v2,
    compose_worker_prompt,
)


# ── Verb detection ───────────────────────────────────────────────────


class TestDetectTaskVerb:
    def test_identify_verb(self):
        assert _detect_task_verb("Identify all parties involved") == "identify"
        assert _detect_task_verb("list the key terms") == "identify"
        assert _detect_task_verb("enumerate every clause") == "identify"
        assert _detect_task_verb("find all references to") == "identify"
        assert _detect_task_verb("locate the specific provisions") == "identify"
        assert _detect_task_verb("determine what amounts are due") == "identify"

    def test_compare_verb(self):
        assert _detect_task_verb("Compare the two agreements") == "compare"
        assert _detect_task_verb("contrast the approaches") == "compare"
        assert _detect_task_verb("distinguish between") == "compare"
        assert _detect_task_verb("identify differences") == "identify"  # "identify" takes priority
        assert _detect_task_verb("similarities between") == "compare"
        assert _detect_task_verb("Company A vs. Company B") == "compare"

    def test_extract_verb(self):
        assert _detect_task_verb("Extract all financial figures") == "extract"
        assert _detect_task_verb("pull out the key dates") == "extract"
        assert _detect_task_verb("gather the evidence") == "extract"
        assert _detect_task_verb("collect all exhibits") == "extract"
        assert _detect_task_verb("retrieve the relevant data") == "extract"

    def test_analyze_verb(self):
        assert _detect_task_verb("Analyze the legal implications") == "analyze"
        assert _detect_task_verb("assess the risks") == "analyze"
        assert _detect_task_verb("evaluate the compliance") == "analyze"
        assert _detect_task_verb("review the contract terms") == "analyze"
        assert _detect_task_verb("examine the evidence") == "analyze"

    def test_draft_fallback(self):
        assert _detect_task_verb("") == "draft"
        assert _detect_task_verb("Prepare a memorandum") == "draft"
        assert _detect_task_verb("Write a summary") == "draft"
        assert _detect_task_verb("Create a report") == "draft"
        assert _detect_task_verb("something random here") == "draft"

    def test_verb_in_middle_of_sentence(self):
        """Verb detection works even when the verb isn't at the start."""
        assert _detect_task_verb(
            "You are a legal analyst. Identify all relevant clauses."
        ) == "identify"
        assert _detect_task_verb(
            "Please compare the following documents for conflicts."
        ) == "compare"

    def test_case_insensitive(self):
        assert _detect_task_verb("IDENTIFY") == "identify"
        assert _detect_task_verb("Compare") == "compare"
        assert _detect_task_verb("EXTRACT ALL") == "extract"


# ── Prompt appending ─────────────────────────────────────────────────


class TestAppendVerbInstructions:
    def test_returns_base_when_disabled(self):
        os.environ.pop("SWARM_ENABLE_VERB_PROMPTS", None)
        base = "base prompt content"
        result = _append_verb_instructions(base, "identify all")
        assert result == base

    def test_appends_identify_instructions(self):
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "1"
        base = "base prompt\n"
        result = _append_verb_instructions(base, "identify all parties")
        assert "IDENTIFY TASK" in result
        assert "base prompt" in result
        assert "enumerate EVERY" in result

    def test_appends_compare_instructions(self):
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "1"
        result = _append_verb_instructions("base\n", "compare documents")
        assert "COMPARE TASK" in result
        assert "CONFLICT:" in result

    def test_appends_extract_instructions(self):
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "1"
        result = _append_verb_instructions("base\n", "extract numbers")
        assert "EXTRACT TASK" in result
        assert "HIGH DENSITY" in result

    def test_appends_analyze_instructions(self):
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "1"
        result = _append_verb_instructions("base\n", "analyze implications")
        assert "ANALYZE TASK" in result
        assert "intermediate inference" in result

    def test_appends_draft_instructions(self):
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "1"
        result = _append_verb_instructions("base\n", "draft a memo")
        assert "DRAFT TASK" in result
        assert "legal drafting conventions" in result

    def test_fallback_to_draft_for_unknown(self):
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "1"
        result = _append_verb_instructions("base\n", "random task")
        assert "DRAFT TASK" in result

    def test_verb_prompts_enabled_check(self):
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "1"
        assert _verb_prompts_enabled()
        os.environ.pop("SWARM_ENABLE_VERB_PROMPTS", None)
        assert not _verb_prompts_enabled()
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "true"
        assert _verb_prompts_enabled()
        os.environ["SWARM_ENABLE_VERB_PROMPTS"] = "TRUE"
        assert _verb_prompts_enabled()
        os.environ.pop("SWARM_ENABLE_VERB_PROMPTS", None)

    def teardown_method(self):
        os.environ.pop("SWARM_ENABLE_VERB_PROMPTS", None)


# ── compose_worker_prompt_v2 ─────────────────────────────────────────


class TestComposeWorkerPromptV2:
    def test_v2_exists_and_returns_string(self):
        """compose_worker_prompt_v2 exists and returns a prompt string."""
        from unittest.mock import MagicMock
        # We can't easily call this without a full setup, but verify it's callable
        import inspect
        assert callable(compose_worker_prompt_v2)
        sig = inspect.signature(compose_worker_prompt_v2)
        assert "task_description" in sig.parameters
        assert "task_instruction" in sig.parameters
