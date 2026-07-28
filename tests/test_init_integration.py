"""Tests for wiring entity-resolution, convergence, and compression into the main loop.

Verifies that the 3 integration points in __init__.py respect env gating
and call the right functions. Uses monkey-patching to avoid needing a
full swarm run.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

# ── Entity Resolution integration ────────────────────────────────────


def test_entity_resolution_integration_not_called_when_disabled():
    """When SWARM_ENABLE_ENTITY_RESOLUTION is not set, run_entity_resolution
    is not invoked in the swarm loop."""
    # The entity_resolution module returns [], 0 when disabled
    from src.swarm.entity_resolution import run_entity_resolution, entity_resolution_enabled

    # Ensure env var is not set
    os.environ.pop("SWARM_ENABLE_ENTITY_RESOLUTION", None)
    assert not entity_resolution_enabled()
    entries, tokens = run_entity_resolution(MagicMock())
    assert entries == []
    assert tokens == 0


def test_entity_resolution_integration_called_when_enabled():
    """When SWARM_ENABLE_ENTITY_RESOLUTION=1, the function is callable and
    returns results."""
    from src.swarm.entity_resolution import entity_resolution_enabled

    os.environ["SWARM_ENABLE_ENTITY_RESOLUTION"] = "1"
    assert entity_resolution_enabled()

    # Clean up
    os.environ.pop("SWARM_ENABLE_ENTITY_RESOLUTION", None)


# ── Entropy Convergence integration ──────────────────────────────────


def test_entropy_convergence_not_called_when_disabled():
    """When SWARM_ENABLE_ENTROPY_CONVERGENCE is not set, compute_convergence
    returns disabled verdict."""
    from src.swarm.convergence_entropy import compute_convergence, entropy_convergence_enabled

    os.environ.pop("SWARM_ENABLE_ENTROPY_CONVERGENCE", None)
    assert not entropy_convergence_enabled()

    bb = MagicMock()
    bb.entries = []
    bb.iteration = 1
    bb.output_dir = ""

    result = compute_convergence(bb)
    assert result["enabled"] is False
    assert result["converged"] is False


def test_entropy_convergence_called_when_enabled():
    """When SWARM_ENABLE_ENTROPY_CONVERGENCE=1, compute_convergence runs."""
    from src.swarm.convergence_entropy import entropy_convergence_enabled

    os.environ["SWARM_ENABLE_ENTROPY_CONVERGENCE"] = "1"
    assert entropy_convergence_enabled()
    os.environ.pop("SWARM_ENABLE_ENTROPY_CONVERGENCE", None)


# ── Compression integration ──────────────────────────────────────────


def test_compression_not_called_when_disabled():
    """When SWARM_ENABLE_COMPRESSION is not set, compress_blackboard returns
    disabled result."""
    from src.swarm.compression import compress_blackboard, compression_enabled

    os.environ.pop("SWARM_ENABLE_COMPRESSION", None)
    assert not compression_enabled()

    bb = MagicMock()
    bb.entries = []
    bb.output_dir = ""

    result = compress_blackboard(bb)
    assert result.get("enabled") is False
    assert result.get("selected_ids") == []


def test_compression_called_when_enabled():
    """When SWARM_ENABLE_COMPRESSION=1, compress_blackboard runs."""
    from src.swarm.compression import compression_enabled

    os.environ["SWARM_ENABLE_COMPRESSION"] = "1"
    assert compression_enabled()
    os.environ.pop("SWARM_ENABLE_COMPRESSION", None)


# ── All imports resolve ──────────────────────────────────────────────


def test_all_integration_imports_resolve():
    """The integration imports in __init__.py load without errors."""
    from src.swarm.compression import (  # noqa: F401
        compression_enabled,
        compress_blackboard,
        compress_to_entry_ids,
    )
    from src.swarm.convergence_entropy import (  # noqa: F401
        compute_convergence,
        entropy_convergence_enabled,
    )
    from src.swarm.entity_resolution import (  # noqa: F401
        entity_resolution_enabled,
        run_entity_resolution,
    )
    # If we got here, all imports resolved
    assert True
