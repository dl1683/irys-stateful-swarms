"""Agent-bench bridge for irys stateful swarms.

Implements AgentBackend so the irys swarm can be evaluated against
agentic benchmarks that test multi-step reasoning, tool use, and
complex document analysis — capabilities where a swarm architecture
actually differentiates from a single LLM call.

Benchmark strategy (2026-06):
  PRIMARY tier: Harvey LAB, OfficeQA Pro, GAIA — these are THE
  priority. They test iterative document analysis, cross-document
  numerical reasoning, and multi-step tool-using agent capability.
  A single-call model cannot score well on these; our swarm should.

  EXPERIMENTAL tier: ARC-AGI-3 — interactive reasoning, separate
  evaluation harness.

  Single-call QA benchmarks (HotpotQA, FinanceBench, CUAD, etc.)
  were dropped — they test model quality, not architecture value.
  Good scores on those prove Gemini is good, not that our swarm is.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Benchmark tier definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSpec:
    name: str
    split: str
    tier: str  # "primary" | "experimental"
    category: str
    sota: str = ""


BENCHMARK_TIERS: list[BenchmarkSpec] = [
    # --- PRIMARY: Agentic benchmarks that test swarm architecture value ---
    # These are the #1 priority. A single LLM call cannot score well on
    # these; our swarm's iterative analysis, blackboard state, and
    # multi-pass extraction should meaningfully outperform.

    BenchmarkSpec(
        "harvey_lab", "tasks", "primary", "legal_document_analysis",
        sota="7.1% (Opus 4.7, single-model)",
    ),
    BenchmarkSpec(
        "officeqa_pro", "test", "primary", "cross_document_numerical",
        sota="66.2% (Opus 4.8); single-call <5%",
    ),
    BenchmarkSpec(
        "gaia", "validation", "experimental", "multi_step_tool_use",
        sota="74.6% (HAL agent); bare model ~45%",
    ),
    # --- EXPERIMENTAL: Requires separate evaluation harness ---
    BenchmarkSpec(
        "arc_agi_3", "test", "experimental", "interactive_reasoning",
    ),
]


def get_benchmarks(
    tiers: list[str] | None = None,
    categories: list[str] | None = None,
    names: list[str] | None = None,
) -> list[BenchmarkSpec]:
    """Filter benchmarks by tier, category, or name."""
    specs = BENCHMARK_TIERS
    if tiers:
        specs = [s for s in specs if s.tier in tiers]
    if categories:
        specs = [s for s in specs if s.category in categories]
    if names:
        specs = [s for s in specs if s.name in names]
    return specs


# ---------------------------------------------------------------------------
# AgentBackend implementation — wraps the irys swarm
# ---------------------------------------------------------------------------

def _ensure_agent_bench():
    """Add agent-bench to sys.path if not already importable."""
    try:
        import agent_bench  # noqa: F401
        return
    except ImportError:
        pass
    candidate = Path(os.getenv(
        "AGENT_BENCH_ROOT",
        Path.home() / "OneDrive" / "Desktop" / "Projects" / "agent-bench",
    ))
    src = candidate / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    elif candidate.exists():
        sys.path.insert(0, str(candidate))


class IrysSwarmBackend:
    """AgentBackend that runs the irys stateful swarm for each query."""

    name = "irys-swarm"
    version = "0.1.0"

    def __init__(
        self,
        worker_model: str | None = None,
        synthesis_model: str | None = None,
        token_budget: int | None = None,
    ):
        self._worker_model = worker_model
        self._synthesis_model = synthesis_model
        self._token_budget = token_budget

    async def run(
        self,
        *,
        query: str,
        context: str = "",
        max_tokens: int = 4096,
    ):
        _ensure_agent_bench()
        from agent_bench import AgentResult

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_sync, query, context, max_tokens,
        )
        return result

    def _run_sync(self, query: str, context: str, max_tokens: int):
        from agent_bench import AgentResult
        from .providers.gemini import GeminiCaller
        from .swarm import run_swarm
        from .swarm.models import Document, Task

        t0 = time.time()

        documents = []
        if context:
            documents.append(Document(
                id="context_0",
                name="source_context",
                text=context,
                size_bytes=len(context.encode("utf-8")),
            ))

        w_model = self._worker_model or os.getenv(
            "SWARM_WORKER_MODEL", "gemini-3.1-flash-lite",
        )
        s_model = self._synthesis_model or os.getenv(
            "SWARM_SYNTHESIS_MODEL", "gemini-3.5-flash",
        )
        r_model = os.getenv("SWARM_REVIEWER_MODEL", "gemini-3.5-flash")

        worker_caller = GeminiCaller(model=w_model)
        sm_model = os.getenv("SWARM_SMART_MODEL", "gemini-3.5-flash")
        smart_caller = GeminiCaller(model=sm_model) if sm_model != w_model else worker_caller
        synthesis_caller = (
            GeminiCaller(model=s_model) if s_model != w_model else worker_caller
        )
        reviewer_caller = GeminiCaller(model=r_model) if r_model else None

        with tempfile.TemporaryDirectory(prefix="irys_bench_") as tmp:
            task = Task(
                instruction=query,
                documents=documents,
                metadata={"title": query[:100], "work_type": "benchmark"},
                output_dir=tmp,
            )

            try:
                deliverable, blackboard = run_swarm(
                    task, worker_caller,
                    smart_caller=smart_caller,
                    synthesis_caller=synthesis_caller,
                    reviewer_caller=reviewer_caller,
                    token_budget=self._token_budget,
                )
            except Exception as e:
                return AgentResult(
                    answer="",
                    error=str(e),
                    latency_ms=int((time.time() - t0) * 1000),
                )

            answer = deliverable if isinstance(deliverable, str) else "\n\n".join(
                f"--- {k} ---\n{v}" for k, v in deliverable.items()
            )

            return AgentResult(
                answer=answer,
                tokens_in=blackboard.tokens_input,
                tokens_out=blackboard.tokens_output,
                cost_usd=_estimate_cost(blackboard),
                latency_ms=int((time.time() - t0) * 1000),
            )


def _estimate_cost(blackboard) -> float:
    MODEL_PRICING = {
        "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
        "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
        "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
        "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
        "claude-fable-5": {"input": 10.00, "output": 50.00},
        "claude-opus-4-8": {"input": 5.00, "output": 25.00},
        "claude-opus-4-7": {"input": 5.00, "output": 25.00},
        "claude-opus-4-6": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    }
    DEFAULT_PRICING = {"input": 0.25, "output": 1.50}
    cost = 0.0
    for model, usage in blackboard.cost_by_model.items():
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost += usage["input"] * pricing["input"] / 1_000_000
        cost += usage["output"] * pricing["output"] / 1_000_000
    if not blackboard.cost_by_model:
        cost = (
            blackboard.tokens_input * DEFAULT_PRICING["input"]
            + blackboard.tokens_output * DEFAULT_PRICING["output"]
        ) / 1_000_000
    return round(cost, 4)


# ---------------------------------------------------------------------------
# AgentBenchScorer — bridges agent-bench scoring into our scorer protocol
# ---------------------------------------------------------------------------

class AgentBenchScorer:
    """Wraps agent-bench's per-benchmark scorers into our Scorer protocol."""

    def __init__(self, benchmark: str):
        _ensure_agent_bench()
        from agent_bench import SCORERS
        if benchmark not in SCORERS:
            raise ValueError(
                f"No agent-bench scorer for '{benchmark}'. "
                f"Available: {sorted(SCORERS.keys())}"
            )
        self._benchmark = benchmark
        self._scorer_fn = SCORERS[benchmark]

    def score_task(self, task_data: dict, run_dir: Path, **kwargs):
        from .scoring import ScoreResult

        output_dir = run_dir / "output"
        output_text = ""
        if output_dir.exists():
            for f in sorted(output_dir.iterdir()):
                if f.is_file():
                    try:
                        output_text += f.read_text(
                            encoding="utf-8-sig", errors="replace",
                        )
                    except Exception:
                        pass

        expected = task_data.get("expected", "")
        query = task_data.get("query", task_data.get("instructions", ""))

        score_val, detail = self._scorer_fn(
            output_text, expected,
            question=query, context=task_data.get("context", ""),
        )

        passed = 1 if score_val >= 0.5 else 0
        return ScoreResult(
            score=max(score_val, 0.0),
            max_score=1.0,
            all_pass=score_val >= 0.5,
            n_criteria=1,
            n_passed=passed,
            criteria_results=[{
                "title": self._benchmark,
                "verdict": "pass" if passed else "fail",
                "reasoning": detail,
                "score": score_val,
            }],
            scorer_type=f"agent_bench:{self._benchmark}",
        )


# ---------------------------------------------------------------------------
# Suite runner — run multiple benchmarks
# ---------------------------------------------------------------------------

async def run_benchmark_suite(
    *,
    tiers: list[str] | None = None,
    names: list[str] | None = None,
    limit: int | None = None,
    concurrency: int = 1,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    worker_model: str | None = None,
    synthesis_model: str | None = None,
) -> dict[str, Any]:
    """Run a suite of agent-bench benchmarks against the irys swarm."""
    _ensure_agent_bench()
    from agent_bench import run_benchmark

    specs = get_benchmarks(tiers=tiers, names=names)
    specs = [s for s in specs if s.name != "arc_agi_3"]

    if not data_dir:
        agent_bench_root = Path(os.getenv(
            "AGENT_BENCH_ROOT",
            Path.home() / "OneDrive" / "Desktop" / "Projects" / "agent-bench",
        ))
        data_dir = agent_bench_root / "benchmarks" / "data"

    if not results_dir:
        results_dir = Path("results") / "bench"
    results_dir.mkdir(parents=True, exist_ok=True)

    backend = IrysSwarmBackend(
        worker_model=worker_model,
        synthesis_model=synthesis_model,
    )

    summaries = {}
    for spec in specs:
        print(f"\n{'='*60}")
        print(f"Benchmark: {spec.name}:{spec.split} "
              f"[{spec.tier}/{spec.category}]")
        print(f"{'='*60}")

        bench_results = results_dir / spec.name
        bench_results.mkdir(parents=True, exist_ok=True)

        completed_list = bench_results / "completed.jsonl"

        try:
            summary = await run_benchmark(
                benchmark=spec.name,
                split=spec.split,
                backend=backend,
                data_dir=data_dir,
                results_dir=bench_results,
                limit=limit,
                concurrency=concurrency,
                completed_list_path=completed_list,
            )
            summaries[spec.name] = {
                "benchmark": spec.name,
                "split": spec.split,
                "tier": spec.tier,
                "category": spec.category,
                "scored": summary.scored,
                "pass_count": summary.pass_count,
                "avg_score": round(summary.avg_score, 4),
                "ci_lower": round(summary.ci_lower, 4),
                "ci_upper": round(summary.ci_upper, 4),
                "total_cost_usd": round(summary.total_cost_usd, 4),
                "examples_attempted": summary.examples_attempted,
            }
            print(f"  Score: {summary.avg_score:.1%} "
                  f"({summary.pass_count}/{summary.scored} passed)")
        except Exception as e:
            print(f"  ERROR: {e}")
            summaries[spec.name] = {
                "benchmark": spec.name,
                "error": str(e),
            }

    suite_report = results_dir / "suite_report.json"
    suite_report.write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8",
    )
    print(f"\nSuite report: {suite_report}")
    return summaries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def add_bench_subparser(sub):
    """Add the 'bench' subcommand to the CLI."""
    bench_p = sub.add_parser(
        "bench", help="Run agent-bench benchmarks against irys swarm",
    )
    bench_p.add_argument(
        "--tier", nargs="*", default=None,
        choices=["primary", "experimental"],
        help="Benchmark tiers to run (default: all)",
    )
    bench_p.add_argument(
        "--benchmark", nargs="*", default=None,
        help="Specific benchmark names to run",
    )
    bench_p.add_argument(
        "--limit", type=int, default=None,
        help="Max examples per benchmark (for smoke tests)",
    )
    bench_p.add_argument(
        "--concurrency", "-j", type=int, default=1,
        help="Parallel tasks per benchmark",
    )
    bench_p.add_argument(
        "--data-dir", type=Path, default=None,
        help="agent-bench data directory",
    )
    bench_p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Output directory for results",
    )
    bench_p.add_argument(
        "--worker-model", default=None,
        help="Override swarm worker model",
    )
    bench_p.add_argument(
        "--synthesis-model", default=None,
        help="Override swarm synthesis model",
    )
    bench_p.add_argument(
        "--list", action="store_true", dest="list_benchmarks",
        help="List available benchmarks and exit",
    )
    return bench_p


def cmd_bench(args):
    """Handle the 'bench' CLI subcommand."""
    if args.list_benchmarks:
        specs = get_benchmarks(
            tiers=args.tier,
            names=args.benchmark,
        )
        print(f"{'Name':<20} {'Split':<10} {'Tier':<12} {'Category':<28} {'SOTA'}")
        print("-" * 100)
        for s in specs:
            print(f"{s.name:<20} {s.split:<10} {s.tier:<12} {s.category:<28} {s.sota}")
        print(f"\nTotal: {len(specs)} benchmarks")
        return

    asyncio.run(run_benchmark_suite(
        tiers=args.tier,
        names=args.benchmark,
        limit=args.limit,
        concurrency=args.concurrency,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        worker_model=args.worker_model,
        synthesis_model=args.synthesis_model,
    ))
