# irys-stateful-swarms

**The highest all-pass rate on the Legal Agent Benchmark at $4.64/task.** On all 2,010 tasks in the [Harvey Legal Agent Benchmark (LAB) v1.0](https://github.com/harveyai/harvey-labs) — 27 legal practice areas — irys-stateful-swarms achieves **32.5% strict all-pass** and **91.44% criteria macro** at **$4.64/task**, exceeding every published result including Harvey's own post-trained [Tenet model](https://www.harvey.ai/blog/post-training-update-harvey-tenet) (19.7%) and all frontier model baselines — with no fine-tuning, no custom training data, and no domain-specific scaffolding. See [benchmark comparison](#benchmark-comparison), [verification](#verification), and [benchmark context](#context).

PS: We use Gemini 3.5 Flash Lite as a judge because it's intelligent, cheap, and really good with rate limits, letting us work faster. We have done evals to ensure that the model is in agreement with other judges. You're free to rescore the entire run with the judge you choose if you disagree with our judge choice

### At a glance

![Harvey LAB — All-Pass Rate](assets/lab_allpass_rate.png)

![Performance vs Cost — Harvey LAB](assets/lab_performance_vs_cost.png)

**62x** more intelligence per dollar than Fable 5. **50x** more than Opus 4.7. **65%** higher all-pass than Harvey Tenet — with zero training.

![What $100 Buys You on LAB](assets/lab_what_100_buys.png)

![Cost per Point of Quality](assets/lab_cost_per_point.png)

![Training Investment vs Performance](assets/lab_training_vs_performance.png)

## Contents

- [Why this matters](#why-this-matters)
- [Full benchmark results](#full-benchmark-results)
  - [Benchmark comparison](#benchmark-comparison)
  - [Frontier cost analysis](#frontier-cost-analysis)
  - [The stateful advantage](#the-stateful-advantage)
- [How stateful swarms reason](#how-stateful-swarms-reason)
- [Why stateful swarms matter](#why-stateful-swarms-matter)
- [Blackboard MCP: Claude Code and Codex](#blackboard-mcp-use-stateful-reasoning-in-claude-code-and-codex)
- [Beyond legal: domain-agnostic](#beyond-legal-the-stateful-swarm-paradigm-is-domain-agnostic)
  - [SWE-bench Verified](#swe-bench-verified-preliminary-scaffold-evaluation)
  - [Datadog 10-K strategic analysis](#datadog-10-k-strategic-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Sources](#sources)

The published run deliberately starts every task from zero prior state — no document pre-processing, no persistent knowledge graphs, no entity pre-linking, and no blackboard reuse across tasks. [Irys](https://www.irys.ai) has proprietary ingestion infrastructure purpose-built for DMS-scale document analysis (hierarchical embeddings, entity linking, obligation extraction, structural parsing), but none of it was used in this benchmark. The firm-knowledge family alone (250 tasks over 9,288 documents) is exactly where that infrastructure would have the largest impact. These results reflect the raw coordination architecture only.

For a technical discussion of the stateful swarm paradigm and the ideas behind this system, see [Stateful Swarms Make AI Agents Cheaper, Safer, Better](https://www.linkedin.com/pulse/stateful-swarms-make-ai-agents-cheaper-safer-better-devansh-devansh-8enxe).

---

## Why this matters

Current AI systems forget everything between sessions. Every question pays the full cost of understanding from scratch — the same documents re-read, the same entities re-discovered, the same analysis re-derived. Context compaction destroys details. Session boundaries erase progress. RAG retrieves text fragments but not the analytical understanding built from them.

**Stateful swarms solve this.** Instead of treating AI reasoning as disposable single-shot computation, irys-stateful-swarms builds persistent, structured analytical state that survives across sessions, accumulates over time, and makes every subsequent interaction cheaper and more accurate than the last. The system coordinates multiple AI agents through a shared, evolving blackboard — a typed, provenance-tracked knowledge base where every observation, analysis, calculation, and gap is preserved with full source attribution. Nothing is summarized away. Nothing is forgotten.

This is not an incremental improvement to existing approaches. It is a paradigm shift: **from stateless inference to stateful reasoning.**

## Full benchmark results

irys-stateful-swarms completed the full public [Harvey Legal Agent Benchmark (LAB) v1.0](https://github.com/harveyai/harvey-labs): 2,010 tasks across 27 legal practice areas — including the new firm-knowledge family (250 tasks over a shared 9,288-document DMS). Every task starts from an empty blackboard with zero prior state so the run does not learn across benchmark tasks. Deployment-specific model assignments and provider routing are intentionally omitted from the public artifact.

| Metric | Result |
|---|---|
| Tasks completed | `2,010 / 2,010` |
| Criteria macro | `91.44%` |
| Criteria micro | `105,146 / 114,437 = 91.88%` |
| Strict all-pass | `654 / 2,010 = 32.5%` |
| Tasks at 95%+ | `1,260 / 2,010 = 62.7%` |
| Total cost | `$9,317` |
| Cost per task | `$4.64` |

### Benchmark comparison

Harvey published official LAB results for Tenet and frontier model baselines in their [Tenet Research Preview](https://www.harvey.ai/blog/post-training-update-harvey-tenet) (August 2026). The table below places irys-stateful-swarms alongside those results. Competitor all-pass rates are from Harvey's publication (holdout set, ~1,200 tasks). irys ran on the public set (2,010 tasks, 27 families).

| System | LAB All-Pass | Est. Cost/Task |
|---|---:|---:|
| **irys-stateful-swarms** | **32.5%** | **$4.64** |
| Muse Spark 1.1 | 20.0% | ~$0.50 |
| Harvey Tenet (Kimi K3 + RL) | 19.7% | ~$8 |
| Grok 4.5 | 12.9% | ~$1 |
| Fable 5 | 11.5% | ~$102 |
| Kimi K3 (base) | 10.8% | ~$8 |
| DeepSeek V4 Flash | 8.3% | ~$0.40 |
| Opus 4.7 | 7.1% | ~$51 |
| GLM-5.2 | 7.1% | — |
| Opus 5 | 6.7% | ~$51 |
| Gemini 3.6 Flash | 3.3% | ~$2 |
| GPT-5.6 Sol | 2.5% | ~$12 |

> **A note on cost and benchmark versions:** The Opus 4.7 cost (~$51/task) was measured on the **original** LAB release (1,251 tasks, 24 families). LAB v1.0 substantially expanded the benchmark to 2,010 tasks and 27 families, adding the firm-knowledge family (250 enterprise-search tasks over a shared 9,288-document DMS), an expanded contracts family, and diligence — all of which involve significantly longer documents and deeper cross-document reasoning than the original task set. Frontier model costs on the v1.0 benchmark would likely be **materially higher** than the original figures, because the new tasks require more tokens to process. irys cost ($4.64/task) is measured on the harder v1.0 benchmark — making the cost advantage over frontier models even larger than the raw numbers suggest.
>
> Harvey's published all-pass results are on their private holdout set (~1,200 tasks), which is not publicly available. There is no way for us to run irys on the holdout set or for Harvey to publish their models' costs on the public set, so a direct apples-to-apples cost comparison is not possible. Opus 4.7 cost from Harvey's [initial LAB publication](https://www.harvey.ai/blog/legal-agent-benchmark-initial-results). Fable 5 and Opus 5 costs are estimated from published per-token pricing applied to the Opus 4.7 baseline — see [frontier cost analysis](#frontier-cost-analysis). Other costs from Harvey's [Tenet Research Preview](https://www.harvey.ai/blog/post-training-update-harvey-tenet). If Harvey or any model provider publishes verified costs on the public benchmark, we will update this table accordingly.

Harvey Tenet is a Kimi K3 base model post-trained with reinforcement learning on ~1,750 legal task environments over 2 months on 150 NVIDIA B300 GPUs. Despite that investment, irys-stateful-swarms — a pure coordination architecture with no fine-tuning, no custom training data, and no domain-specific scaffolding — achieves 60% higher all-pass. The performance comes from the architecture: structured state-building, typed provenance, signal-driven gap identification, and multi-iteration convergence.

### Frontier cost analysis

Harvey's [initial LAB publication](https://www.harvey.ai/blog/legal-agent-benchmark-initial-results) (May 2026) reported that Claude Opus 4.7 cost **~$51/task** at 7.1% all-pass — the highest-performing model at the time. This published cost serves as the baseline for estimating newer Claude models, since all share the same tokenizer (introduced with Opus 4.7):

| Model | Input (per MTok) | Output (per MTok) | Tokenizer | Per-Token vs Opus 4.7 | Est. LAB Cost/Task |
|---|---:|---:|---|---|---:|
| Claude Opus 4.7 (published baseline) | $5.00 | $25.00 | Opus 4.7 | 1x | **~$51** |
| Claude Opus 5 | $5.00 | $25.00 | Opus 4.7 | 1x | **~$51** |
| Claude Fable 5 | $10.00 | $50.00 | Opus 4.7 | 2x | **~$102** |

> Opus 4.7 LAB cost from [Harvey's initial publication](https://www.harvey.ai/blog/legal-agent-benchmark-initial-results), measured on the **original** benchmark (1,251 tasks). The v1.0 benchmark that irys ran on (2,010 tasks) added substantially harder tasks — including 250 firm-knowledge tasks over a 9,288-document DMS — so real frontier costs on v1.0 would likely exceed these estimates. Per-token pricing from [Anthropic's published API rates](https://docs.anthropic.com/en/docs/about-claude/models). All three models use the same tokenizer (introduced with Opus 4.7), so the cost difference is purely per-token pricing. Fable 5 is exactly 2x on both input ($10 vs $5) and output ($50 vs $25), with mandatory extended thinking.

#### Head-to-head comparison

| | **irys** | Harvey Tenet | Fable 5 | Opus 4.7 |
|---|---:|---:|---:|---:|
| LAB All-Pass | **32.5%** | 19.7% | 11.5% | 7.1% |
| Cost/Task | **$4.64** | ~$8 | ~$102 | ~$51 |
| All-pass per dollar | **7.00** | 2.46 | 0.11 | 0.14 |
| Cost per all-pass point | **$0.14** | $0.41 | $8.87 | $7.18 |
| Training investment | **Zero** | 150 B300 GPUs, 2 months | — | — |

irys-stateful-swarms delivers **62x** the intelligence per dollar of Fable 5, **50x** Opus 4.7, and **2.8x** Harvey Tenet — with no fine-tuning, no custom training data, and no domain-specific scaffolding.

<details>
<summary>What $100 buys on LAB (data)</summary>

| System | Tasks per $100 | All-pass rate | Expected all-pass tasks per $100 |
|---|---:|---:|---:|
| **irys-stateful-swarms** | **21.6** | **32.5%** | **7.0** |
| Harvey Tenet | 12.5 | 19.7% | 2.5 |
| Opus 4.7 | 2.0 | 7.1% | 0.14 |
| Fable 5 | 1.0 | 11.5% | 0.11 |

</details>

<details>
<summary>Post-training vs coordination architecture (data)</summary>

| Approach | Base Model | Method | LAB All-Pass | Uplift |
|---|---|---|---:|---:|
| Raw model | Kimi K3 | None | 10.8% | — |
| Post-training | Kimi K3 | LoRA + GSPO, 150 B300 GPUs, 2 months | 19.7% | +82% |
| Coordination architecture | General-purpose models | Stateful swarms, zero training | **32.5%** | **+201%** |

Harvey invested in domain-specific post-training: RL over ~1,750 legal environments on 150 GPUs for 2 months. That lifted Kimi K3 from 10.8% to 19.7% (+82%). irys achieves +201% over the same baseline with zero training compute.

</details>

The most expensive frontier models deliver the worst results — Fable 5 at ~$102/task achieves only 11.5% all-pass, spending 22x what irys costs for 65% lower performance. Frontier intelligence alone does not solve long-horizon document analysis — coordination does.

### Verification

The complete outputs from the full benchmark run are available as downloadable archives in [GitHub Releases](https://github.com/dl1683/irys-stateful-swarms/releases). You can score these outputs yourself using the [Harvey LAB scorer](https://github.com/harveyai/harvey-labs) to independently verify these numbers.


### Context

The published run uses the public LAB task set. Private holdout results are not directly interchangeable with public-set results, so this README makes no claim of private-holdout equivalence. Judge agreement, deployment configuration, and other run limitations should be reviewed in the release artifacts before drawing comparisons.

### Architecture is the public claim

The public result measures the complete system, not an isolated model. The repository therefore makes no model-specific performance attribution. Its reusable contribution is the coordination architecture: structured state-building, typed provenance, signal-driven gap identification, and multi-iteration convergence.

The implementation supports multiple providers and explicit deployment configuration. Operators should choose and record their own configuration without committing production values.

### The stateful advantage

These results were achieved under the hardest possible condition: **zero prior state.** Every task starts from an empty blackboard, with no document memory, entity knowledge, or accumulated understanding.

#### What we deliberately did not use

The benchmark run excludes several [Irys](https://www.irys.ai) production capabilities that are purpose-built for exactly the kind of work LAB tests:

- **Proprietary document ingestion** — hierarchical structural parsing, section-level embeddings, table extraction, and format-aware chunking. On the benchmark, the system reads raw documents from scratch every time.
- **Persistent knowledge graphs** — entity linking, obligation tracking, cross-document relationship resolution. The benchmark starts with an empty graph per task.
- **Blackboard reuse** — in production, analytical state from prior queries persists and compounds. The benchmark forbids reuse: each of the 2,010 tasks starts from zero.
- **DMS-optimized retrieval** — the firm-knowledge family (250 tasks, 9,288 shared documents) is the exact use case Irys's ingestion pipeline is designed for. On the benchmark, the system receives the raw document set with no pre-processing.

These capabilities would have the largest impact on the firm-knowledge tasks (document management, multi-filing analysis, cross-reference extraction) — precisely the tasks where building prior state eliminates redundant work. The 32.5% all-pass rate reflects none of that advantage.

#### The production multiplier

In a stateful deployment, grounded document understanding can be retained and reused. Subsequent queries can build on that state instead of rediscovering the same source facts.

[Irys](https://www.irys.ai) combines stateful swarm coordination with hierarchical embeddings, persistent knowledge graphs, entity linking, and typed provenance tracking to reduce the cost of multi-turn inference by up to **1,000x** compared to stateless re-computation. The system doesn't spend tokens constantly re-reading documents, re-extracting entities, or re-deriving analyses it has already performed. Provenance tracking allows Irys to deterministically isolate exactly which state needs updating when new information arrives — rather than re-processing everything, the system targets only the affected subgraph. Combined with deterministic algorithms for entity resolution, obligation tracking, and conflict detection, the vast majority of follow-up work never touches an LLM at all.

This is the economic case for stateful swarms: the cost of AI-assisted analysis shifts from "pay full price for every question" to "invest in understanding once, then query cheaply forever."

## How stateful swarms reason

The best way to understand the stateful swarm paradigm is to look at how the system actually thinks. Each task produces a **blackboard** — persistent structured state that evolves over multiple iterations as workers read documents, extract evidence, cross-reference findings, and build toward a complete answer. This blackboard is the core artifact — not the final output, but the accumulated understanding that produced it.

A complete example is included in [`examples/compare-credit-agreement-to-commitment-letter/`](examples/compare-credit-agreement-to-commitment-letter/) — a banking task that scored **40/40 (perfect)**. You can browse every blackboard snapshot to see exactly how the system builds its understanding.

### Iteration 0 — The system plans before it reads

Before reading any document in detail, the seed planner scans document structure and produces a strategy with targeted questions:

```json
{
  "id": "e564",
  "type": "strategy",
  "content": "This is a comparison and issue-flagging task supported by targeted extraction. The approach involves extracting the baseline terms from the term sheet, commitment letter, and no-flex confirmation, extracting the corresponding drafted terms from the draft credit agreement, and performing a side-by-side gap analysis to identify any deviations, unauthorized changes, or missing provisions.",
  "created_by": {
    "worker_id": "seed_planner",
    "description": "analytical_framework",
    "iteration": 0
  }
}
```

The system then generates **signals** — specific questions that workers must answer:

```json
{
  "id": "s341",
  "type": "question",
  "content": "What are the exact interest rate margins, SOFR floors, and OID for Term Loan B in the draft credit agreement, and do they match the term sheet and commitment letter?",
  "origin_entry": "seed_plan",
  "priority": "high",
  "status": "open"
}
```

```json
{
  "id": "s346",
  "type": "question",
  "content": "What are the Asset Sale Prepayment terms (net proceeds percentage, annual threshold, reinvestment periods, cash consideration requirement) in the draft credit agreement, and do they deviate from the term sheet?",
  "origin_entry": "seed_plan",
  "priority": "high",
  "status": "open"
}
```

**7 entries, 12 open signals.** The swarm knows what it's looking for before reading a single page.

### Iteration 5 — Workers extract grounded evidence

Parallel workers read both documents and write structured findings to the blackboard. Each observation links to its source document, section, and evidence:

```json
{
  "id": "e716",
  "type": "observation",
  "content": "Northbrook Capital Markets, LLC commits to provide a first lien senior secured term loan B facility in an aggregate principal amount of $350,000,000.",
  "source": {
    "document": "commitment-letter.docx",
    "section": "Full Document (part 1)",
    "evidence": ""
  },
  "created_by": {
    "worker_id": "reader_commitment-letter.do",
    "description": "initial_reading",
    "iteration": 0
  },
  "confidence": 0.9
}
```

Workers also identify what's missing. Gap entries flag incomplete extraction and link back to the signals they're trying to answer:

```json
{
  "id": "e1977",
  "type": "gap",
  "content": "Rows 1.0 through 5.0 and 7.0 through 50.0 are currently unextracted from comparison-template.xlsx.",
  "source": {
    "document": "comparison-template.xlsx",
    "evidence": "Document 'comparison-template.xlsx' has ~50 enumerable items but only 0 extracted."
  },
  "created_by": {
    "worker_id": "w1_72fa",
    "description": "Enumerate all 50 row headers and baseline financial terms from comparison-template.xlsx to establish the comparison framework.",
    "iteration": 1
  },
  "addresses_signals": ["s483"]
}
```

**2,203 entries:** 2,023 observations, 78 calculations, 54 analyses, 41 gaps, and 7 strategies. The blackboard is dense with source-grounded facts.

### Iteration 12 — Cross-document analysis reveals deviations

By the final iteration, the system has built enough state for a stronger model to perform cross-document analysis. It finds specific deviations between the commitment letter and the draft credit agreement:

```json
{
  "id": "e865",
  "type": "analysis",
  "content": "Section 2.06 of the draft credit agreement specifies an annual agency fee of $50,000. This is a deviation from the Commitment Letter, which requires an Administrative Agent Fee of $150,000 per annum, payable annually in advance.",
  "source": {
    "document": "draft-credit-agreement.docx"
  },
  "created_by": {
    "worker_id": "flash35_analyst",
    "description": "direct_analysis",
    "iteration": 12
  },
  "confidence": 0.98,
  "supports": ["e260", "e261", "e35"]
}
```

```json
{
  "id": "e2331",
  "type": "gap",
  "content": "The 6-month soft call provision is absent from the draft credit agreement.",
  "source": {
    "document": "comparison-template.xlsx",
    "section": "6.0",
    "evidence": "Section 6.0 (Term Loan B — Voluntary Prepayment / Soft Call) shows no mapping to the draft credit agreement."
  },
  "created_by": {
    "worker_id": "w12_ecc1",
    "description": "Perform targeted re-extraction of comparison-template.xlsx to identify the remaining 48 missing items",
    "iteration": 12
  },
  "confidence": 0.98
}
```

**Final state: 2,400 entries** (2,044 observations, 113 analyses, 87 calculations, 135 gaps, 21 strategies). **210 signals** — 127 addressed, 45 still open, 38 expired. The system found 10+ material deviations — unauthorized margin increases, missing fee definitions, tightened covenant triggers, restricted reinvestment periods — each grounded in specific clauses from specific documents.

From 7 entries to 2,400 grounded findings — a **343x expansion** of structured analytical state over 12 iterations.

**This is what statefulness means in practice.** In a stateless system, all 2,400 entries would be discarded after generating the output. The next question about the same credit agreement would start from zero — re-reading the same documents, re-extracting the same terms, re-discovering the same deviations. In a stateful swarm, this analytical state persists. The next question costs a fraction of the first because the expensive understanding has already been built.

### When it doesn't get a perfect score, you can see exactly why

Not every task scores perfectly — but the blackboard makes failures **auditable**. You can trace exactly what the system knew, what it missed, and where the reasoning fell short.

**Example: International Sanctions Entity Extraction** ([`examples/extract-transaction-entity-details/`](examples/extract-transaction-entity-details/)) — scored **80/85**.

The system was asked to extract entity details from a complex sanctions transaction. Here's what happened on the five missed criteria:

**Missed: "Identify Haverford National Bank as OCC-chartered national bank"** — the system found the bank:

```json
{
  "id": "e266",
  "type": "observation",
  "content": "LC Issuing Bank: Haverford National Bank, 1200 Chestnut Street, Philadelphia, PA 19107, USA (SWIFT: HAVNUS33)"
}
```

It got the name, the exact street address, the city, the SWIFT code — but didn't identify the charter type. The fact is *there*, the classification step is what's missing.

**Missed: "Isabelle M. Renard — confirm Swiss/French dual nationality"** — the system found her:

```json
{
  "id": "e219",
  "type": "observation",
  "content": "Screening ID 9: Isabelle M. Renard (DOB: Not provided), Direct Shareholder of Crestmoor (27%), Switzerland / France"
}
```

It even extracted "Switzerland / France" — but didn't explicitly flag this as *dual nationality* in a way the scorer recognized.

**Missed: "Beneficiary name inconsistency"** — the system found *both* name variants in separate entries:

```json
{"id": "e95", "content": "The exporter is Zenith Petrochem Industries LLC, located in Jebel Ali Free Zone, UAE."}
```
```json
{"id": "e48", "content": "Zenith Petrochemical Industries LLC, Jebel Ali Free Zone, Dubai, UAE"}
```

Both "Zenith Petrochem" and "Zenith Petrochemical" are in the blackboard — the discrepancy is *visible in the state* — but no worker explicitly flagged the inconsistency.

**Missed: "OFAC 50% rule aggregation principle"** — the system got close:

```json
{
  "id": "e676",
  "type": "analysis",
  "content": "Orion Gulf's 49% stake in Zenith is 1% below the OFAC 50% rule threshold, but aggregate ownership by blocked persons could trigger a violation."
}
```

It identified the 49% threshold proximity and even mentioned aggregation — but didn't elaborate on the aggregation *principle* with enough specificity.

---

**Example: UCC Lien Extraction** ([`examples/extract-lien-and-debt-information/`](examples/extract-lien-and-debt-information/)) — scored **54/59**.

**Missed: "Debtor name discrepancy between filings"** — the system extracted both name variants:

```json
{"id": "e773", "content": "Debtor: Pinnacle Industrial Solutions, Inc., a corporation organized in Ohio, Charter No. 2187650"}
```
```json
{"id": "e885", "content": "Filing OH-2019-0178443 (Tristate Capital Equipment Corp.) against Pinnacle Industrial Solutions is LAPSED as of May 15, 2024."}
```

"Pinnacle Industrial Solutions, Inc." in one entry, "Pinnacle Industrial Solutions" (without Inc.) in another. The variance exists in the blackboard but wasn't flagged.

**Missed: "PMSI super-priority under UCC §9-324(a)"** — the system identified the concept:

```json
{
  "id": "e1062",
  "type": "observation",
  "content": "Allegheny Equipment Finance LLC holds a purchase-money security interest (PMSI) in five specific pieces of equipment, which may have super-priority status over the proposed senior secured credit facility regarding those specific assets."
}
```

It found the PMSI, recognized it *may have super-priority*, but didn't cite the specific UCC section.

---

**The pattern:** In every near-miss, the raw information was in the blackboard. The system read the right documents, extracted the right facts, and even flagged related concerns. What's missing is the final verification step — the explicit cross-reference, the legal citation, the formal classification. These are the kinds of failures that are **fixable through better state processing**, not fundamental architectural limitations.

This is what makes stateful swarms fundamentally different from stateless approaches. The blackboard doesn't just produce an answer — it produces a complete, inspectable, debuggable reasoning trace. Every conclusion is traceable to evidence. Every evidence entry is traceable to a source document. Every gap is explicitly logged. Instead of a black box, you get a structured analytical artifact that persists, accumulates, and improves over time.

### Explore the examples

| Example | Domain | Score | What it shows |
|---|---|---:|---|
| [`compare-credit-agreement-to-commitment-letter/`](examples/compare-credit-agreement-to-commitment-letter/) | Banking | 40/40 | Perfect cross-document deviation analysis |
| [`draft-safe-agreement/`](examples/draft-safe-agreement/) | Venture Capital | 69/69 | Perfect generative drafting |
| [`extract-transaction-entity-details/`](examples/extract-transaction-entity-details/) | Sanctions | 80/85 | Near-miss: entities found, specifics missed |
| [`extract-lien-and-debt-information/`](examples/extract-lien-and-debt-information/) | Banking/UCC | 54/59 | Near-miss: facts extracted, cross-references missed |
| [`compare-merger-remedies/`](examples/compare-merger-remedies/) | Antitrust | 56/61 | Near-miss: complex multi-jurisdiction comparison |
| [`datadog-strategic-analysis/`](examples/datadog-strategic-analysis/) | Finance/SEC | N/A | Domain-agnostic proof: 7 10-K filings, 12,657-word investment memo ([comparison](examples/datadog-strategic-analysis/COMPARISON.md)) |

Browse any task's `swarm/blackboard_iter_*.json` files to trace the full reasoning evolution. The complete outputs for all 2,010 tasks are available in [GitHub Releases](https://github.com/dl1683/irys-stateful-swarms/releases).

## Why stateful swarms matter

The AI industry has a statefulness problem. Every major AI system today — coding agents, research assistants, document analysts — treats each interaction as an isolated event. The model reasons, produces output, and forgets. The next interaction starts from zero. Context windows get compacted, destroying details that seemed unimportant but become critical later. Session boundaries erase everything.

**This is not a minor inconvenience. It is a fundamental architectural failure.** A system that forgets what it learned yesterday will always pay the full cost of understanding today. It will always re-read documents it has already analyzed. It will always re-discover entities it has already identified. It will always re-derive conclusions it has already reached.

Stateful swarms break this cycle. The blackboard is not a temporary scratchpad — it is persistent, structured, typed, provenance-tracked analytical state that survives across sessions and accumulates over time. The cost of understanding a document set is paid once. Every subsequent interaction builds on what came before.

irys-stateful-swarms achieves its benchmark results using **only API calls** to standard language models — no fine-tuning, no custom embeddings, no latent space manipulation. The entire system is coordination logic and structured state management. We're open-sourcing this to demonstrate that the stateful swarm paradigm works, and to invite the community to build on it.

### What the benchmark deliberately leaves out

irys-stateful-swarms uses only vanilla API calls and builds its entire understanding from scratch for every task. **This is intentional** — it's the only fair way to benchmark a stateful system.

In practice, [Irys](https://www.irys.ai), our unified legal AI platform, maintains persistent document indexes, entity graphs, knowledge graphs, and matter-level context across sessions. When an attorney asks a follow-up about the same credit agreement, the system doesn't re-extract 2,400 entries — they're already there. When a new document arrives on an existing deal, the system reconciles it against what it already knows, flags contradictions, and updates its understanding incrementally. Irys also brings citation verification against 50M+ court opinions, drafting with tracked changes, and matter management that organizes all documents, notes, and analysis in one workspace.

The benchmark strips all of that away. Every task starts with an empty blackboard: no prior knowledge, document memory, knowledge graphs, citation databases, or matter context. The run therefore includes work that a persistent stateful system could reuse.

We made this choice because persistent state would be an unfair advantage on a benchmark: the system would be learning from the benchmark itself. The benchmark consequently does not measure the reuse benefits of a persistent deployment.

### Complementary systems we've built

We've open-sourced several systems that address the layers surrounding stateful swarm coordination. Each tackles a different part of the full stack — from how individual models reason, to how information is represented and retrieved, to how analytical state persists across sessions.

---

**[Latent Space Reasoning](https://github.com/dl1683/Latent-Space-Reasoning)** — Can a frozen language model reason better without any training? This project demonstrates that the answer is yes, by controlling the model's latent trajectories at inference time through diffusion denoise repair.

The core mechanism uses diffusion denoise trajectories as an editable reasoning substrate. Rather than fine-tuning weights, the system extracts compact semantic anchors from intermediate model states, diagnoses where information flow breaks down using a decomposed four-head selector (evaluating spend, source quality, promotion value, and retention safety independently), and applies masked-span repair at selected denoise steps. The repair-spend gate makes surgical decisions about *where* in the latent space to intervene — not a single repairability signal, but four independent evaluation heads.

Results across multiple domains and model families: **+19.6pp arithmetic improvement** on Qwen3-4B (32% to 51.6%) using just 2-token random prefix perturbation with zero training. The frontier diffusion repair mode achieves score 0.531 vs 0.413 greedy baseline (+28.8%) on planning tasks. Oracle coverage reaches 100% across 25 diverse reasoning tasks from just 10 two-token directions. On legal reasoning across 12 complex scenarios, oracle perturbation beats the baseline on 11/12 tasks (92%) with average +1.6 points on a 10-point scale. Validated across Qwen3 (0.6B, 1.7B, 4B, 8B), DeepSeek-1.5B, phi-2, and LLaDA-MoE (7B), with architecture-dependent mechanisms: 4B models show convergence aid, 8B models show both computation and convergence improvement.

For stateful swarms, this suggests a research path for improving worker reasoning without changing the coordination architecture.

---

**[Fractal Embeddings](https://github.com/dl1683/moonshot-fractal-embeddings)** — Standard dense retrievers treat all embedding dimensions equally. But semantic information is inherently hierarchical — truncating to 64 dimensions should preserve domain-level intent, while 384 dimensions capture fine-grained distinctions. Fractal Embeddings align the dimensional structure of embeddings to this semantic zoom.

The approach structures embeddings so that prefix lengths correspond to semantic coarseness: 64 dims capture domain (L0), 128 dims capture category (L1), full 384 dims capture fine-grained intent. Unlike Matryoshka Representation Learning (MRL), which minimizes accuracy loss at each truncation level, Fractal Embeddings trains with prefix-stratified supervision — L0 labels for the first 64 dimensions, L0+L1 labels for 128, full labels for 384. A learnable fractal head embeds task representations through intermediate projections aligned to dimensional boundaries. The key empirical finding: **class separation ratio (inter-class / intra-class distance) predicts representation quality with R²=0.554**, dominating both alignment and uniformity metrics (R²<0.07 each). This was validated causally through rank-constrained perturbation surgery — not just correlation, but proven causal influence of geometric hierarchy on quality.

We proved that correct geometric hierarchy **causally improves** embedding quality, while wrong hierarchy actively hurts. Validated across 6 NLP encoder architectures (BERT, DeBERTa, E5, BGE-base/large, MiniLM), vision models (ViT-Large on CIFAR-10, ResNet-50 on CIFAR-100), and biological neural systems (32 mouse V1 Neuropixels sessions — 30/32 PASS, mean r=0.736). Cross-dataset extension covers 14 datasets including DBpedia, AG News, Yahoo Answers, and GoEmotions.

For document analysis, this is the difference between an embedding that treats a contract clause the same regardless of context, and one that natively understands that a SOFR floor clause lives inside a credit agreement section, inside a banking transaction. Better hierarchical retrieval means better cross-reference detection — exactly where stateful swarms' near-misses happen.

---

**[CTI Universal Law](https://github.com/dl1683/moonshot-cti-universal-law)** — Why does representation quality follow particular patterns across architectures, datasets, and even biological neural systems? This project derives the answer from first principles using extreme value theory, producing a universal law that is *proven, not fitted*.

The functional form is derived from Gumbel race competition among K classes before any constants are estimated: `logit(q_norm) = α × κ_nearest − β × log(K−1) + C_dataset`, where κ_nearest is the nearest-class separation signal-to-noise ratio. Leave-one-architecture-out cross-validation across 192 data points (12 NLP architectures × 4 datasets) yields **α=1.477 with coefficient of variation 2.3% and R²=0.955**. The law exhibits three-level universality: (1) functional form holds across all modalities, (2) α is universal within architecture families (NLP decoders CV=2.3%), (3) C_dataset varies by task.

Causal evidence goes beyond correlation: confusion-matrix causal prediction achieves r=0.842 with 93-100% sign accuracy across 182 test points (p<10⁻³⁵). Pre-registered RWKV-4 boundary test confirmed α=2.887 within the predicted interval. Blind out-of-distribution validation on unseen architectures and datasets yields r=0.817 (p=0.013). Cross-model ranking across 9 architectures achieves Spearman ρ=0.833 (p=0.005), meaning κ values predict MAP@10 ranking without running retrieval. The law generalizes to biological systems: 32 mouse V1 Neuropixels sessions show 30/32 PASS with mean r=0.736, validated across 5 cortical areas in 30 mice with ≥87% consistency per area.

For multi-model systems, representation diagnostics offer a general way to compare candidate components without exposing or hard-coding a deployment configuration.

---

**[MapU](https://github.com/dl1683/MapU)** *(active development — architecture is being reworked)* — Persistent, provenance-backed knowledge memory for agentic systems. Every assertion MapU stores carries source attribution, confidence, temporal validity, and conflict state. When you query it, you don't just get an answer — you get `next_steps` guidance: actionable investigation targets derived from identified gaps in the knowledge base.

MapU provides 14 MCP tools for agent integration (bootstrap, ingest, query, investigate, lookup entities, list gaps, track activity, record sessions, handoff context), plus REST API, CLI, and Python package surfaces, all backed by PostgreSQL with pgvector. The system handles document updates through explicit conflict-aware supersession — when evidence changes, MapU doesn't silently overwrite; it tracks the change ordering and can roll back. The mandatory resumption protocol (`mapu resume` first, read gaps and recent activity, execute priority actions) ensures agents pick up where they left off without re-reading everything.

This is the persistence layer that makes stateful swarms practical in production. In a benchmark, the system must start from zero on every task — that's fair evaluation. But in practice, a lawyer working a deal doesn't start from scratch every morning. Within the same matter, the system persists its document understanding, entity graphs, and analytical findings across sessions. Instead of re-reading a 200-page credit agreement every time a user asks a follow-up question, grounded entries remain available to query, extend, and refine. Background maintenance reconciles new documents against existing state, flags contradictions, and updates entity relationships. **Statefulness is the difference between an AI that assists and an AI that understands.**

---

A production stateful swarm combines coordination (irys-stateful-swarms) with improved reasoning (Latent Space), better representations (Fractal Embeddings, CTI), and persistent matter-level memory (MapU). Each layer reinforces the others — better reasoning produces higher-quality state, better representations improve cross-reference detection within that state, and persistent memory ensures none of it is ever discarded. We're releasing each piece independently so the community can explore these directions.

## Blackboard MCP: use stateful reasoning in Claude Code and Codex

The blackboard reasoning system that powers irys-stateful-swarms is available as a standalone MCP server at [`packages/blackboard-mcp/`](packages/blackboard-mcp/). It gives any AI agent persistent structured reasoning — zero API calls, zero cost.

**What it does:** 14 tools for creating and managing blackboards — typed entries (observation, analysis, calculation, strategy, gap), automatic contradiction detection with confidence decay, signal tracking for open questions, convergence gating that blocks premature synthesis, cross-session persistence, and document provenance. The intelligence comes from the agent. The blackboard just provides structured state management as pure computation.

**Why it matters:** When you install this in Claude Code or Codex, your agent gains the ability to build persistent analytical state. A blackboard created during one session is discoverable in the next — the agent calls `bb_list`, finds prior work, and extends it instead of starting from scratch. Complex multi-document analysis, contradiction tracking, gap identification, and evidence-chain building all happen through the blackboard, producing auditable reasoning traces instead of black-box answers.

**Interactive exports:** Call `bb_export` to generate a self-contained HTML file with an interactive knowledge graph — force-directed layout, cluster detection, cross-document synthesis, evidence chains, confidence gauges, and a structured briefing. The export is a single file with zero dependencies that works offline. Share it with anyone; open it in any browser. Mobile responsive with touch navigation.

**Graph overview with briefing panel** — 146 findings from a Datadog financial analysis, with trust audit and source-attributed conclusions:

![Graph overview with briefing panel](packages/blackboard-mcp/screenshots/graph-overview.png)

**Node detail panel** — click any node to see full content, confidence, provenance, tags, and connected findings with relationship types:

![Node detail panel](packages/blackboard-mcp/screenshots/node-detail.png)

**Source fragility analysis** — the skeptic lens shows what breaks if you remove each source document, with reading progress tracking:

![Source fragility analysis](packages/blackboard-mcp/screenshots/source-analysis.png)

### Install

Blackboard MCP is published on npm as [`@iqidis/blackboard-mcp`](https://www.npmjs.com/package/@iqidis/blackboard-mcp), no clone or build required.

**Claude Code** (one-liner, from your project root):
```bash
claude mcp add blackboard -- npx @iqidis/blackboard-mcp
```

Or add directly to `.mcp.json`:
```json
{
  "mcpServers": {
    "blackboard": {
      "type": "stdio",
      "command": "npx",
      "args": ["@iqidis/blackboard-mcp"]
    }
  }
}
```

**Codex CLI** (add to `~/.codex/config.toml`):
```toml
[mcp_servers.blackboard]
command = "npx"
args = ["-y", "@iqidis/blackboard-mcp"]
```

**From source** (if you're working inside this repo directly):
```bash
cd packages/blackboard-mcp && npm install && npm run build
node dist/index.js
```

A packaged Claude Code plugin is also available at [`packages/blackboard-mcp/claude-plugin/`](packages/blackboard-mcp/claude-plugin/) for marketplace-style installs.

Once configured, the agent automatically uses the blackboard for complex analysis tasks and skips it for simple questions. See [`packages/blackboard-mcp/SETUP.md`](packages/blackboard-mcp/SETUP.md) for details.

## Beyond legal: the stateful swarm paradigm is domain-agnostic

irys-stateful-swarms was validated on the Harvey LAB benchmark, but the underlying paradigm — task decomposition, persistent blackboard state-building, multi-agent coordination with typed provenance — is not legal-specific. Any domain where professionals build understanding over time through repeated analysis of complex documents is a domain where stateful swarms outperform stateless approaches: financial due diligence, regulatory compliance, medical research synthesis, insurance underwriting, patent analysis, investigative journalism.

**We've already proven this across two very different domains.**

### SWE-bench Verified: preliminary scaffold evaluation

A preliminary [SWE-bench Verified](https://www.swebench.com/) run explored Blackboard MCP under a thin wrapper. Patch-production gaps and environment failures left incomplete coverage, so the current artifacts are not suitable for a headline product comparison. The working materials remain under `benchmarks/swebench/` and should be rerun under a publication-safe manifest that separates attempted, evaluable, and resolved instances before any result is cited.

### Datadog 10-K strategic analysis

With zero code changes, we pointed irys-stateful-swarms at seven Datadog 10-K annual filings (FY2019–FY2025) and asked for a strategic-priority analysis. The system produced a 12,657-word investment memo tracing product strategy, go-to-market changes, competitive positioning, financial trajectory, and risk factors across the filings. This is qualitative evidence of domain transfer, not a controlled benchmark result.

A follow-on comparison evaluated blackboard reuse across several configurations. Exact routing, configuration-tied performance, and internal operating data are omitted under the repository's publication policy. The bounded public observation is that persistent blackboard state let later queries reuse structured evidence accumulated during earlier queries. The underlying comparison artifacts require a separate publication audit before they should be cited.

---

We're actively adapting the system to run across multiple benchmarks spanning different fields of knowledge work. The swarm framework is being generalized with benchmark adapters so we can evaluate against diverse task types and domains.

We're a small team, and benchmark runs at scale take real compute and time. We'll be releasing results as we complete them. If you're working on benchmarks for knowledge-intensive tasks and would be interested in partnering or having irys-stateful-swarms evaluated on your benchmark, reach out at [devansh@iqidis.ai](mailto:devansh@iqidis.ai).

## Installation

```bash
pip install -e .
```

Requires Python 3.12+.

### Environment variables

```bash
# Required: credentials for at least one supported provider
GEMINI_API_KEY=...
GEMINI_API_KEYS=k1,k2,k3       # Multiple keys are optional
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Optional: deployment-specific model overrides
# Do not commit production values.
SWARM_WORKER_MODEL=<provider-model-id>
SWARM_SYNTHESIS_MODEL=<provider-model-id>
SWARM_REVIEWER_MODEL=<provider-model-id>
```

## Usage

### Run a single task

```bash
python -m src.cli run <task_directory> --output-dir results/
```

The task directory should contain:
- A `task.json` with an `instructions` field (or an `instruction.md` file)
- Source documents in a `source_documents/` subdirectory (or alongside task.json)

Supported document formats: PDF, DOCX, XLSX, PPTX, TXT, MD, JSON, EML.

### Run from a manifest

```bash
python -m src.cli run-manifest <manifest.json> --output-dir results/
```

### Score outputs (requires Harvey LAB)

```bash
python -m src.cli score <results_dir> --bench-root /path/to/harvey-labs
```

## Contributing

Contributions welcome! Areas of interest:
- New worker strategies (extraction patterns, cross-document reasoning, gap detection)
- Alternative blackboard schemas and entry types
- Benchmark adapters for non-legal domains
- Performance optimizations (token efficiency, parallelism, scheduling)
- New document format support
- Visualization and debugging tools for blackboard evolution

The point of open sourcing is to push the boundaries of what stateful swarms can do. Don't be scared to explore unconventional ideas.

### Open Research Questions

These are the hard problems we're actively working on. If you make progress on any of them, we want to hear about it.

1. **Cross-document entity resolution without LLM calls.** The blackboard contains near-duplicate entities ("Zenith Petrochem" vs "Zenith Petrochemical") that workers extract but don't reconcile. Can deterministic string similarity, edit distance, or lightweight embedding comparisons close this gap without burning tokens?

2. **Optimal convergence detection.** The supervisor currently runs a fixed number of iterations with a gap-based convergence check. Is there a better signal for "the blackboard has stabilized" — information-theoretic, graph-structural, or confidence-distribution-based?

3. **Blackboard compression for synthesis.** The synthesis phase receives thousands of entries but context windows are finite. What's the best strategy for selecting, ranking, or clustering entries to maximize information density in the synthesis prompt without losing critical details?

4. **Multi-benchmark generalization.** We've proven domain transfer on SEC filings. What breaks when you run the system on medical research papers, patent filings, or insurance underwriting documents? Where does the architecture need domain-specific adaptation vs. where does it generalize cleanly?


### Monthly Bounty Program ($2,000/month)

[Iqidis](https://iqidis.ai) sponsors a monthly bounty pool for the top 10 contributors:

| Rank | Bounty |
|------|--------|
| 1st  | $500   |
| 2nd  | $350   |
| 3rd  | $275   |
| 4th  | $200   |
| 5th  | $175   |
| 6th  | $150   |
| 7th  | $125   |
| 8th  | $100   |
| 9th  | $75    |
| 10th | $50    |

**Additional perks:**
- All Top 10 contributors listed in this README
- Active contributors offered interviews at [Iqidis](https://iqidis.ai) and access to our network of **1.5M+ members** including engineers, managers, and builders from Google, Nvidia, OpenAI, Anthropic, Meta AI, and other top AI organizations

Bounties given out monthly on the 15th.

## Sources

- Harvey LAB repository: <https://github.com/harveyai/harvey-labs>
- Harvey initial LAB results: <https://www.harvey.ai/blog/legal-agent-benchmark-initial-results>

## License

MIT — see [LICENSE](LICENSE).
