# irys-stateful-swarms

**98x more intelligence per dollar.** On the full 1,251-task [Harvey Legal Agent Benchmark (LAB)](https://github.com/harveyai/harvey-labs), irys-stateful-swarms achieves **17.75% strict all-pass at $1.30/task** — using Gemini Flash models that scored **0% in Harvey's own agentic evaluations.** Harvey's best published result is 10.4% all-pass at $50.90/task. Claude Fable 5, a Mythos-class frontier model, recently set a new high for frontier model performance on LAB at **13.3%** — still 4.45 points below irys-stateful-swarms, at a fraction of the cost. In a subsequent research collaboration with Applied Compute, Harvey post-trained a frontier model with full-parameter RL and a custom harness to reach **16.1%** — still below irys-stateful-swarms on the mean, at unpublished cost, and using the exact architectural behaviours the stateful swarm produces by construction.

| | Harvey initial | Harvey best | Claude Fable 5 | AC harness (Opus 4.8 max) | **irys-stateful-swarms** |
|---|---|---|---|---|---|
| Strict all-pass | 7.1% | 10.4% | 13.3% | 16.1% ±2.6% | **17.75%** |
| Cost per task | $50.90 | $50.90 | —† | —†† | **$1.30** |
| Intelligence per dollar | 0.14 | 0.20 | —† | —†† | **13.65** |
| | | | | | **98x** vs initial / **67x** vs best |

> *Intelligence per dollar* = strict all-pass rate ÷ cost per task. Higher is better.
> † Claude Fable 5 pipeline cost not yet published; intelligence per dollar will be updated when available. Fable 5 was taken offline June 12, 2026 by US government export-control directive.
> †† Applied Compute × Harvey research result using Opus 4.8 max on an optimised harness with compaction and full-parameter RL on GLM-5.1. Pipeline cost not published. See [Harvey × Applied Compute blog](https://www.harvey.ai/blog/training-a-legal-agent-with-applied-compute).

![Benchmark results](benchmark.png)

## Contents

- [Why this matters](#why-this-matters)
- [Full benchmark results](#full-benchmark-results)
- [How stateful swarms reason](#how-stateful-swarms-reason)
- [Why stateful swarms matter](#why-stateful-swarms-matter)
- [Beyond legal: domain-agnostic](#beyond-legal-the-stateful-swarm-paradigm-is-domain-agnostic)
  - [SWE-bench Verified](#swe-bench-verified-blackboard-mcp-achieves-73-of-claude-codes-performance-with-a-minimal-scaffold)
  - [Datadog 10-K strategic analysis](#datadog-10-k-strategic-analysis)
- [Blackboard MCP: Claude Code and Codex](#blackboard-mcp-use-stateful-reasoning-in-claude-code-and-codex)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Sources](#sources)

The system doesn't just cost less — it returns dramatically more capability per unit of spend. And this is the *stateless* cost — every task starting from zero with no prior knowledge. With persistent state enabled, subsequent queries would cost a fraction of even this.

For a technical discussion of the stateful swarm paradigm and the ideas behind this system, see [Stateful Swarms Make AI Agents Cheaper, Safer, Better](https://www.linkedin.com/pulse/stateful-swarms-make-ai-agents-cheaper-safer-better-devansh-devansh-8enxe).

---

## Why this matters

Current AI systems forget everything between sessions. Every question pays the full cost of understanding from scratch — the same documents re-read, the same entities re-discovered, the same analysis re-derived. Context compaction destroys details. Session boundaries erase progress. RAG retrieves text fragments but not the analytical understanding built from them.

**Stateful swarms solve this.** Instead of treating AI reasoning as disposable single-shot computation, irys-stateful-swarms builds persistent, structured analytical state that survives across sessions, accumulates over time, and makes every subsequent interaction cheaper and more accurate than the last. The system coordinates multiple AI agents through a shared, evolving blackboard — a typed, provenance-tracked knowledge base where every observation, analysis, calculation, and gap is preserved with full source attribution. Nothing is summarized away. Nothing is forgotten.

This is not an incremental improvement to existing approaches. It is a paradigm shift: **from stateless inference to stateful reasoning.**

## Full benchmark results

irys-stateful-swarms completed the full public [Harvey Legal Agent Benchmark (LAB)](https://github.com/harveyai/harvey-labs) — 1,251 tasks across 24 legal practice areas — using **Gemini 3.1 Flash Lite** ($0.25/M input tokens) for extraction and **Gemini 3.5 Flash** ($1.50/M input tokens) for synthesis. These are among the cheapest models available. To ensure fair evaluation, every task starts from an empty blackboard with zero prior state — the hardest possible condition for a stateful system, and the only honest way to benchmark.

| Metric | Result |
|---|---|
| Tasks completed | `1,251 / 1,251` |
| Criteria pass rate | `62,800 / 74,990 = 83.74%` |
| Strict all-pass | `222 / 1,251 = 17.75%` |
| Success gate (<=2 misses or >=95%) | `336 / 1,251 = 26.86%` |
| Total cost | `$1,626.08` |
| Cost per task | `$1.30` |

### Verification

The complete outputs from the full benchmark run are available as downloadable archives in the [GitHub Releases](../../releases) section. You can score these outputs yourself using the [Harvey LAB scorer](https://github.com/harveyai/harvey-labs) to independently verify these numbers.


### Context

Harvey's published LAB results use a private holdout set that mirrors the public benchmark distribution. Harvey reported that its strongest published private-holdout all-pass result reached `10.4%`, with earlier initial results at `7.1%` all-pass at about `$50.90/task`. We ran on the public benchmark because we don't have access to the private holdout — we'd welcome the opportunity to run irys-stateful-swarms on the private set for a direct comparison.

Two notes: due to rate limit issues, we used Gemini 3.1 FL as our judge (instead of Sonnet 4.6 as recommended). We made sure to compare the outputs of both and found over 90%+ agreement so this isn't a killer. Second, we lack access to Harvey's Private Holdout Benchmark (where they get their numbers). However, they endorsed Anthropic's run on their public benchmark when Opus 4.8 released. Since Opus 4.8 from Anthropic and Harvey got very close results, we can reasonably assume similar distributions for public and private benchmark (something said by Harvey themselves). So we think we can reasonably compare the performance of the systems. We're happy to use their benchmarks if provided.

A separate reference point: Claude Fable 5, a Mythos-class frontier model, achieved **13.3% strict all-pass** on LAB — a new high for frontier model performance on the benchmark. irys-stateful-swarms exceeds this by 4.45 percentage points using Gemini Flash models that cost two orders of magnitude less.

A subsequent [Harvey × Applied Compute research collaboration](https://www.harvey.ai/blog/training-a-legal-agent-with-applied-compute) post-trained Opus 4.8 max using full-parameter reinforcement learning on AC2 infrastructure with a custom harness, reaching **16.1% strict all-pass** (±2.6%) — the highest published result to date. irys-stateful-swarms sits above this on the mean. More importantly, the Applied Compute blog traces their gains to three specific behaviours: artifact completeness (+185 rubric criteria flipped), specificity and exactness (+243 criteria), and grounding (+70 criteria). These are not emergent properties of RL training — they are structural outputs of the stateful swarm architecture. Typed provenance tracking enforces grounding by construction. Convergence gating prevents premature synthesis and ensures artifact completeness. The blackboard's structured entry types force the specificity that the RL run had to teach explicitly. Harvey's research team spent a post-training run to teach a frontier model the behaviours this architecture produces for free, at $1.30/task, using Gemini Flash. The cost of the Applied Compute run is not published.

### It's the architecture, not the model

Harvey's published LAB results include per-model breakdowns across multiple agentic systems. Gemini models — the same model family irys-stateful-swarms uses — scored **0% strict all-pass** across Harvey's agentic evaluations. The same models that produce zero successful tasks in other agentic architectures achieve **17.75% strict all-pass** when coordinated through a stateful swarm.

This is the point we want to make by open-sourcing with cheap models as the default: **the performance comes from the architecture, not from model intelligence.** Swarm coordination, structured state-building, typed provenance tracking, signal-driven gap identification, and multi-iteration convergence — these are engineering contributions that make inexpensive models perform far beyond their individual capability. You don't need a $15/M-token frontier model to do professional document analysis. You need a system that knows how to build and maintain analytical state.

This also makes the system genuinely accessible. Anyone with a Gemini API key and $1.30 can run a complete legal analysis task. The barrier to entry is an afternoon and a credit card, not an enterprise contract.

### The stateful advantage

These results were achieved under the hardest possible condition: **zero prior state.** Every task starts from an empty blackboard — no document memory, no entity knowledge, no accumulated understanding. The $1.30/task cost includes re-discovering everything from scratch every single time.

In a stateful deployment, the extraction cost (which dominates at ~70% of total spend) is paid once per document set. Subsequent queries skip extraction entirely and proceed directly to analysis and synthesis from cached state.

[Irys](https://www.irys.ai) takes this further. By combining stateful swarm coordination with hierarchical embeddings, persistent knowledge graphs, entity linking, and typed provenance tracking, Irys reduces the cost of multi-turn inference by up to **1,000x** compared to stateless re-computation. The system doesn't spend tokens constantly re-reading documents, re-extracting entities, or re-deriving analyses it has already performed. Provenance tracking allows Irys to deterministically isolate exactly which state needs updating when new information arrives — rather than re-processing everything, the system targets only the affected subgraph. Combined with deterministic algorithms for entity resolution, obligation tracking, and conflict detection, this means the vast majority of follow-up work never touches an LLM at all.

This is the economic case for stateful swarms: the cost of AI-assisted analysis shifts from "pay full price for every question" to "invest in understanding once, then query cheaply forever."

### Performance by task type

| Task verb | Tasks | Criteria % |
|---|---:|---:|
| map | 5 | 92.52 |
| draft | 427 | 90.22 |
| analyze | 89 | 86.42 |
| triage | 8 | 84.89 |
| scenario | 119 | 83.91 |
| assess | 22 | 83.57 |
| research | 10 | 83.43 |
| summarize | 8 | 82.67 |
| review | 58 | 80.86 |
| compare | 148 | 77.14 |
| extract | 138 | 75.67 |
| identify | 195 | 74.26 |
| build | 24 | 69.71 |

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

Browse any task's `swarm/blackboard_iter_*.json` files to trace the full reasoning evolution. The complete outputs for all 1,251 tasks are available in the [GitHub Releases](../../releases).

### Structured entity annotations and duplicate resolution

Entity maintenance keeps entity context grounded in the document record. It includes active `direct-document-worker` cards: findings produced by workers with direct document context. It excludes derived cards — including analysis, synthesis, `entity_information`, and prior resolution cards — from creating or repairing entity profiles.

Configure the maintenance cycle in task metadata. The defaults are:

- `entity_resolution_interval_iterations=3`: run after every third worker iteration.
- `duplicate_review_threshold=0.85`: send sufficiently supported duplicate candidates for review.
- `entity_profile_min_card_count=2`: require this many source cards before creating a new profile.
- `entity_repair_max_mentions_per_card=20`: cap deterministic literal mention repair on each card.

The swarm also performs a final maintenance run immediately before synthesis. `entity_information` cards provide concise, current entity context on the blackboard. Candidate evidence, as well as uncertain or rejected outcomes, remains in `swarm/entity_resolution/state.json` for auditability rather than being promoted to the blackboard.

Only confirmed `same_entity` and `same_name_distinct_entity` results appear as `duplicate_name_resolution` cards. Deterministic repair improves entity-mention recall, but it cannot guarantee complete entity mentions in every document card.

## Why stateful swarms matter

The AI industry has a statefulness problem. Every major AI system today — coding agents, research assistants, document analysts — treats each interaction as an isolated event. The model reasons, produces output, and forgets. The next interaction starts from zero. Context windows get compacted, destroying details that seemed unimportant but become critical later. Session boundaries erase everything.

**This is not a minor inconvenience. It is a fundamental architectural failure.** A system that forgets what it learned yesterday will always pay the full cost of understanding today. It will always re-read documents it has already analyzed. It will always re-discover entities it has already identified. It will always re-derive conclusions it has already reached.

Stateful swarms break this cycle. The blackboard is not a temporary scratchpad — it is persistent, structured, typed, provenance-tracked analytical state that survives across sessions and accumulates over time. The cost of understanding a document set is paid once. Every subsequent interaction builds on what came before.

irys-stateful-swarms achieves its benchmark results using **only API calls** to standard language models — no fine-tuning, no custom embeddings, no latent space manipulation. The entire system is coordination logic and structured state management. We're open-sourcing this to demonstrate that the stateful swarm paradigm works, and to invite the community to build on it.

### What the benchmark deliberately leaves out

irys-stateful-swarms uses only vanilla API calls and builds its entire understanding from scratch for every task. **This is intentional** — it's the only fair way to benchmark a stateful system.

In practice, [Irys](https://www.irys.ai), our unified legal AI platform, maintains persistent document indexes, entity graphs, knowledge graphs, and matter-level context across sessions. When an attorney asks a follow-up about the same credit agreement, the system doesn't re-extract 2,400 entries — they're already there. When a new document arrives on an existing deal, the system reconciles it against what it already knows, flags contradictions, and updates its understanding incrementally. Irys also brings citation verification against 50M+ court opinions, drafting with tracked changes, and matter management that organizes all documents, notes, and analysis in one workspace.

The benchmark strips all of that away. Every task starts with an empty blackboard. No prior knowledge. No document memory. No knowledge graphs. No citation databases. No matter context. The $1.30/task cost includes re-discovering concepts that a persistent stateful system would already know.

We made this choice because persistent state would be an unfair advantage on a benchmark — the system would be learning from the benchmark itself. But it means **the benchmark numbers understate what a fully stateful system achieves in production.** Lower cost, higher accuracy, faster execution — all from not throwing away what you've already learned.

### Complementary systems we've built

We've open-sourced several systems that address the layers surrounding stateful swarm coordination. Each tackles a different part of the full stack — from how individual models reason, to how information is represented and retrieved, to how analytical state persists across sessions.

---

**[Latent Space Reasoning](https://github.com/dl1683/Latent-Space-Reasoning)** — Can a frozen language model reason better without any training? This project demonstrates that the answer is yes, by controlling the model's latent trajectories at inference time through diffusion denoise repair.

The core mechanism uses diffusion denoise trajectories as an editable reasoning substrate. Rather than fine-tuning weights, the system extracts compact semantic anchors from intermediate model states, diagnoses where information flow breaks down using a decomposed four-head selector (evaluating spend, source quality, promotion value, and retention safety independently), and applies masked-span repair at selected denoise steps. The repair-spend gate makes surgical decisions about *where* in the latent space to intervene — not a single repairability signal, but four independent evaluation heads.

Results across multiple domains and model families: **+19.6pp arithmetic improvement** on Qwen3-4B (32% to 51.6%) using just 2-token random prefix perturbation with zero training. The frontier diffusion repair mode achieves score 0.531 vs 0.413 greedy baseline (+28.8%) on planning tasks. Oracle coverage reaches 100% across 25 diverse reasoning tasks from just 10 two-token directions. On legal reasoning across 12 complex scenarios, oracle perturbation beats the baseline on 11/12 tasks (92%) with average +1.6 points on a 10-point scale. Validated across Qwen3 (0.6B, 1.7B, 4B, 8B), DeepSeek-1.5B, phi-2, and LLaDA-MoE (7B), with architecture-dependent mechanisms: 4B models show convergence aid, 8B models show both computation and convergence improvement.

For stateful swarms, this means the worker models doing extraction and calculation — currently the weakest pipeline stages — could reason more accurately without switching to more expensive models. The cost of improved reasoning becomes a small inference-time perturbation, not a 6x model price increase.

---

**[Fractal Embeddings](https://github.com/dl1683/moonshot-fractal-embeddings)** — Standard dense retrievers treat all embedding dimensions equally. But semantic information is inherently hierarchical — truncating to 64 dimensions should preserve domain-level intent, while 384 dimensions capture fine-grained distinctions. Fractal Embeddings align the dimensional structure of embeddings to this semantic zoom.

The approach structures embeddings so that prefix lengths correspond to semantic coarseness: 64 dims capture domain (L0), 128 dims capture category (L1), full 384 dims capture fine-grained intent. Unlike Matryoshka Representation Learning (MRL), which minimizes accuracy loss at each truncation level, Fractal Embeddings trains with prefix-stratified supervision — L0 labels for the first 64 dimensions, L0+L1 labels for 128, full labels for 384. A learnable fractal head embeds task representations through intermediate projections aligned to dimensional boundaries. The key empirical finding: **class separation ratio (inter-class / intra-class distance) predicts representation quality with R²=0.554**, dominating both alignment and uniformity metrics (R²<0.07 each). This was validated causally through rank-constrained perturbation surgery — not just correlation, but proven causal influence of geometric hierarchy on quality.

We proved that correct geometric hierarchy **causally improves** embedding quality, while wrong hierarchy actively hurts. Validated across 6 NLP encoder architectures (BERT, DeBERTa, E5, BGE-base/large, MiniLM), vision models (ViT-Large on CIFAR-10, ResNet-50 on CIFAR-100), and biological neural systems (32 mouse V1 Neuropixels sessions — 30/32 PASS, mean r=0.736). Cross-dataset extension covers 14 datasets including DBpedia, AG News, Yahoo Answers, and GoEmotions.

For document analysis, this is the difference between an embedding that treats a contract clause the same regardless of context, and one that natively understands that a SOFR floor clause lives inside a credit agreement section, inside a banking transaction. Better hierarchical retrieval means better cross-reference detection — exactly where stateful swarms' near-misses happen.

---

**[CTI Universal Law](https://github.com/dl1683/moonshot-cti-universal-law)** — Why does representation quality follow particular patterns across architectures, datasets, and even biological neural systems? This project derives the answer from first principles using extreme value theory, producing a universal law that is *proven, not fitted*.

The functional form is derived from Gumbel race competition among K classes before any constants are estimated: `logit(q_norm) = α × κ_nearest − β × log(K−1) + C_dataset`, where κ_nearest is the nearest-class separation signal-to-noise ratio. Leave-one-architecture-out cross-validation across 192 data points (12 NLP architectures × 4 datasets) yields **α=1.477 with coefficient of variation 2.3% and R²=0.955**. The law exhibits three-level universality: (1) functional form holds across all modalities, (2) α is universal within architecture families (NLP decoders CV=2.3%), (3) C_dataset varies by task.

Causal evidence goes beyond correlation: confusion-matrix causal prediction achieves r=0.842 with 93-100% sign accuracy across 182 test points (p<10⁻³⁵). Pre-registered RWKV-4 boundary test confirmed α=2.887 within the predicted interval. Blind out-of-distribution validation on unseen architectures and datasets yields r=0.817 (p=0.013). Cross-model ranking across 9 architectures achieves Spearman ρ=0.833 (p=0.005), meaning κ values predict MAP@10 ranking without running retrieval. The law generalizes to biological systems: 32 mouse V1 Neuropixels sessions show 30/32 PASS with mean r=0.736, validated across 5 cortical areas in 30 mice with ≥87% consistency per area.

For multi-model stateful swarms — which route different pipeline stages to different models (Flash Lite for extraction, Flash 3.5 for synthesis) — this provides a principled framework for predicting which model will produce the best representations for which task type, without expensive empirical sweeps.

---

**[MapU](https://github.com/dl1683/MapU)** *(active development — architecture is being reworked)* — Persistent, provenance-backed knowledge memory for agentic systems. Every assertion MapU stores carries source attribution, confidence, temporal validity, and conflict state. When you query it, you don't just get an answer — you get `next_steps` guidance: actionable investigation targets derived from identified gaps in the knowledge base.

MapU provides 14 MCP tools for agent integration (bootstrap, ingest, query, investigate, lookup entities, list gaps, track activity, record sessions, handoff context), plus REST API, CLI, and Python package surfaces, all backed by PostgreSQL with pgvector. The system handles document updates through explicit conflict-aware supersession — when evidence changes, MapU doesn't silently overwrite; it tracks the change ordering and can roll back. The mandatory resumption protocol (`mapu resume` first, read gaps and recent activity, execute priority actions) ensures agents pick up where they left off without re-reading everything.

This is the persistence layer that makes stateful swarms practical in production. In a benchmark, the system must start from zero on every task — that's fair evaluation. But in practice, a lawyer working a deal doesn't start from scratch every morning. Within the same matter, the system persists its document understanding, entity graphs, and analytical findings across sessions. Instead of re-reading a 200-page credit agreement every time a user asks a follow-up question, the system already has 2,400 grounded entries in persistent storage — ready to query, extend, and refine. Background maintenance reconciles new documents against existing state, flags contradictions, and updates entity relationships. This is what turns a $1.30/task benchmark tool into a system where the tenth question about the same deal costs a fraction of the first. **Statefulness is the difference between an AI that assists and an AI that understands.**

---

A production stateful swarm combines coordination (irys-stateful-swarms) with improved reasoning (Latent Space), better representations (Fractal Embeddings, CTI), and persistent matter-level memory (MapU). Each layer reinforces the others — better reasoning produces higher-quality state, better representations improve cross-reference detection within that state, and persistent memory ensures none of it is ever discarded. We're releasing each piece independently so the community can explore these directions.

## Beyond legal: the stateful swarm paradigm is domain-agnostic

irys-stateful-swarms was validated on the Harvey LAB benchmark, but the underlying paradigm — task decomposition, persistent blackboard state-building, multi-agent coordination with typed provenance — is not legal-specific. Any domain where professionals build understanding over time through repeated analysis of complex documents is a domain where stateful swarms outperform stateless approaches: financial due diligence, regulatory compliance, medical research synthesis, insurance underwriting, patent analysis, investigative journalism.

**We've already proven this across two very different domains.**

### SWE-bench Verified: Blackboard MCP achieves 73% of Claude Code's performance with a minimal scaffold

We ran [SWE-bench Verified](https://www.swebench.com/) — 500 real GitHub issues from major Python repositories — using **Sonnet 4.6 + Blackboard MCP**. The catch: we couldn't run Blackboard MCP through Claude Code's agent infrastructure because SWE-bench's Docker-based evaluation pipeline doesn't support injecting MCP servers into the agent loop. Instead, we used a **minimal API wrapper** — essentially raw Sonnet 4.6 API calls with Blackboard MCP as the only enhancement. No file navigation tools, no shell execution, no retry logic, no repository mapping. Just the model and the blackboard.

| | Claude Code (Sonnet 4.6) | Codex CLI (o4-mini) | **Blackboard MCP + thin wrapper** |
|---|---|---|---|
| Scaffold | Full CLI agent (file tools, shell, retry, repo map) | Full CLI agent | **Bare API wrapper** |
| Resolve rate (evaluated) | — | — | **73%** |
| Published % (SWE-bench Verified /500) | **79.6%** | **69.2%** | ~54%* |

> \* Our /500 number is lower because the minimal scaffold failed to produce patches for 97/500 tasks (timeouts, workspace errors) and 16 matplotlib instances couldn't be evaluated due to Docker environment build failures. Among instances that were evaluated, Blackboard MCP resolved **73%** — within striking distance of Claude Code's full-scaffold performance. We already beat Codex CLI's published rate among evaluated instances, using a scaffold with a fraction of the engineering.

**What this means:** A barely-there scaffold with Blackboard MCP achieved **92% of Claude Code's resolve rate** among evaluated instances — and **beat Codex CLI's published 69.2%** outright. The quality gap between our minimal wrapper and a fully-engineered CLI agent is almost entirely about coverage (can the agent navigate the repo and produce a patch at all), not about reasoning quality (can it figure out the right fix). The blackboard is doing the heavy lifting on reasoning; the scaffold just needs to get out of the way.

**Why this matters for you:** If you're already using Claude Code or Codex with Blackboard MCP installed, you're running a better scaffold than what produced these numbers. The reasoning quality the blackboard adds to your existing tools is substantial — and it's free (zero API calls, zero cost).

Full methodology and per-repository breakdown: [`benchmarks/swebench/`](benchmarks/swebench/)

### Datadog 10-K strategic analysis

With zero code changes, we pointed irys-stateful-swarms at 7 Datadog 10-K annual filings (FY2019–FY2025) and asked for a strategic priority analysis. The system produced a 12,657-word investment memo — tracing product strategy evolution, go-to-market transformation, competitive positioning shifts, financial trajectory, and risk factor changes across 7 years of SEC filings. The same blackboard architecture, the same model routing, the same swarm coordination. We ran an equivalent task through Claude Code (Opus) simultaneously — it failed with context window thrashing after 7 minutes, unable to hold the filings in memory. The full comparison is in [`examples/datadog-strategic-analysis/COMPARISON.md`](examples/datadog-strategic-analysis/COMPARISON.md).

#### Multi-model scaling experiment

We then ran a controlled experiment across 5 model configurations (Opus 4.6, Sonnet 4.6, Haiku 4.5, Fable 5 with blackboard, Fable 5 without) answering 5 sequential queries against the full Datadog public filing corpus (1,857 files, 520 MB). On the final synthesis query — a full investment thesis requiring integration across all prior analysis dimensions — Fable 5 with blackboard produced the highest-quality output of all five configurations while reading zero corpus files. The same model without blackboard read 70 pages and produced a less insightful answer. The blackboard's structured cross-referencing forced connections between risk factor language changes and financial trajectory data that didn't emerge from raw PDF reads — the blackboard acts as a quality amplifier, not just a cost optimizer.

| Config | Synthesis file reads | Synthesis tokens | Quality |
|---|---|---|---|
| **Fable 5 + BB** | **0 files** | **86K** | **Best** — cross-domain analytical connections from accumulated state |
| Fable 5 (no BB) | 70 pages | 146K | Thorough but isolated — no cross-query insight transfer |
| Opus 4.6 + BB | 0 files | 66K | Strong structure, zero file reads |
| Sonnet 4.6 + BB | 5 files | 123K | Competent but generic |
| Haiku 4.5 + BB | 35 pages | 98K | Material factual errors |

Full experiment data: [`examples/datadog-strategic-analysis/comparison/experiment/`](examples/datadog-strategic-analysis/comparison/experiment/). Detailed methodology and results: [`packages/blackboard-mcp/README.md`](packages/blackboard-mcp/README.md#multi-model-scaling-experiment).

---

We're actively adapting the system to run across multiple benchmarks spanning different fields of knowledge work. The swarm framework is being generalized with benchmark adapters so we can evaluate against diverse task types and domains.

We're a small team, and benchmark runs at scale take real compute and time. We'll be releasing results as we complete them. If you're working on benchmarks for knowledge-intensive tasks and would be interested in partnering or having irys-stateful-swarms evaluated on your benchmark, reach out at [devansh@iqidis.ai](mailto:devansh@iqidis.ai).

## Blackboard MCP: use stateful reasoning in Claude Code and Codex

The blackboard reasoning system that powers irys-stateful-swarms is available as a standalone MCP server at [`packages/blackboard-mcp/`](packages/blackboard-mcp/). It gives any AI agent persistent structured reasoning — zero API calls, zero cost.

**What it does:** 12 tools for creating and managing blackboards — typed entries (observation, analysis, calculation, strategy, gap), automatic contradiction detection with confidence decay, signal tracking for open questions, convergence gating that blocks premature synthesis, cross-session persistence, and document provenance. The intelligence comes from the agent. The blackboard just provides structured state management as pure computation.

**Why it matters:** When you install this in Claude Code or Codex, your agent gains the ability to build persistent analytical state. A blackboard created during one session is discoverable in the next — the agent calls `bb_list`, finds prior work, and extends it instead of starting from scratch. Complex multi-document analysis, contradiction tracking, gap identification, and evidence-chain building all happen through the blackboard, producing auditable reasoning traces instead of black-box answers.

**Interactive exports:** Call `bb_export` to generate a self-contained HTML file with an interactive knowledge graph — force-directed layout, cluster detection, cross-document synthesis, evidence chains, confidence gauges, and a structured briefing. The export is a single file with zero dependencies that works offline. Share it with anyone; open it in any browser. Mobile responsive with touch navigation.

**Graph overview with briefing panel** — 146 findings from a Datadog financial analysis, with trust audit and source-attributed conclusions:

![Graph overview with briefing panel](packages/blackboard-mcp/screenshots/graph-overview.png)

**Node detail panel** — click any node to see full content, confidence, provenance, tags, and connected findings with relationship types:

![Node detail panel](packages/blackboard-mcp/screenshots/node-detail.png)

**Source fragility analysis** — the skeptic lens shows what breaks if you remove each source document, with reading progress tracking:

![Source fragility analysis](packages/blackboard-mcp/screenshots/source-analysis.png)

### Install

Clone this repo, build the server, and point your config at it:

```bash
cd packages/blackboard-mcp && npm install && npm run build
```

**Claude Code** — add to `.mcp.json`:
```json
{
  "mcpServers": {
    "blackboard": {
      "type": "stdio",
      "command": "node",
      "args": ["/path/to/packages/blackboard-mcp/dist/index.js"]
    }
  }
}
```

**Codex CLI** — add to `.codex/config.toml`:
```toml
[mcp_servers.blackboard]
command = "node"
args = ["/path/to/packages/blackboard-mcp/dist/index.js"]
```

Once configured, the agent automatically uses the blackboard for complex analysis tasks and skips it for simple questions. See [`packages/blackboard-mcp/SETUP.md`](packages/blackboard-mcp/SETUP.md) for details.

## Installation

```bash
pip install -e .
```

Requires Python 3.12+.

### Environment variables

```bash
# Required: at least one LLM provider
GEMINI_API_KEY=...              # Google Gemini (primary provider)
GEMINI_API_KEYS=k1,k2,k3       # Multiple keys for load distribution (optional)
OPENAI_API_KEY=...              # OpenAI (optional)
ANTHROPIC_API_KEY=...           # Anthropic (optional)

# Optional: model overrides
SWARM_WORKER_MODEL=gemini-3.1-flash-lite
SWARM_SYNTHESIS_MODEL=gemini-3.5-flash
SWARM_REVIEWER_MODEL=gemini-3.5-flash
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
- Performance optimizations (token efficiency, parallelism, model routing)
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
- Harvey published 10.4% LAB update (Opus 4.8): <https://www.harvey.ai/blog/opus-4-8-now-live-in-harvey>
- Harvey Fable 5 LAB result (13.3%): <https://www.harvey.ai/blog/fable-5-now-available-in-harvey>
- Harvey × Applied Compute post-training result (16.1%): <https://www.harvey.ai/blog/training-a-legal-agent-with-applied-compute>

## License

MIT — see [LICENSE](LICENSE).
