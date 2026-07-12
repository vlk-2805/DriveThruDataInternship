# Context Engineering Framework — Medical Assistant (GPU)

A GPU-backed medical-assistant demo built to illustrate the **context
engineering lifecycle** end-to-end: gathering raw context from live sources,
turning it into ranked/compressed context, constructing a prompt from it, and
generating (then evaluating) an LLM response — all inspected live through a
Gradio dashboard.

This repo contains two notebooks that implement the **same lifecycle** at two
different levels of depth:

| Notebook | Description |
|---|---|
| `ContextEngineeringFramework_GPU.ipynb` | **Core pipeline.** A linear WRITE → CHUNK → RETRIEVE → RANK → ISOLATE → COMPRESS → CONSTRUCT → EXECUTE flow. Good starting point / minimal reference implementation. |
| `ContextEngineeringFramework_GPU_configurable_fixed.ipynb` | **Full, configurable architecture.** Everything in the core pipeline, plus an MCP-style source layer, a tool registry, an agent orchestrator, knowledge/context graphs, workflow rules, memory, historical data, a reasoning stage, and an insights engine — with a drag-and-drop Gradio UI for reordering/enabling context-engineering steps per query. |

Both notebooks share the same underlying models, data sources, and core
building blocks (chunking, vector store, ranking, routing, compression,
prompt construction, LLM generation, evaluation), so this README documents
them together and calls out where the configurable version extends the core
one.

## Shared foundations

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `Qwen/Qwen2.5-3B-Instruct` (loaded via `transformers`, `device_map="auto"`, fp16) — used as a general medical-assistant chat model
- **Vector index**: FAISS (dense retrieval)
- **Sentence splitting**: `nltk` (`punkt_tab`)
- **Data sources** (`ContextCollector`):
  - OpenFDA drug label API
  - Wikipedia (`wikipedia` package)
  - DuckDuckGo search (`duckduckgo-search`)
- **Core context-engineering stages** (present in both notebooks):
  - `Chunker` — splits collected documents into sentence-grouped chunks
  - `VectorStore` — embeds and indexes chunks, retrieves top matches for a query
  - `ContextRanker` — re-scores retrieved chunks by cosine similarity to the query
  - `ContextRouter` — buckets ranked chunks into `symptoms` / `conditions` / `treatment` / `warnings` by keyword
  - `ContextCompressor` — truncates each bucket to a handful of items and joins them into compact text
  - `PromptBuilder` — assembles the final LLM prompt from the compressed context
  - `MedicalLLM` — wraps the Qwen model for chat-style generation
  - `ContextEvaluator` — scores retrieved/ranked context relevance via cosine similarity
  - `MedicalAssistant` — orchestrates the full lifecycle end to end and returns a dict with every intermediate artifact plus timings/analytics

## Notebook 1 — Core pipeline (`ContextEngineeringFramework_GPU.ipynb`)
<img width="1777" height="973" alt="image" src="https://github.com/user-attachments/assets/3a7a71b7-2194-457c-bbcd-98e6f374d67b" />

Implements the lifecycle as a fixed sequence:

```
WRITE (collect from OpenFDA/Wikipedia/DuckDuckGo)
  → CHUNK
  → RETRIEVE (FAISS)
  → RANK
  → QUALITY (evaluate)
  → ISOLATE (route into buckets)
  → COMPRESS
  → CONSTRUCT (build prompt)
  → EXECUTE (LLM generate)
```

`MedicalAssistant.run(query)` runs every stage unconditionally, in this fixed
order, timing each stage and returning the full lifecycle (raw docs, chunks,
retrieved/ranked chunks, quality scores, routed/compressed context, prompt,
response, and analytics such as compression ratio). A Gradio UI
(`lifecycle_demo`) surfaces each stage's output as its own tab.

## Notebook 2 — Configurable architecture (`ContextEngineeringFramework_GPU_configurable_fixed.ipynb`)

Implements a fuller reference architecture, extending the core pipeline with:

| Architecture-diagram box | Notebook component |
|---|---|
| Source systems | APIs/data sources called by `ContextCollector` |
| MCP | `MCPLayer` — uniform connector in front of heterogeneous source systems |
| Context Builder | `ContextCollector` (registered as MCP sources) |
| Tools Integration | `Tool` / `ToolRegistry` |
| Workflows / Rules | `WorkflowRulesEngine` — flags red-flag symptoms, decides EMERGENCY_ESCALATION vs STANDARD_DIAGNOSTIC vs GENERAL_INFO |
| Agent Orchestrator | `AgentOrchestrator` — selects/calls relevant tools and consults the Knowledge/Context Graphs and Workflow engine |
| RAG | `VectorStore` (FAISS) |
| Knowledge Graph | `KnowledgeGraph` (networkx, persisted, seeded with medical facts) |
| Context Graph | `ContextGraph` (per-session networkx graph of entities in the current query) |
| Knowledge Management System | `KnowledgeManagementSystem` — persistence facade unifying VectorStore, KnowledgeGraph, ContextGraph |
| Context Enrichment | `ContextEnrichment` — merges compressed text context with KG facts, context-graph summary, and workflow decision |
| Memory | `MemoryStore` — short-term (in-process) + long-term (persisted) conversational memory |
| Historical Data | `HistoricalDataStore` — persisted log of past queries/analytics/responses |
| Reasoning | `ReasoningEngine` — combines enriched context, memory, and historical matches into structured notes |
| Reports/Dashboards/Insights, Actions/Decisions/Risks | `InsightsEngine` — derives actions/decisions/risk flags from the response |
| Query Processor | `MedicalAssistant.run(query)` entry point |

It also replaces the fixed RANK → ISOLATE → COMPRESS sequence with a
**`ContextEngineeringPipeline`**: a user-configurable, ordered list of
enable/disable-able steps (`rank`, `dedup`, `diversity`, `prune`,
`summarize`, `truncate`) that the caller can supply per query via
`step_config`, instead of always running the same three stages in the same
order.

Full lifecycle order implemented in `MedicalAssistant.run()`:

```
WRITE → CHUNK → RETRIEVE (RAG) → RANK → QUALITY → ISOLATE → COMPRESS
  → ORCHESTRATE (Tools + Knowledge Graph + Context Graph + Workflows)
  → ENRICH → REASON (Memory + Historical Data) → CONSTRUCT → EXECUTE
  → INSIGHTS → PERSIST (Memory / Historical Data / Knowledge Management System)
```

The Gradio UI adds a **drag-and-drop pipeline builder** — available
context-engineering steps can be dragged into an ordered pipeline (replacing
the old checkbox + comma-separated textbox controls) — plus additional tabs
for the knowledge-graph visualization, orchestration trace, memory/historical
matches, reasoning notes, and derived insights.

## Requirements

- Python 3.x with GPU access recommended (Qwen2.5-3B loaded in fp16 via
  `device_map="auto"`; CPU fallback possible but slow).
- Internet access for OpenFDA, Wikipedia, and DuckDuckGo lookups.

## Installation

```bash
pip install transformers accelerate bitsandbytes
pip install sentence-transformers
pip install faiss-cpu
pip install wikipedia
pip install duckduckgo-search
pip install requests pandas scikit-learn nltk
pip install gradio
```

The configurable notebook additionally requires:

```bash
pip install networkx
```

Both notebooks also run, near the top:

```python
nltk.download('punkt_tab')
```

## Usage

1. Run all cells top to bottom in either notebook. This loads the embedding
   model and Qwen2.5-3B-Instruct, defines all pipeline classes, instantiates
   `MedicalAssistant`, and launches a Gradio dashboard.
2. In the UI, enter a medical question (e.g. *"What are common causes of
   cough and headache?"*) and run the pipeline to see every lifecycle stage's
   output (collected documents, chunks, retrieved/ranked context, routed
   buckets, compressed context, prompt, and final response), plus analytics
   (compression ratio, average ranking score, per-stage timings).
   - In the configurable notebook, also choose/reorder which
     context-engineering steps to apply before running, and inspect the
     additional orchestration/knowledge-graph/memory/reasoning/insights tabs.
3. Programmatic use:

   ```python
   assistant = MedicalAssistant()
   result = assistant.run("What are common causes of cough and headache?")
   print(result["response"])
   print(result["analytics"])
   ```

   For the configurable notebook, an optional `step_config` can be passed to
   customize the context-engineering pipeline, e.g.:

   ```python
   step_config = [
       {"step": "rank", "enabled": True, "params": {}},
       {"step": "prune", "enabled": True, "params": {}},
       {"step": "truncate", "enabled": True, "params": {"keep_n": 5}},
   ]
   result = assistant.run(query, step_config=step_config)
   ```

## Notes

- Both notebooks collect from a fixed demo query set (OpenFDA search term
  `"headache"`, Wikipedia pages `Fever`/`Headache`/`Influenza`/`COVID-19`) in
  addition to a DuckDuckGo search on the user's actual query — this is a
  demonstration corpus, not a production retrieval setup.
- The configurable notebook persists state to disk (knowledge graph, memory,
  historical data) via `KnowledgeManagementSystem`/`MemoryStore`/
  `HistoricalDataStore`, so re-running cells across sessions can carry over
  prior state — delete the corresponding JSON/pickle files to reset.
- This is a research/demo project illustrating context-engineering concepts,
  not a validated medical device — outputs should not be used for real
  clinical decision-making.
