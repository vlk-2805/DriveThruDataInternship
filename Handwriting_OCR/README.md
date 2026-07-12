# Medical Prescription OCR + NER-Based Agentic RAG

An end-to-end pipeline that reads a handwritten medical prescription image,
extracts structured drug information, cross-checks it against live medical
data sources, and answers free-text questions about the prescription using a
tool-using ReAct agent. This is the baseline pipeline later refactored into an
MCP (Model Context Protocol) architecture in
[`handwritingocr_mcp.ipynb`](./README_handwritingocr_mcp.md).

## Architecture

<img width="1705" height="909" alt="image" src="https://github.com/user-attachments/assets/2b53d60a-901d-4c8a-b544-f9ea0e981b67" />


## Pipeline components

- **`HandwritingOCR`** — uses `Qwen/Qwen2-VL-2B-Instruct` (vision-language
  model) to extract and lightly spell-correct handwritten text from a
  prescription image, preserving line breaks.
- **`MedicalKnowledgeRetriever`** — builds a hybrid dense (FAISS,
  `all-MiniLM-L6-v2` embeddings) + sparse (BM25) index over passages pulled
  from:
  - Wikipedia (via `wikipedia-api`)
  - OpenFDA drug label API (`description`, `indications_and_usage`,
    `warnings`, `dosage_and_administration`, `adverse_reactions`,
    `drug_interactions`, `contraindications`)
  - DuckDuckGo web search as a fallback when no structured data is found
  - RxNorm (NLM RxNav) for canonical drug-name lookups
- **`MedicalTools`** — wraps the retriever's capabilities as agent-callable
  tools (`load_drug_knowledge`, `search_knowledge_base`,
  `fetch_openfda_field`, `fetch_wikipedia`, `lookup_rxnorm`,
  `web_search_engine`) behind a simple `dispatch(name, args)` interface.
- **`NERCorrector`** — reconciles GLiNER-extracted drug names against RxNorm
  canonical forms and Wikipedia page existence, flagging corrections and
  their sources.
- **ReAct agent** — a Thought → Action → Args loop (few-shot prompted, driven
  by the same Qwen2-VL model in text-only mode) that calls `MedicalTools`
  directly to gather evidence before producing a final answer.
- **`MedicalQA`** — lightweight question-answering wrapper that delegates to
  the agent once the retriever has indexed relevant passages.
- **`MedicalPipeline`** — orchestrates OCR → NER → knowledge loading →
  correction → structured JSON → agent Q&A.
- **Gradio UI** — upload a prescription image, ask a question, and view OCR
  output, extracted/corrected entities, structured prescription JSON, the
  agent's answer, sources, and reasoning trace.

## Requirements

- Python 3.12 (tested on Kaggle with an NVIDIA Tesla T4 GPU); CPU fallback
  supported but slower.
- Internet access for Wikipedia, OpenFDA, RxNorm, and DuckDuckGo lookups.

## Installation

```bash
pip install transformers==4.51.3 accelerate qwen-vl-utils
pip install wikipedia-api requests sentence-transformers faiss-cpu
pip install gliner
pip install rank-bm25
pip install duckduckgo-search
pip install gradio pandas
```

## Usage

1. Run all cells top to bottom. This loads the OCR/NER models, builds the
   pipeline, and launches a Gradio demo (`demo.launch(...)`).
2. In the UI:
   - Upload a prescription image.
   - Optionally enter a medical question (e.g. *"What is the dosage for this
     medicine?"*).
   - Click **Run Pipeline** to see OCR text, NER + corrections, structured
     prescription JSON, the agent's answer, sources, and its reasoning trace.
3. Programmatic use:

   ```python
   pipeline = MedicalPipeline()
   result = pipeline.process_image("prescription.png")
   print(result["structured_prescription"])

   qa = pipeline.ask("What are the side effects?", result["structured_prescription"])
   print(qa["answer"])
   ```

## Notes

- Originally developed/run as a Kaggle notebook (GPU: NVIDIA Tesla T4,
  internet enabled).
- Tool calls are dispatched via direct Python method calls
  (`self.tools.dispatch(...)`) — see the MCP notebook for a version where this
  boundary is replaced with a proper client/server protocol.
- Research/demo project only — not validated for clinical use.
