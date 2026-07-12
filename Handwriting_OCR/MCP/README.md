# Medical Prescription OCR + MCP-Based Agentic RAG

Refactors a handwriting-OCR + medical RAG pipeline (see the companion
[`handwritingocr+ner+rag.ipynb`](../README.md)) into a proper
**Model Context Protocol (MCP)** architecture. All tool calls that the reasoning
agent makes (drug lookups, knowledge-base search, OpenFDA/Wikipedia/RxNorm
fetches) now travel through MCP client/server messages instead of direct Python
method calls, so the same agent code could talk to a remote/out-of-process tool
server with no changes.

## Architecture

<img width="1342" height="716" alt="image" src="https://github.com/user-attachments/assets/331e8cb5-4eb5-4651-a528-2790a62f0e89" />


| Component        | Original                          | MCP version                                |
|-------------------|------------------------------------|---------------------------------------------|
| Tool invocation   | `self.tools.dispatch(name, args)` | `self.mcp_client.call_tool_sync(name, args)` |
| Tool discovery    | Hard-coded `dispatch()` dict       | `await client.list_tools()` at runtime       |
| Tool schema       | None                                | JSON Schema per tool                         |
| Protocol boundary | None                                | MCP messages                                 |

## Pipeline

1. **OCR** — `HandwritingOCR` (Qwen2-VL-2B-Instruct) reads a prescription image
   and returns cleaned, spelling-corrected text.
2. **NER** — `GLiNER` (`urchade/gliner_medium-v2.1`) extracts `drug`, `strength`,
   `dosage`, `frequency`, `duration`, `form`, and `route` entities from the OCR
   text.
3. **Knowledge loading (via MCP)** — for every detected drug entity, the
   pipeline calls the `load_drug_knowledge` MCP tool, which fetches Wikipedia
   summaries and OpenFDA label sections and indexes them (FAISS dense +
   BM25 sparse) inside the MCP server.
4. **Entity correction** — `NERCorrector` reconciles OCR'd drug names against
   RxNorm canonical names and Wikipedia page existence.
5. **Agentic Q&A** — a `ReActAgent` runs a Thought → Action → Args loop, calling
   MCP tools (`search_knowledge_base`, `fetch_openfda_field`, `fetch_wikipedia`,
   `lookup_rxnorm`, `web_search_engine`, `finish`) to answer free-text medical
   questions grounded in the retrieved evidence.
6. **UI** — a Gradio app ties the above into an interactive demo: upload an
   image, ask a question, and see OCR output, NER + corrections, the structured
   prescription JSON, the agent's answer, sources, and its full reasoning
   trace (including every MCP call/response).

## MCP components

- **`mcp_server` (`Server("medical-rag-server")`)** — registers:
  - `list_tools()` — returns JSON-Schema-described tool definitions
    (`load_drug_knowledge`, `search_knowledge_base`, `fetch_openfda_field`,
    `fetch_wikipedia`, `lookup_rxnorm`, `web_search_engine`).
  - `call_tool(name, arguments)` — executes the requested tool against the
    shared `MedicalKnowledgeRetriever` instance and returns `TextContent`.
- **`MCPClient`** — an in-process client used by the notebook (no subprocess
  required). Exposes `list_tools()`, `call_tool()` (async), and
  `call_tool_sync()` (thread-safe sync wrapper for use inside the ReAct loop).
  In production this can be swapped for a real MCP `stdio_client` /
  `ClientSession` without touching the agent code.
- **`ReActAgent`** — identical reasoning loop to a plain-dispatch agent; the
  only change is that every tool call goes through
  `self.mcp_client.call_tool_sync(action, args)` instead of a hard-coded
  Python dispatcher.

## Requirements

- Python 3.10+, GPU recommended (falls back to CPU) for the Qwen2-VL OCR model.
- Internet access for Wikipedia, OpenFDA, RxNorm, and DuckDuckGo lookups.

## Installation

```bash
pip install transformers==4.51.3 accelerate qwen-vl-utils
pip install wikipedia-api requests sentence-transformers faiss-cpu
pip install gliner rank-bm25 duckduckgo-search gradio
pip install mcp   # MCP Python SDK
```

## Usage

1. Run all cells in order (top to bottom). The notebook will:
   - Load the Qwen2-VL OCR model and GLiNER NER model.
   - Spin up the in-process MCP server and client.
   - Launch a Gradio interface (`demo.launch(debug=True, share=True)`).
2. In the Gradio UI:
   - Upload a photo of a handwritten prescription.
   - Optionally type a medical question (e.g. *"What are the side effects of
     the medicines prescribed?"*).
   - Click **Run Pipeline** to see OCR text, extracted/corrected entities, the
     structured prescription JSON, the agent's answer, its sources, and the
     step-by-step MCP reasoning trace.
3. Programmatic use (without the UI):

   ```python
   pipeline = MedicalPipeline()
   result = pipeline.process_image("prescription.png")
   answer = pipeline.ask(
       "What are the side effects of these medicines?",
       result["structured_prescription"],
   )
   ```

## Notes

- `MedicalKnowledgeRetriever` and `HandwritingOCR` are unchanged from the
  original pipeline — retrieval logic now simply lives *inside* the MCP
  server rather than being called directly by the agent.
- Data sources: Wikipedia (`wikipedia-api`), OpenFDA drug label API, RxNorm
  (NLM RxNav), and DuckDuckGo as a web-search fallback.
- This is a research/demo notebook, not a medical device — outputs should not
  be used for real clinical decision-making.
