# Patient Passbook — Context Engineering Pipeline

A synthetic-data demo that applies the **context engineering lifecycle**
(WRITE → SELECT/RETRIEVE → RANK → ISOLATE → QUALITY → COMPRESS, plus a
memory layer) to build a longitudinal **patient health passbook**: a
validated, deduplicated XML record rendered into a downloadable PDF, with
every pipeline stage inspectable through a Gradio dashboard.

## What it does

1. Generates a synthetic population of patients (demographics, ailment
   history, immunizations, medications, lab results, activity/smartwatch
   logs) and pulls in public medical datasets for realism.
2. Registers all of that as a set of heterogeneous **data sources**.
3. For a given patient ID, runs the full context-engineering lifecycle to
   collect, chunk, retrieve, rank, route, evaluate, and compress that
   patient's records.
4. Merges in short-term/long-term **memory** (recent events + historical
   snapshots) and a chronological **timeline**.
5. Validates, deduplicates, and assembles the result into an XML passbook,
   versions it, and renders it as a formatted **PDF**.
6. Exposes all of the above — including per-stage logs — in a Gradio UI.

## Architecture / pipeline

![Uploading image.png…]()


## Components

### Data generation
- **`SyntheticPatientGenerator`** (uses `Faker`) — creates patients with
  `basic_details`, `parents_details`, `ailment_history`, `immunization`,
  `current_medication`, `lab_test_results`, `family_medical_history`,
  `health_history`, `daily_activity_log`, `physical_activity`
  (`PASSBOOK_FIELDS`).
- **`enrich_patient_record`** — randomly adds ailment history, vaccines,
  medications.
- Additional synthetic generators for lab reports, daily activity logs,
  physical activity/steps, and smartwatch summaries (avg steps, heart rate).
- 500 synthetic patients are generated and saved to
  `patients.parquet`.
- **Public datasets** pulled in for realism/breadth:
  - NIH Chest X-Ray metadata (CSV, via GitHub)
  - MedQuAD medical Q&A (`keivalya/MedQuad-MedicalQnADataset`)
  - MedDialog / augmented clinical notes (`AGBonnet/augmented-clinical-notes`)
  - PubMedQA (`qiaojin/PubMedQA`, `pqa_labeled`)
- **`DATA_SOURCES`** — a dict registering all of the above (patient_record,
  historical_report, lab_reports, ehr_data, smart_watch, medical_images,
  voice_conversation, email_chat, public_data) as the heterogeneous source
  layer the context pipeline reads from.

### Context engineering (per patient)
- **`ContextTracker`** — logs every stage name + details for later display.
- **`PatientContextCollector`** — gathers all records for a given patient ID
  across every registered data source (WRITE).
- **`PatientChunker`** — splits collected records into fixed-size chunks.
- **`PatientVectorStore`** — FAISS (`IndexFlatL2`, 384-dim) index over chunk
  embeddings; supports `add_documents` and `search`.
- **`PatientContextRanker`** — re-scores retrieved chunks by embedding
  similarity to the query.
- **`PatientContextRouter`** — buckets ranked chunks into passbook-style
  categories (e.g. `basic_details`, and others).
- **`ContextEvaluator`** — filters noisy/low-quality chunks per bucket.
- **`ContextCompressor`** — reduces each bucket down to compact text.
- `run_context_pipeline(patient_id)` wires these into the full
  WRITE → SELECT/RETRIEVE → RANK → ISOLATE → QUALITY → COMPRESS sequence.

### Memory management
- **`ShortTermMemory`** — recent events per patient, with a 24-hour expiry.
- **`LongTermMemory`** — persisted (pickle) snapshots per patient (e.g.
  yearly diagnosis records), survives across runs.
- **`MemoryTracker`** — logging, mirrors `ContextTracker` for the memory
  stage.
- **`MemoryRetrievalEngine`** — pulls both short-term and long-term memory
  for a patient.
- **`MemoryConsolidator`** — merges retrieved memory into `recent_events`
  and `historical_events`.
- **`PatientTimelineBuilder`** — builds a chronological timeline from
  ailment history plus consolidated memory.
- **`MemoryAwareContextBuilder`** — merges consolidated memory into the
  compressed context under a `memory` key.
- `run_memory_pipeline(patient)` wires the above together.

### Passbook generation
- **`PassbookValidator`** — checks that all mandatory fields (basic_details,
  parents_details, lab_test_results, current_medication, etc.) are present.
- **`PassbookCleaner`** — deduplicates repeated entries (immunizations,
  medications, ailment history).
- **`XMLPassbookBuilder`** — serializes the cleaned patient record into an
  XML passbook document.
- **`PassbookVersionManager`** — saves each XML version to disk for
  auditability.
- **`PatientPDFGenerator`** (via `reportlab`) — renders a formatted PDF
  passbook.
- **`PassbookGenerator`** — orchestrates validate → clean → build XML →
  version → render PDF, returning the XML, version path, and PDF path.
- **`PassbookArchive`** — stores generated PDF paths per patient ID.

### Dashboard
- A Gradio app (`Patient Passbook Generation System`) where entering a
  Patient ID:
  - Shows a live patient summary.
  - On **Generate Passbook**, runs the full context + memory + passbook
    pipelines and displays: context-engineering logs, memory logs, an XML
    preview, a downloadable PDF, and per-stage tabs (**WRITE**,
    **SELECT-Retrieve**, **SELECT-Rank**, **ISOLATE**, **COMPRESS**) showing
    the intermediate output of each lifecycle stage.

## Requirements

- Python 3.x; GPU optional (no LLM generation step in this notebook — it
  focuses on the retrieval/compression/document-generation lifecycle, not
  free-text answer generation).
- Internet access to download the public Hugging Face datasets and NIH CSV
  metadata on first run.

## Installation

```bash
pip install transformers accelerate bitsandbytes
pip install sentence-transformers
pip install faiss-cpu
pip install datasets
pip install gradio
pip install reportlab
pip install pandas numpy
pip install faker
pip install pyarrow
pip install kagglehub Pillow
```

## Usage

1. Run all cells top to bottom. This will:
   - Generate 500 synthetic patients and save them to
     `/content/passbook_system/patients.parquet`.
   - Download and sample the public medical datasets (NIH Chest X-Ray
     metadata, MedQuAD, MedDialog/clinical notes, PubMedQA).
   - Build the context-engineering and memory pipelines.
   - Launch the Gradio dashboard (`demo.launch(share=True)`).
2. In the UI, enter a **Patient ID** (e.g. the ID of `patients[0]`, printed
   earlier in the notebook) and click **Generate Passbook** to:
   - View the patient summary.
   - Inspect context-engineering and memory logs.
   - Preview the generated XML.
   - Download the generated PDF passbook.
   - Step through the WRITE / SELECT-Retrieve / SELECT-Rank / ISOLATE /
     COMPRESS tabs to see exactly what each lifecycle stage produced.
3. Programmatic use:

   ```python
   pipeline_result = run_context_pipeline(patient_id)
   memory_context, timeline = run_memory_pipeline(get_patient_by_id(patient_id))
   result = passbook_generator.generate(get_patient_by_id(patient_id))
   print(result["pdf"])   # path to the generated PDF
   ```

## Storage layout

All generated artifacts are written under `/content/passbook_system/`:

| Path | Contents |
|---|---|
| `vector_db/` | FAISS vector store data |
| `short_term_memory/` | Short-term memory state |
| `long_term_memory/` | Persisted long-term memory (`ltm.pkl`) |
| `pdfs/` | Generated passbook PDFs |
| `versions/` | Versioned XML passbook snapshots |
| `patients.parquet` | Generated synthetic patient dataset |

## Notes

- All patient data is **synthetically generated** (via `Faker` and random
  sampling) — no real patient information is used or required.
- Long-term memory and versioned XML snapshots persist to disk across
  notebook runs; delete the corresponding files under
  `/content/passbook_system/` to reset state.
- This is a research/demo project illustrating context-engineering and
  document-generation concepts, not a certified medical records system.
