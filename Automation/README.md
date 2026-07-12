# Invoice RPA Demos: Scripted RPA vs. Agentic RPA

Two small desktop-automation scripts that both do the same basic job — take
invoice data, write it out, and save it as a file via UI automation (opening
Notepad, typing, saving) — but differ in **how** the content to be typed/saved
is decided. Together they're a simple before/after comparison of classic RPA
vs. an LLM-driven "agentic" version of the same task.

| Script | Approach |
|---|---|
| `rpa.py` | **Classic RPA.** Fixed invoice dict, hardcoded template, hardcoded filename. No model involved. |
| `apa.py` | **Agentic RPA.** Same kind of invoice dict, but an LLM (Qwen2-0.5B-Instruct) is prompted to normalize the vendor name, compose a one-line summary, and choose a filename — with output validation and a deterministic fallback if the model's output can't be trusted. |

Both scripts use `pyautogui` to literally drive the OS UI (press Start, type
"Notepad", press Enter, type text, `Ctrl+S`, type a filename, press Enter) —
there is no API call to Notepad; it's simulated keyboard/mouse input.

## `rpa.py` — classic RPA

- `invoice_data` is a hardcoded dict (vendor, amount, date, invoice number).
- Builds a fixed multi-line invoice text block via an f-string template.
- Opens Notepad, types the block, saves it under a hardcoded filename
  (`rpademo3.txt`).
- No reasoning, no model, no validation — same steps every run, same output
  shape every time.

## `apa.py` — agentic RPA

- Loads `Qwen/Qwen2-0.5B-Instruct` via `transformers`.
- `run_agent(invoice_data)`:
  - Prompts the model (system + user messages, chat template) to return
    **only** a JSON object with `vendor` (normalized), `filename`
    (`VendorName_InvoiceNo.txt`), and `summary` (one sentence with vendor,
    amount, date).
  - Extracts the last JSON-looking block from the model's raw output with a
    regex and parses it.
  - **`validate_result`** — rejects the model's output unless the real
    vendor name, amount, and invoice number actually appear in it (guards
    against the model hallucinating example values instead of using the
    real input).
  - **`build_result_directly`** — a deterministic fallback that constructs
    the same `{vendor, filename, summary}` shape directly from
    `invoice_data` (string normalization, no model) if generation or
    validation fails.
- `agentic_rpa()` ties it together: run the agent, fall back if needed, then
  open Notepad, type `result["summary"]`, and save as `result["filename"]`.

## Why this pairing is useful

Side by side, these two scripts isolate exactly one variable: **who decides
the content and filename** — a fixed template (`rpa.py`) vs. an LLM with
guardrails (`apa.py`) — while the actual automation mechanics (UI driving via
`pyautogui`) stay identical. That makes it a small, concrete illustration of
where "agentic" behavior can be layered onto existing RPA scripts, and what
extra scaffolding (prompt constraints, output validation, deterministic
fallback) is needed to make an LLM-in-the-loop step trustworthy enough to
drive a real UI action like saving a file.

## Requirements

```bash
pip install pyautogui transformers torch
```

- A desktop environment with keyboard/mouse focus available — `pyautogui`
  actually controls the OS, so these scripts must be run with a visible
  desktop session (not headless), and Notepad (or an equivalent text editor
  reachable via the Start menu) available.
- `apa.py` additionally downloads `Qwen/Qwen2-0.5B-Instruct` on first run
  (internet access + disk space required); no GPU is required for this small
  model but one will speed things up.

## Usage

Each script is standalone and runs its demo immediately on execution:

```bash
python rpa.py
python apa.py
```

Both scripts print a "Starting in 5 seconds..." countdown before taking
control of the mouse/keyboard — use that window to switch focus to a safe
desktop, since the automation will type into whatever window/Notepad
instance ends up focused.

## Notes

- These are **conceptual/demo scripts**, not part of a larger project or
  packaged tool — they exist to explore and compare RPA vs. agentic-RPA
  patterns, not to be deployed as-is.
- Invoice data is hardcoded/synthetic in both scripts; there's no file input,
  OCR, or real invoice source involved.
- Because `pyautogui` drives real UI input, running either script will take
  over keyboard/mouse focus for a few seconds — don't run them unattended on
  a machine where that could cause problems.
