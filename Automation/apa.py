import pyautogui
import time
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================
# MODEL SETUP
# =========================================

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

print("Loading Qwen2-0.5B-Instruct...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded!")

# =========================================
# TOOL FUNCTIONS
# =========================================

def open_notepad():

    pyautogui.press("win")
    time.sleep(1)

    pyautogui.write("Notepad", interval=0.05)
    time.sleep(1)

    pyautogui.press("enter")
    time.sleep(2)


def save_file(filename):

    pyautogui.hotkey("ctrl", "s")
    time.sleep(2)

    pyautogui.write(filename, interval=0.03)
    time.sleep(1)

    pyautogui.press("enter")


# =========================================
# FALLBACK: BUILD RESULT DIRECTLY
# =========================================

def build_result_directly(invoice_data):
    """
    Constructs the normalized result purely from invoice_data,
    used as a fallback when the model output fails validation.
    """
    vendor_raw = invoice_data.get("vendor", "Unknown")

    # Normalize vendor name: title-case, strip trailing punctuation
    vendor = vendor_raw.strip().rstrip(".")
    vendor = " ".join(word.capitalize() for word in vendor.split())

    invoice_no = invoice_data.get("invoice_no", "")
    amount     = invoice_data.get("amount", "0.00")
    date       = invoice_data.get("date", "")

    # Filename: replace spaces with underscores, append invoice number
    vendor_slug = vendor.replace(" ", "_")
    filename    = f"{vendor_slug}_{invoice_no}.txt"

    summary = (
        f"Invoice from {vendor} for amount \u20b9{amount} dated {date}."
    )

    return {
        "vendor":   vendor,
        "filename": filename,
        "summary":  summary,
    }


# =========================================
# VALIDATE MODEL OUTPUT
# =========================================

def validate_result(result, invoice_data):
    """
    Returns True only if the model output actually reflects
    the real invoice data, not the example values.
    """
    required_keys = {"vendor", "filename", "summary"}
    if not required_keys.issubset(result.keys()):
        print("Validation failed: missing keys.")
        return False

    vendor_raw   = invoice_data.get("vendor", "").lower()
    amount       = invoice_data.get("amount", "")
    invoice_no   = invoice_data.get("invoice_no", "")

    summary_lower   = result.get("summary",  "").lower()
    filename_lower  = result.get("filename", "").lower()
    vendor_lower    = result.get("vendor",   "").lower()

    # Strip common suffixes so "acme corp." matches "acme corp"
    vendor_core = re.sub(r'[\s.,]+$', '', vendor_raw).strip()

    # The vendor name must appear (loosely) in the output
    if vendor_core not in vendor_lower and vendor_core not in summary_lower:
        print(f"Validation failed: vendor '{vendor_core}' not found in output.")
        return False

    # The real amount must appear in the summary
    if amount not in summary_lower:
        print(f"Validation failed: amount '{amount}' not found in summary.")
        return False

    # The real invoice number must appear in filename or summary
    if invoice_no not in filename_lower and invoice_no.lower() not in summary_lower:
        print(f"Validation failed: invoice_no '{invoice_no}' not found in output.")
        return False

    return True


# =========================================
# AGENT FUNCTION
# =========================================

def run_agent(invoice_data):

    # =========================================
    # PROMPT — no example output; schema only
    # =========================================

    messages = [
        {
            "role": "system",
            "content": (
                "You are an invoice-processing AI. "
                "Return ONLY a valid JSON object with exactly three keys:\n"
                "  vendor   – normalized vendor name (title-case, no trailing punctuation)\n"
                "  filename – a safe filename: VendorName_InvoiceNo.txt\n"
                "  summary  – one sentence: "
                "\"Invoice from <vendor> for amount \u20b9<amount> dated <date>.\"\n"
                "Do not include any explanation, markdown, or extra text."
            )
        },
        {
            "role": "user",
            "content": (
                "Normalize the following invoice into the JSON format described.\n\n"
                f"vendor:     {invoice_data['vendor']}\n"
                f"amount:     {invoice_data['amount']}\n"
                f"date:       {invoice_data['date']}\n"
                f"invoice_no: {invoice_data['invoice_no']}\n\n"
                "Output the JSON object now."
            )
        }
    ]

    # =========================================
    # APPLY CHAT TEMPLATE
    # =========================================

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    # =========================================
    # GENERATE
    # =========================================

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    # =========================================
    # DECODE ONLY NEW TOKENS
    # =========================================

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    generated = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    print("\n========== RAW OUTPUT ==========\n")
    print(generated)

    # =========================================
    # EXTRACT JSON
    # =========================================

    matches = re.findall(r'\{[\s\S]*?\}', generated)

    if not matches:
        print("No JSON found in model output — using fallback.")
        return None

    json_text = matches[-1]

    # =========================================
    # PARSE JSON
    # =========================================

    try:
        result = json.loads(json_text)
    except Exception as e:
        print("JSON parsing failed:", e, "— using fallback.")
        return None

    # =========================================
    # VALIDATE: ensure output reflects real data
    # =========================================

    if not validate_result(result, invoice_data):
        print("Model output did not match invoice data — using fallback.")
        return None

    return result


# =========================================
# MAIN AGENTIC WORKFLOW
# =========================================

def agentic_rpa():

    invoice_data = {
        "vendor":     "Agentic corp.",
        "amount":     "1550.00",
        "date":       "2026-05-22",
        "invoice_no": "INV-4021"
    }

    print("Starting in 5 seconds...")
    time.sleep(5)

    # =========================================
    # AGENT REASONING (with fallback)
    # =========================================

    result = run_agent(invoice_data)

    if result is None:
        print("Using direct fallback to build result from invoice data.")
        result = build_result_directly(invoice_data)

    print("\n========== FINAL RESULT ==========\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # =========================================
    # OPEN NOTEPAD
    # =========================================

    open_notepad()

    # =========================================
    # TYPE SUMMARY
    # =========================================

    pyautogui.write(result["summary"], interval=0.01)

    time.sleep(1)

    # =========================================
    # SAVE FILE
    # =========================================

    save_file(result["filename"])

    print("\nAgentic RPA completed successfully!")
    print(f"Saved file: {result['filename']}")


# =========================================
# RUN
# =========================================

agentic_rpa()