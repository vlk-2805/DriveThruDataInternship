import pyautogui
import time

def rpa_process_invoice():

    invoice_data = {
        "vendor": "Acme Corp",
        "amount": "1250.00",
        "date": "2026-05-23",
        "invoice_no": "INV-4021"
    }

    # Give user time before automation starts
    print("Starting RPA in 5 seconds...")
    time.sleep(5)

    # Step 1: Open Start Menu
    pyautogui.press("win")
    time.sleep(1)

    # Step 2: Search for Notepad
    pyautogui.write("Notepad", interval=0.1)
    time.sleep(1)

    # Step 3: Open Notepad
    pyautogui.press("enter")
    time.sleep(2)

    # Step 4: Type Invoice Data
    invoice_text = f"""
INVOICE DETAILS
====================

Vendor      : {invoice_data['vendor']}
Amount      : {invoice_data['amount']}
Date        : {invoice_data['date']}
Invoice No  : {invoice_data['invoice_no']}
"""

    pyautogui.write(invoice_text, interval=0.03)

    time.sleep(1)

    # Step 5: Save File
    pyautogui.hotkey("ctrl", "s")
    time.sleep(2)

    # Step 6: Enter Filename
    filename = "rpademo3.txt"

    pyautogui.write(filename, interval=0.05)
    time.sleep(1)

    # Step 7: Press Save
    pyautogui.press("enter")

    print("Invoice saved successfully!")

# Run the automation
rpa_process_invoice()