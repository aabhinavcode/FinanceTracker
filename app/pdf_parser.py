# pdf_parser.py — CIBC-only, beginner-friendly
import pdfplumber
import pandas as pd
import re
from datetime import date
import io

# ---- Simple constants ----
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
AMOUNT_AT_END = re.compile(r"(-?\$?\d[\d,]*\.?\d{0,2})$")  # e.g. 1,234.56 / -45.00 / $12.30

# Lines that usually mean “this is a payment” (CIBC wordings)
PAYMENT_WORDS = ["PAYMENT", "PAYMENT - THANK YOU", "ONLINE PAYMENT", "MOBILE PAYMENT", "E-PMT"]

# A few high-precision category rules (fix obvious mislabels)
CATEGORY_RULES = [
    ("DENTAL", "Health"),
    ("DENTIST", "Health"),
    ("PHARMACY", "Health"),
    ("UBER EATS", "Restaurants"),
    ("PIZZA", "Restaurants"),
    ("STARBUCKS", "Restaurants"),
    ("TIM HORTONS", "Restaurants"),
    ("LOBLAW", "Grocery"),
    ("WALMART", "Grocery"),
    ("COSTCO", "Grocery"),
    ("NETFLIX", "Subscriptions"),
    ("SPOTIFY", "Subscriptions"),
]

# ---- Tiny helpers ----
def read_pdf_text(file_obj) -> str:
    """Extract text from all pages and join to one string."""
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj.getvalue()
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        pages = [ (page.extract_text() or "") for page in pdf.pages ]
    return "\n".join(pages)

def guess_year(big_text: str) -> int:
    """Find a 4-digit year anywhere; else use current year."""
    m = re.search(r"\b(20\d{2})\b", big_text)
    return int(m.group(1)) if m else date.today().year

def parse_amount(token: str):
    """Turn '$1,234.56' or '-45.00' into float, else None."""
    token = token.replace(",", "").replace("$", "")
    try:
        return float(token)
    except:
        return None

def to_date(mon: str, day: str, year: int):
    """Build Timestamp from month/day/year (or NaT)."""
    try:
        return pd.to_datetime(f"{mon} {int(day)} {year}", errors="coerce")
    except:
        return pd.NaT

def looks_like_payment(desc: str) -> bool:
    d = (desc or "").upper()
    return any(w in d for w in PAYMENT_WORDS)

def apply_category_rules(description: str, current_category: str | None) -> str | None:
    if not description:
        return current_category
    d = description.upper()
    for word, cat in CATEGORY_RULES:
        if word in d:
            return cat
    return current_category

# ---- Core: CIBC line parser ----
def parse_cibc_lines(lines: list[str], year: int):
    """Return (transactions_list, payments_list) for CIBC PDFs."""
    all_tx, all_pay = [], []
    inside_txn = inside_pay = False

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue

        # Section toggles (these are what we saw in your statement)
        if "Your payments" in ln:
            inside_txn, inside_pay = False, True
            continue
        if "Your new charges and credits" in ln:
            inside_txn, inside_pay = True, False
            continue
        if any(bad in ln for bad in ["Important Notice", "Spend Report", "Prepared for:", "Trademark"]):
            inside_txn = inside_pay = False
            continue

        # Quick prefilters
        if not any(m in ln for m in MONTHS):
            continue

        m_amt = AMOUNT_AT_END.search(ln)
        if not m_amt:
            continue
        amount_token = m_amt.group(1)
        amount = parse_amount(amount_token)
        if amount is None:
            continue

        parts = ln.split()
        if len(parts) < 6:
            continue

        # Expect: TransMon TransDay PostMon PostDay ... Description ... Amount
        trans_mon, trans_day = parts[0], parts[1]
        post_mon,  post_day  = parts[2], parts[3]
        trans_date = to_date(trans_mon, trans_day, year)
        post_date  = to_date(post_mon,  post_day,  year)

        # description = between second date and amount
        amount_start = ln.rfind(amount_token)
        head = f"{trans_mon} {trans_day} {post_mon} {post_day} "
        if not ln.startswith(head):  # if parsing shifts, skip line
            continue
        description = ln[len(head):amount_start].strip()

        if inside_pay or looks_like_payment(description):
            all_pay.append(["CIBC", trans_date, post_date, description, amount])
        elif inside_txn:
            maybe_cat = parts[-2]  # token before amount
            if re.fullmatch(r"[A-Za-z\-]+", maybe_cat):
                category = maybe_cat
                if description.endswith(maybe_cat):
                    description = description[: -len(maybe_cat)].strip()
            else:
                category = None
            category = apply_category_rules(description, category)
            all_tx.append(["CIBC", trans_date, post_date, description, category, amount])

    return all_tx, all_pay

# ---- Public API used by app.py ----
def extract_transactions(uploaded_files):
    """
    Input: list of Streamlit UploadedFile (or file-like) objects.
    Output: (df_transactions, df_payments)
      df_transactions: Bank, Trans date, Post date, Description, Category, Amount
      df_payments    : Bank, Trans date, Post date, Description, Amount
    """
    all_tx, all_pay = [], []

    for f in uploaded_files:
        big_text = read_pdf_text(f)
        year = guess_year(big_text)
        lines = [ln for ln in big_text.split("\n") if ln.strip()]

        tx, pay = parse_cibc_lines(lines, year)   # CIBC-only
        all_tx.extend(tx)
        all_pay.extend(pay)

    df_transactions = pd.DataFrame(all_tx, columns=["Bank","Trans date","Post date","Description","Category","Amount"])
    df_payments     = pd.DataFrame(all_pay, columns=["Bank","Trans date","Post date","Description","Amount"])

    # Ensure types for charts
    for c in ["Trans date","Post date"]:
        if c in df_transactions: df_transactions[c] = pd.to_datetime(df_transactions[c], errors="coerce")
        if c in df_payments:     df_payments[c]     = pd.to_datetime(df_payments[c], errors="coerce")
    if "Amount" in df_transactions: df_transactions["Amount"] = pd.to_numeric(df_transactions["Amount"], errors="coerce")
    if "Amount" in df_payments:     df_payments["Amount"]     = pd.to_numeric(df_payments["Amount"], errors="coerce")

    return df_transactions, df_payments
