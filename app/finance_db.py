# finance_db.py — DB helpers

import hashlib
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://postgres:123@localhost:5432/FinanceDatabase"
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

def create_tables():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            bank VARCHAR(50),
            trans_date DATE,
            post_date DATE,
            description TEXT,
            category VARCHAR(100),
            amount NUMERIC,
            row_hash TEXT UNIQUE
        );"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS payments (
            id SERIAL PRIMARY KEY,
            bank VARCHAR(50),
            trans_date DATE,
            post_date DATE,
            description TEXT,
            amount NUMERIC,
            row_hash TEXT UNIQUE
        );"""))

def create_indexes_and_views():
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_transactions_trans_date ON transactions (trans_date);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_transactions_category   ON transactions (category);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_transactions_row_hash   ON transactions (row_hash);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_payments_trans_date     ON payments (trans_date);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_payments_row_hash       ON payments (row_hash);"))
        conn.execute(text("""
        CREATE OR REPLACE VIEW v_monthly_spend AS
        SELECT date_trunc('month', trans_date)::date AS month, SUM(COALESCE(amount,0)) AS total_spend
        FROM transactions GROUP BY 1 ORDER BY 1;"""))
        conn.execute(text("""
        CREATE OR REPLACE VIEW v_category_spend AS
        SELECT COALESCE(NULLIF(TRIM(category),''),'Uncategorized') AS category,
               SUM(COALESCE(amount,0)) AS total_spend
        FROM transactions GROUP BY 1 ORDER BY total_spend DESC;"""))
        conn.execute(text("""
        CREATE OR REPLACE VIEW v_top_merchants AS
        SELECT LEFT(COALESCE(description,''), 80) AS merchant_hint,
               SUM(COALESCE(amount,0)) AS total_spend, COUNT(*) AS txn_count
        FROM transactions GROUP BY 1
        ORDER BY total_spend DESC, txn_count DESC LIMIT 200;"""))

def _norm_date(x):
    if pd.isna(x): return ""
    try: return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception: return str(x)

def _norm_text(x):
    if x is None: return ""
    return str(x).strip().upper()

def _norm_amount(x):
    if pd.isna(x): return ""
    try: return f"{float(x):.2f}"
    except Exception: return str(x)

def _hash_row(bank, trans_date, post_date, description, category, amount):
    s = "|".join([
        _norm_text(bank), _norm_date(trans_date), _norm_date(post_date),
        _norm_text(description), _norm_text(category), _norm_amount(amount)
    ])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _prepare_tx_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["bank","trans_date","post_date","description","category","amount","row_hash"])
    cols_lower = {c.lower(): c for c in df.columns}
    bank_col = cols_lower.get("bank", "Bank" if "Bank" in df.columns else None)
    tcol = cols_lower.get("trans date", "Trans date" if "Trans date" in df.columns else None)
    pcol = cols_lower.get("post date", "Post date" if "Post date" in df.columns else None)
    dcol = cols_lower.get("description", "Description" if "Description" in df.columns else None)
    ccol = cols_lower.get("category", "Category" if "Category" in df.columns else None)
    acol = cols_lower.get("amount", "Amount" if "Amount" in df.columns else None)
    if not all([bank_col, tcol, pcol, dcol, acol]):
        raise ValueError("Incoming transactions DataFrame is missing required columns.")
    df2 = df.copy()
    df2["row_hash"] = df2.apply(lambda r: _hash_row(r.get(bank_col), r.get(tcol), r.get(pcol),
                                                    r.get(dcol), r.get(ccol) if ccol in df2.columns else "",
                                                    r.get(acol)), axis=1)
    df2 = df2.rename(columns={
        bank_col:"bank", tcol:"trans_date", pcol:"post_date", dcol:"description",
        ccol:"category" if ccol else "category", acol:"amount"
    })[["bank","trans_date","post_date","description","category","amount","row_hash"]]
    return df2

def _prepare_pay_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["bank","trans_date","post_date","description","amount","row_hash"])
    cols_lower = {c.lower(): c for c in df.columns}
    bank_col = cols_lower.get("bank", "Bank" if "Bank" in df.columns else None)
    tcol = cols_lower.get("trans date", "Trans date" if "Trans date" in df.columns else None)
    pcol = cols_lower.get("post date", "Post date" if "Post date" in df.columns else None)
    dcol = cols_lower.get("description", "Description" if "Description" in df.columns else None)
    acol = cols_lower.get("amount", "Amount" if "Amount" in df.columns else None)
    if not all([bank_col, tcol, pcol, dcol, acol]):
        raise ValueError("Incoming payments DataFrame is missing required columns.")
    df2 = df.copy()
    df2["row_hash"] = df2.apply(lambda r: _hash_row(r.get(bank_col), r.get(tcol), r.get(pcol),
                                                    r.get(dcol), "", r.get(acol)), axis=1)
    df2 = df2.rename(columns={
        bank_col:"bank", tcol:"trans_date", pcol:"post_date", dcol:"description", acol:"amount"
    })[["bank","trans_date","post_date","description","amount","row_hash"]]
    return df2

def _existing_hashes(table_name: str) -> set:
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT row_hash FROM {table_name};")).fetchall()
    return set(r[0] for r in rows)

def upsert_transactions(df: pd.DataFrame) -> int:
    df2 = _prepare_tx_df(df)
    if df2.empty: return 0
    existing = _existing_hashes("transactions")
    new_df = df2[~df2["row_hash"].isin(existing)]
    if new_df.empty: return 0
    new_df.to_sql("transactions", engine, if_exists="append", index=False, method="multi", chunksize=1000)
    return len(new_df)

def upsert_payments(df: pd.DataFrame) -> int:
    df2 = _prepare_pay_df(df)
    if df2.empty: return 0
    existing = _existing_hashes("payments")
    new_df = df2[~df2["row_hash"].isin(existing)]
    if new_df.empty: return 0
    new_df.to_sql("payments", engine, if_exists="append", index=False, method="multi", chunksize=1000)
    return len(new_df)

def read_all_transactions(limit: int | None = None) -> pd.DataFrame:
    if limit:
        return pd.read_sql(text(f"""
            SELECT bank, trans_date, post_date, description, category, amount, row_hash
            FROM transactions ORDER BY trans_date DESC, id DESC LIMIT {int(limit)};
        """), engine)
    return pd.read_sql(text("""
        SELECT bank, trans_date, post_date, description, category, amount, row_hash
        FROM transactions ORDER BY trans_date DESC, id DESC;
    """), engine)

def read_all_payments(limit: int | None = None) -> pd.DataFrame:
    if limit:
        return pd.read_sql(text(f"""
            SELECT bank, trans_date, post_date, description, amount, row_hash
            FROM payments ORDER BY trans_date DESC, id DESC LIMIT {int(limit)};
        """), engine)
    return pd.read_sql(text("""
        SELECT bank, trans_date, post_date, description, amount, row_hash
        FROM payments ORDER BY trans_date DESC, id DESC;
    """), engine)

def read_view_monthly() -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM v_monthly_spend;"), engine)

def read_view_category() -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM v_category_spend;"), engine)

def read_view_top_merchants() -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM v_top_merchants;"), engine)
