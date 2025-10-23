import streamlit as st
import pandas as pd

from pdf_parser import extract_transactions
import finance_db as db
import ml_models as ml

st.set_page_config(page_title="Finance Intelligence App", layout="wide")

# -------- Sidebar (minimal) --------
with st.sidebar:
    st.header("Upload Statement")
    uploaded_files = st.file_uploader("Upload CIBC PDF statements", type="pdf", accept_multiple_files=True)
    save_to_db = st.button("Save to Database")
    load_from_db = st.button("Load from Database")
    st.caption("Tip: Save once → then load from DB next time.")

st.title("Personal Finance Intelligence Dashboard")

# -------- Decide data source --------
if uploaded_files and not load_from_db:
    df_t, df_p = extract_transactions(uploaded_files)
elif load_from_db:
    try:
        db.create_tables(); db.create_indexes_and_views()
        df_t = db.read_all_transactions()
        df_p = db.read_all_payments()
        st.success("Loaded data from database successfully.")
    except Exception as e:
        st.error(f"Database load error: {e}")
        st.stop()
else:
    st.info("Upload one or more PDFs **or** click 'Load from Database' to view existing data.")
    st.stop()

# -------- KPIs --------
amt_col = "Amount" if "Amount" in df_t.columns else "amount"
date_col = "Trans date" if "Trans date" in df_t.columns else "trans_date"

total_spend = pd.to_numeric(df_t[amt_col], errors="coerce").sum()
total_payments = pd.to_numeric(df_p["Amount"] if "Amount" in df_p.columns else df_p["amount"], errors="coerce").sum()
txn_count = len(df_t)
months_cov = (
    df_t[date_col].dt.to_period("M").nunique()
    if date_col in df_t.columns and pd.api.types.is_datetime64_any_dtype(df_t[date_col]) else 0
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Spend", f"${total_spend:,.2f}")
k2.metric("Transactions", f"{txn_count:,}")
k3.metric("Total Payments", f"${total_payments:,.2f}")
k4.metric("Months Covered", f"{months_cov}")

st.markdown("---")

# -------- Tabs (Overview / Raw / AI) --------
tab1, tab2, tab3 = st.tabs(["Overview", "Raw Data", "AI Insights"])

with tab1:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Monthly Spending Trend")
        df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
        df_t["Month"] = df_t[date_col].dt.to_period("M")
        monthly_spend = df_t.groupby("Month")[amt_col].sum()
        monthly_spend.index = monthly_spend.index.astype(str)
        st.line_chart(monthly_spend)

    with right:
        st.subheader("Spending by Category")
        cat_col = "Category" if "Category" in df_t.columns else "category"
        if cat_col in df_t.columns:
            cat_spend = df_t.groupby(cat_col)[amt_col].sum().sort_values(ascending=False)
            st.bar_chart(cat_spend)
        else:
            st.info("No category column available.")

with tab2:
    st.subheader("Transactions")
    st.dataframe(df_t, use_container_width=True, height=320)

    st.subheader("Payments")
    st.dataframe(df_p, use_container_width=True, height=220)

    diff = total_spend - total_payments
    st.caption(f"Payments vs Transactions difference: **${diff:,.2f}**")

with tab3:
    st.subheader("Monthly Spending Forecast")
    try:
        fc, hist = ml.forecast_with_prophet(df_t, months_ahead=4)
        fig = ml.plot_prophet_forecast(fc, hist)
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")

    st.markdown("---")
    st.subheader("AI-Detected Suspicious Transactions")
    try:
        # fixed contamination (3%) for simplicity
        anoms = ml.detect_anomalies_isoforest(df_t, contamination=0.03)
        if anoms.empty or (~anoms["is_anomaly"]).all():
            st.success("No suspicious transactions detected with the current model.")
        else:
            # KPIs
            kpis = ml.anomaly_quick_kpis(anoms, df_t.rename(columns={date_col: "trans_date", amt_col: "amount"}))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("# flagged", f"{kpis['flagged']}")
            c2.metric("% rows", f"{kpis['pct_rows']:.1f}%")
            c3.metric("$ flagged", f"${kpis['flagged_amount']:,.2f}")
            c4.metric("% of spend", f"{kpis['pct_amount']:.1f}%")

            st.dataframe(anoms[anoms["is_anomaly"]].head(50), use_container_width=True)
            st.caption("Most suspicious first (lowest anomaly_score).")
    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")

# -------- Save to DB --------
if save_to_db:
    try:
        db.create_tables()
        ins_tx = db.upsert_transactions(df_t)
        ins_py = db.upsert_payments(df_p)
        st.success(f"Transactions inserted: {ins_tx}")
        st.success(f"Payments inserted: {ins_py}")
        if hasattr(db, "create_indexes_and_views"):
            db.create_indexes_and_views()
    except Exception as e:
        st.error(f"Database error: {e}")
