import streamlit as st
import pandas as pd
from pdf_parser import extract_transactions
import finance_db as db
import ml_models as ml

st.set_page_config(page_title="Finance Intelligence App", layout="wide")
df_transactions = pd.DataFrame()
df_payments = pd.DataFrame()
# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Upload Statement")
    uploaded_files = st.file_uploader(
        "Upload CIBC PDF statements", type="pdf", accept_multiple_files=True
    )
    save_to_db = st.button("Save to Database")
    load_from_db = st.button("Load from Database")
    st.caption("Tip: Save once → then load from DB next time.")

# ---------------- Main ----------------
st.title("Personal Finance Intelligence Dashboard")

# ---------- Decide data source ----------
source = None

if uploaded_files and not load_from_db:
    # Parse PDFs
    df_t, df_p = extract_transactions(uploaded_files)
    source = "uploaded"

elif load_from_db:
    try:
        db.create_tables()
        db.create_indexes_and_views()
        df_t = db.read_all_transactions()
        df_p = db.read_all_payments()
        source = "database"
        st.success("Loaded data from database successfully.")
    except Exception as e:
        st.error(f"Database load error: {e}")
        st.stop()

else:
    st.info("Upload one or more PDFs **or** click 'Load from Database' to view existing data.")
    st.stop()

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)

total_spend = df_t["Amount"].sum() if "Amount" in df_t.columns else df_t["amount"].sum()
txn_count = len(df_t)
total_payments = df_p["Amount"].sum() if "Amount" in df_p.columns else df_p["amount"].sum()

date_col = "Trans date" if "Trans date" in df_t.columns else "trans_date"
months_cov = (
    df_t[date_col].dt.to_period("M").nunique()
    if date_col in df_t.columns and pd.api.types.is_datetime64_any_dtype(df_t[date_col])
    else 0
)

col1.metric("Total Spend", f"${total_spend:,.2f}")
col2.metric("Transactions", f"{txn_count:,}")
col3.metric("Total Payments", f"${total_payments:,.2f}")
col4.metric("Months Covered", f"{months_cov}")

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2,tab3 = st.tabs(["Overview", "Raw Data","AI Insights"])

# ===== Overview Tab =====
with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Monthly Spending Trend")
        if not df_t.empty:
            if not pd.api.types.is_datetime64_any_dtype(df_t[date_col]):
                df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
            df_t["Month"] = df_t[date_col].dt.to_period("M")
            monthly_spend = df_t.groupby("Month")["Amount" if "Amount" in df_t.columns else "amount"].sum()
            monthly_spend.index = monthly_spend.index.astype(str)
            st.line_chart(monthly_spend)
        else:
            st.info("No transactions available to plot.")

    with right:
        st.subheader("Spending by Category")
        cat_col = "Category" if "Category" in df_t.columns else "category"
        amt_col = "Amount" if "Amount" in df_t.columns else "amount"
        if cat_col in df_t.columns:
            cat_spend = df_t.groupby(cat_col)[amt_col].sum().sort_values(ascending=False)
            st.bar_chart(cat_spend)
        else:
            st.info("No category data available.")

# ===== Raw Data Tab =====
with tab2:
    st.subheader("Transactions")
    st.dataframe(df_t, use_container_width=True, height=300)

    st.subheader("Payments")
    st.dataframe(df_p, use_container_width=True, height=250)

    st.subheader("Payments vs Transactions")
    diff = total_spend - total_payments
    st.write(f"**Total Transactions:** ${total_spend:,.2f}")
    st.write(f"**Total Payments:**     ${total_payments:,.2f}")
    st.write(f"**Difference:**         ${diff:,.2f}")

with tab3:
    st.header("🤖 AI Insights - Monthly Spending Forecast (Prophet)")

    if not df_t.empty:
        try:
            forecast, prophet_df = ml.forecast_with_prophet(df_t, months_ahead=4)
            fig = ml.plot_prophet_forecast(forecast, prophet_df)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"⚠️ Forecasting failed: {e}")
    else:
        st.info("📁 Upload a PDF statement or load data from DB to see ML insights.")
        # ---------- ⚠️ AI Insights - Anomaly Detection ----------
st.header("⚠️ AI Insights - Anomalous Transactions")

if not df_t.empty:
    contamination = st.slider("Anomaly rate (contamination)", 0.01, 0.10, 0.03, 0.01,
                              help="Approx % of transactions to flag as anomalies.")
    try:
        anomalies = ml.detect_anomalies_isoforest(df_t, contamination=contamination)
        if anomalies.empty or (~anomalies["is_anomaly"]).all():
            st.success("No suspicious transactions detected at this threshold.")
        else:
            st.dataframe(anomalies[anomalies["is_anomaly"]].head(50), use_container_width=True)
            st.caption("Showing the most suspicious rows first (lowest anomaly_score).")
    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")
else:
    st.info("Upload PDFs or load from DB to run anomaly detection.")


# ---------- Save to DB ----------
if save_to_db:
    try:
        db.create_tables()
        before_tx = len(db.read_all_transactions()) if hasattr(db, "read_all_transactions") else 0
        before_py = len(db.read_all_payments()) if hasattr(db, "read_all_payments") else 0

        ins_tx = db.upsert_transactions(df_t)
        ins_py = db.upsert_payments(df_p)

        after_tx = len(db.read_all_transactions()) if hasattr(db, "read_all_transactions") else before_tx + ins_tx
        after_py = len(db.read_all_payments()) if hasattr(db, "read_all_payments") else before_py + ins_py

        dup_tx = (len(df_t) - ins_tx)
        dup_py = (len(df_p) - ins_py)

        st.success(f"✅ Transactions: inserted {ins_tx}, skipped {dup_tx} duplicates")
        st.success(f"✅ Payments:     inserted {ins_py}, skipped {dup_py} duplicates")

        if hasattr(db, "create_indexes_and_views"):
            db.create_indexes_and_views()
    except Exception as e:
        st.error(f"Database error: {e}")

# ---------- 🤖 AI Insights - Spending Forecast ----------
# ---------- 🤖 AI Insights - Spending Forecast ----------
