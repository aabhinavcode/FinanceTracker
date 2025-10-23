# ml_models.py
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest


# --------------------------
# Helpers (column robustness)
# --------------------------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standard names: trans_date, amount, category, description."""
    d = df.copy()
    d.columns = [c.lower().replace(" ", "_") for c in d.columns]
    # map common variants
    rename = {}
    if "trans date" in df.columns:  # just in case someone passes original caps
        rename["Trans date"] = "trans_date"
    if "amount" not in d.columns and "Amount" in df.columns:
        rename["Amount"] = "amount"
    if "trans_date" not in d.columns and "trans date" in d.columns:
        rename["trans date"] = "trans_date"
    # apply safe rename
    d = d.rename(columns={
        "Trans date": "trans_date",
        "Amount": "amount",
        "Category": "category",
        "Description": "description",
    })
    # after rename, enforce types / defaults
    if "trans_date" not in d.columns or "amount" not in d.columns:
        # last try (snake already)
        if "trans date" in d.columns:
            d = d.rename(columns={"trans date": "trans_date"})
        if "Amount" in d.columns:
            d = d.rename(columns={"Amount": "amount"})
    if "category" not in d.columns: d["category"] = "Uncategorized"
    if "description" not in d.columns: d["description"] = ""

    d["trans_date"] = pd.to_datetime(d["trans_date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d = d.dropna(subset=["trans_date", "amount"])
    return d


# --------------------------
# Prophet Forecast (simple)
# --------------------------
def forecast_with_prophet(df_transactions: pd.DataFrame, months_ahead: int = 3):
    d = _std_cols(df_transactions)
    monthly = d.groupby(d["trans_date"].dt.to_period("M"))["amount"].sum().reset_index()
    monthly["trans_date"] = monthly["trans_date"].dt.to_timestamp()

    prophet_df = monthly.rename(columns={"trans_date": "ds", "amount": "y"})
    # slight smoothing for stability
    prophet_df["y"] = prophet_df["y"].rolling(window=3, min_periods=1).mean()

    m = Prophet(
        growth="flat",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
    )
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=months_ahead, freq="MS")
    fcst = m.predict(future)
    fcst["yhat"] = fcst["yhat"].clip(lower=0)
    fcst["yhat_lower"] = fcst["yhat_lower"].clip(lower=0)
    fcst["yhat_upper"] = fcst["yhat_upper"].clip(lower=0)
    return fcst, prophet_df


def plot_prophet_forecast(forecast: pd.DataFrame, hist_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_df["ds"], hist_df["y"], marker="o", label="Historical")
    ax.plot(forecast["ds"], forecast["yhat"], linestyle="--", label="Forecast")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2)
    ax.set_title("Monthly Spending Forecast (Prophet)")
    ax.set_xlabel("Month"); ax.set_ylabel("Total Spending ($)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0); return fig


# --------------------------
# IsolationForest Anomalies
# --------------------------
def _features_for_anomaly(d: pd.DataFrame):
    x = d.copy()
    x["merchant"] = x["description"].str.upper().str.extract(r"^(.{0,30})", expand=False).fillna("")
    x["day_of_month"] = x["trans_date"].dt.day
    x["day_of_week"]  = x["trans_date"].dt.dayofweek
    x["month"]        = x["trans_date"].dt.month

    cat_med = x.groupby("category")["amount"].median().rename("cat_median")
    mch_med = x.groupby("merchant")["amount"].median().rename("merch_median")
    x = x.merge(cat_med, on="category", how="left").merge(mch_med, on="merchant", how="left")
    gmed = x["amount"].median()
    x["cat_median"].fillna(gmed, inplace=True)
    x["merch_median"].fillna(gmed, inplace=True)
    x["cat_median_delta"]   = x["amount"] - x["cat_median"]
    x["merch_median_delta"] = x["amount"] - x["merch_median"]
    x["log_amount"]         = np.log1p(x["amount"])

    feat_cols = ["amount","log_amount","day_of_month","day_of_week","month",
                 "cat_median_delta","merch_median_delta"]
    return x, feat_cols


def detect_anomalies_isoforest(df_transactions: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    d = _std_cols(df_transactions)
    if len(d) < 30:
        return pd.DataFrame(columns=["trans_date","description","category","amount","anomaly_score","is_anomaly"])
    Xdf, cols = _features_for_anomaly(d)
    X = Xdf[cols].astype(float).fillna(0.0).values
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X)
    Xdf["anomaly_score"] = iso.decision_function(X)      # higher = more normal
    Xdf["is_anomaly"] = (iso.predict(X) == -1)
    out = Xdf.sort_values("anomaly_score")
    return out[["trans_date","description","category","amount","anomaly_score","is_anomaly"]]


def anomaly_quick_kpis(anom_df: pd.DataFrame, full_df: pd.DataFrame):
    """Return a small dict for display; accepts either Amount/amount."""
    if anom_df is None or anom_df.empty or full_df is None or full_df.empty:
        return dict(flagged=0, pct_rows=0.0, flagged_amount=0.0, pct_amount=0.0)

    # standardize amount in the copy of full_df for robust math
    f = _std_cols(full_df)
    flagged = anom_df[anom_df["is_anomaly"]]
    total_amt = np.nansum(pd.to_numeric(f["amount"], errors="coerce"))
    flagged_amt = np.nansum(pd.to_numeric(flagged["amount"], errors="coerce"))

    return dict(
        flagged=len(flagged),
        pct_rows=(len(flagged) / len(f) * 100.0) if len(f) else 0.0,
        flagged_amount=flagged_amt,
        pct_amount=(flagged_amt / total_amt * 100.0) if total_amt else 0.0,
    )
