# ml_models.py
import pandas as pd
import matplotlib.pyplot as plt
import io
from prophet import Prophet
import numpy as np
from sklearn.ensemble import IsolationForest
def forecast_with_prophet(df_transactions: pd.DataFrame, months_ahead: int = 3):
    df = df_transactions.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if "trans_date" not in df.columns or "amount" not in df.columns:
        raise ValueError("❌ DataFrame must have 'Trans date' and 'Amount' columns")

    # ✅ Clean & prepare
    df["trans_date"] = pd.to_datetime(df["trans_date"])
    monthly = df.groupby(df["trans_date"].dt.to_period("M"))["amount"].sum().reset_index()
    monthly["trans_date"] = monthly["trans_date"].dt.to_timestamp()

    prophet_df = monthly.rename(columns={"trans_date": "ds", "amount": "y"})

    # ✅ Clip negative or extreme outliers (helps stabilize forecast)
# Smooth historical values to remove random spikes
    prophet_df["y"] = prophet_df["y"].rolling(window=3, min_periods=1).mean()

    # ✅ Tune Prophet
    model = Prophet(
        growth="flat",
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False,
        changepoint_prior_scale=0.05,   # smoother trend line
        seasonality_mode="multiplicative"  # better for financial data
    )

    model.fit(prophet_df)

    # ✅ Forecast future months
    future = model.make_future_dataframe(periods=months_ahead, freq='MS')
    forecast = model.predict(future)

    # Prevent negative forecast
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    return forecast, prophet_df


def plot_prophet_forecast(forecast, prophet_df):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Historical data
    ax.plot(prophet_df['ds'], prophet_df['y'], label='Historical', marker='o', color='blue')

    # Forecast
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--', color='orange')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2)

    ax.set_title("📈 Monthly Spending Forecast (Prophet)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Spending ($)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return fig

# -------------------------------
# ⚠️ Anomaly Detection (IsolationForest)
# -------------------------------
import numpy as np
from sklearn.ensemble import IsolationForest

def _normalize_tx_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts either Title-Case or snake_case; returns a clean copy."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["trans_date","description","category","amount"])
    d = df.copy()
    d.columns = [c.lower().replace(" ", "_") for c in d.columns]
    # Required columns: trans_date, amount; optional: category, description
    if "trans_date" not in d.columns or "amount" not in d.columns:
        # try to map common variants if they exist
        if "trans_date" not in d.columns and "transdate" in d.columns:
            d = d.rename(columns={"transdate": "trans_date"})
        if "amount" not in d.columns and "amt" in d.columns:
            d = d.rename(columns={"amt": "amount"})
    if "trans_date" not in d.columns or "amount" not in d.columns:
        raise ValueError("Transactions must have 'Trans date'/'Amount' (or 'trans_date'/'amount').")
    d["trans_date"] = pd.to_datetime(d["trans_date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    if "category" not in d.columns: d["category"] = "Uncategorized"
    if "description" not in d.columns: d["description"] = ""
    d = d.dropna(subset=["trans_date", "amount"])
    return d

def _featureize_for_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple, explainable numeric features IsolationForest can use.
    - amount: raw amount
    - log_amount: log(1+amount) to damp spikes
    - day_of_month / day_of_week / month: calendar context
    - cat_median_delta: amount minus median for that category
    - merch_median_delta: amount minus median for that merchant (derived from description)
    """
    d = df.copy()
    d["merchant"] = d["description"].str.upper().str.extract(r"^(.{0,30})", expand=False).fillna("")
    d["day_of_month"] = d["trans_date"].dt.day
    d["day_of_week"]  = d["trans_date"].dt.dayofweek
    d["month"]        = d["trans_date"].dt.month

    # medians by category & merchant (robust against outliers)
    cat_med = d.groupby("category")["amount"].median().rename("cat_median")
    mch_med = d.groupby("merchant")["amount"].median().rename("merch_median")
    d = d.merge(cat_med, on="category", how="left")
    d = d.merge(mch_med, on="merchant", how="left")

    d["cat_median"].fillna(d["amount"].median(), inplace=True)
    d["merch_median"].fillna(d["amount"].median(), inplace=True)

    d["cat_median_delta"]   = d["amount"] - d["cat_median"]
    d["merch_median_delta"] = d["amount"] - d["merch_median"]
    d["log_amount"]         = np.log1p(d["amount"])  # log(1+x)

    feature_cols = [
        "amount", "log_amount",
        "day_of_month", "day_of_week", "month",
        "cat_median_delta", "merch_median_delta",
    ]
    return d, feature_cols

def detect_anomalies_isoforest(
    df_transactions: pd.DataFrame,
    contamination: float = 0.03,
    n_estimators: int = 200,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Returns suspicious transactions with IsolationForest score.
    - contamination ~ % of points to mark anomalous (1-5% typical).
    """
    d = _normalize_tx_columns(df_transactions)
    if len(d) < 30:  # not enough data to learn normal pattern
        return pd.DataFrame(columns=list(d.columns) + ["anomaly_score","is_anomaly"])

    d_feat, feature_cols = _featureize_for_anomaly(d)
    X = d_feat[feature_cols].astype(float).fillna(0.0).values

    # Train on *all* data (unsupervised)
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    iso.fit(X)

    # Decision_function: lower is more anomalous; negatives usually anomalies
    scores = iso.decision_function(X)  # higher = more normal
    preds  = iso.predict(X)            # 1 = normal, -1 = anomaly

    out = d_feat.copy()
    out["anomaly_score"] = scores
    out["is_anomaly"]    = (preds == -1)

    # Sort: most suspicious first (lowest score)
    out = out.sort_values("anomaly_score")
    # Return only meaningful columns for UI
    keep = ["trans_date","description","category","amount","anomaly_score","is_anomaly"]
    return out[keep]
