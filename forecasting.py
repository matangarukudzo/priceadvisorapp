import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_prices(src) -> pd.DataFrame:
    """Load WFP/HDX-format or simple (date, price[, commodity, market, unit]) CSVs."""
    df = pd.read_csv(src)
    df.columns = [c.strip().lower().lstrip("#") for c in df.columns]
    df = df[~df["date"].astype(str).str.startswith("#")]  # HDX hashtag row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    col = "usdprice" if "usdprice" in df.columns else "price"  # USD avoids ZWL/ZiG distortion
    df["price"] = pd.to_numeric(df[col], errors="coerce")
    for c, default in [("commodity", "Commodity"), ("market", "All markets"), ("unit", "KG")]:
        if c not in df.columns:
            df[c] = default
    return df.dropna(subset=["date", "price"])[["date", "commodity", "market", "unit", "price"]]


def monthly_series(df, commodity, market="All markets") -> pd.Series:
    d = df[df.commodity == commodity]
    if market != "All markets":
        d = d[d.market == market]
    s = d.groupby(d.date.dt.to_period("M").dt.to_timestamp())["price"].mean()
    s = s.asfreq("MS").interpolate(limit=3)
    if s.isna().any():  # keep the most recent gap-free run
        s = s[s[s.isna()].index[-1]:].iloc[1:]
    return s


def _hw(y, h):
    f = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()
    return f.forecast(h), (y - f.fittedvalues).std()


def _sarima(y, h):
    f = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)).fit(disp=False)
    return f.forecast(h), f.resid.iloc[13:].std()


def _naive(y, h):
    last = y.iloc[-12:].values
    idx = pd.date_range(y.index[-1], periods=h + 1, freq="MS")[1:]
    return pd.Series([last[i % 12] for i in range(h)], index=idx), y.diff(12).dropna().std()


MODELS = {"Holt-Winters": _hw, "Seasonal ARIMA": _sarima, "Seasonal naive (baseline)": _naive}


def compare_models(y: pd.Series, h: int):
    """Backtest each model on the last 12 months, refit the winner on all data."""
    train, test = y[:-12], y[-12:]
    scores = {}
    for name, fn in MODELS.items():
        try:
            scores[name] = float((abs(fn(train, 12)[0].values - test.values) / test.values).mean() * 100)
        except Exception:
            continue
    best = min(scores, key=scores.get)
    fc, sigma = MODELS[best](y, h)
    steps = np.sqrt(np.arange(1, h + 1))
    return {
        "scores": scores, "best": best, "forecast": fc,
        "lower": (fc - 1.28 * sigma * steps).clip(lower=0),  # ~80% interval
        "upper": fc + 1.28 * sigma * steps,
    }
