import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="AgriPrice Advisor", page_icon="🌽", layout="wide")
st.title("🌽 AgriPrice Advisor")
st.caption("Forecast crop prices and find the most profitable month to sell.")


@st.cache_data
def demo_data() -> pd.DataFrame:
    """Synthetic monthly prices: upward trend + seasonality (low at harvest, high in lean season)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2019-01-01", periods=84, freq="MS")
    season = 25 * np.sin((dates.month - 4) / 12 * 2 * np.pi)
    trend = np.linspace(300, 420, len(dates))
    return pd.DataFrame({"date": dates, "price": (trend + season + rng.normal(0, 10, len(dates))).round(2)})


def fit_forecast(series: pd.Series, horizon: int):
    model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit()
    return model.forecast(horizon)


# ---------- Sidebar inputs ----------
st.sidebar.header("Your inputs")
crop = st.sidebar.text_input("Crop", "Maize")
file = st.sidebar.file_uploader("Price data CSV (columns: date, price)", type="csv")
qty = st.sidebar.number_input("Quantity to sell (kg)", 100, 1_000_000, 5000, step=100)
horizon = st.sidebar.slider("Forecast horizon (months)", 3, 12, 8)
storage = st.sidebar.slider("Storage + loss cost (% of value per month)", 0.0, 5.0, 1.5, 0.1) / 100
unit = st.sidebar.text_input("Currency per kg/ton label", "USD per ton")

df = pd.read_csv(file, parse_dates=["date"]) if file else demo_data()
if not file:
    st.info("Showing demo data. Upload your own market price CSV in the sidebar (e.g. FAO/FEWS NET/national market data).")

series = df.set_index("date")["price"].asfreq("MS").interpolate()
if len(series) < 36:
    st.error("Need at least 36 monthly observations for seasonal forecasting.")
    st.stop()

# ---------- Backtest (hold out last 12 months) ----------
train, test = series[:-12], series[-12:]
mape = (abs(fit_forecast(train, 12).values - test.values) / test.values).mean() * 100

# ---------- Forecast & recommendation ----------
fc = fit_forecast(series, horizon)
now_price = series.iloc[-1]
months_ahead = np.arange(1, horizon + 1)
net = fc.values * (1 - storage * months_ahead)  # price after storage/loss costs
best_i = int(np.argmax(net))
best_net = net[best_i]
uplift_pct = (best_net - now_price) / now_price * 100

if best_net > now_price:
    advice = f"Hold and sell in **{fc.index[best_i]:%B %Y}**"
else:
    advice = "**Sell now**: waiting does not beat storage costs"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest price", f"{now_price:,.0f}")
c2.metric("Best net price", f"{best_net:,.0f}", f"{uplift_pct:+.1f}%")
c3.metric("Extra income (est.)", f"{max(best_net - now_price, 0) * qty / 1000:,.0f}")
c4.metric("Backtest error (MAPE)", f"{mape:.1f}%")
st.subheader(f"Recommendation for {crop}: {advice}")

fig = go.Figure()
fig.add_scatter(x=series.index, y=series.values, name="Historical", line=dict(color="#2e7d32"))
fig.add_scatter(x=fc.index, y=fc.values, name="Forecast", line=dict(color="#f9a825", dash="dash"))
fig.add_scatter(x=fc.index, y=net, name="Net of storage costs", line=dict(color="#c62828", dash="dot"))
fig.update_layout(yaxis_title=unit, height=420, margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

st.caption("Extra income assumes the quantity is priced per ton. Forecasts are estimates, not guarantees.")
