import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from forecasting import load_prices, monthly_series, compare_models

st.set_page_config(page_title="AgriPrice Advisor Zimbabwe", page_icon="ðŸŒ½", layout="wide")
st.title("ðŸŒ½ AgriPrice Advisor: Zimbabwe")
st.caption("Forecast crop prices and find the most profitable month to sell.")

DATA_FILE = "data/wfp_food_prices_zwe.csv"


@st.cache_data
def demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-01-01", periods=96, freq="MS")
    rows = []
    for crop, base in [("Maize (demo)", 0.40), ("Sorghum (demo)", 0.45), ("Wheat (demo)", 0.60)]:
        p = base * (1 + 0.15 * np.sin((dates.month - 4) / 12 * 2 * np.pi)) + rng.normal(0, 0.015, len(dates))
        rows.append(pd.DataFrame({"date": dates, "commodity": crop, "market": "All markets", "unit": "KG", "price": p}))
    return pd.concat(rows)


# ---------- Data source ----------
st.sidebar.header("1. Data")
upload = st.sidebar.file_uploader("Upload CSV (WFP format, or date + price [+ commodity])", type="csv")
if upload:
    df = load_prices(upload)
elif os.path.exists(DATA_FILE):
    df = load_prices(DATA_FILE)
else:
    df = demo_data()
    st.info("Demo data shown. Add the WFP Zimbabwe file to `data/` (see data/README.md) or upload a CSV. "
            "For tobacco, upload auction prices (e.g. TIMB), since WFP does not track it.")

st.sidebar.header("2. Crop & market")
crop = st.sidebar.selectbox("Commodity", sorted(df.commodity.unique()))
markets = ["All markets"] + sorted(df[df.commodity == crop].market.unique())
market = st.sidebar.selectbox("Market", markets)

st.sidebar.header("3. Your situation")
qty = st.sidebar.number_input("Quantity to sell (kg)", 100, 10_000_000, 5000, step=100)
horizon = st.sidebar.slider("Forecast horizon (months)", 3, 12, 8)
storage = st.sidebar.slider("Storage + loss cost (% per month)", 0.0, 5.0, 1.5, 0.1) / 100

series = monthly_series(df, crop, market)
if len(series) < 36:
    st.error(f"Only {len(series)} usable months for this selection; need at least 36. Try 'All markets'.")
    st.stop()

unit = df[df.commodity == crop].unit.iloc[0]
per_kg = 1 / 1000 if any(k in str(unit).upper() for k in ["MT", "TON"]) else 1

res = compare_models(series, horizon)
fc, lo, up = res["forecast"], res["lower"], res["upper"]
now = series.iloc[-1]
k = np.arange(1, horizon + 1)
net, net_lo = fc.values * (1 - storage * k), lo.values * (1 - storage * k)
i = int(np.argmax(net))
gain = max(net[i] - now, 0) * qty * per_kg

# ---------- Output ----------
if net[i] > now:
    st.subheader(f"{crop}: hold and sell in {fc.index[i]:%B %Y}")
else:
    st.subheader(f"{crop}: sell now. Waiting does not beat storage costs")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest price (USD)", f"{now:,.3f}")
c2.metric("Best net price", f"{net[i]:,.3f}", f"{(net[i] - now) / now * 100:+.1f}%")
c3.metric("Est. extra income (USD)", f"{gain:,.0f}")
c4.metric("Best model error", f"{res['scores'][res['best']]:.1f}%")

if net_lo[i] < now:
    st.warning("Risk: in the pessimistic case (lower band) holding would earn less than selling now.")

fig = go.Figure()
fig.add_scatter(x=series.index, y=series.values, name="Historical", line=dict(color="#2e7d32"))
fig.add_scatter(x=fc.index, y=up, line=dict(width=0), showlegend=False)
fig.add_scatter(x=fc.index, y=lo, fill="tonexty", fillcolor="rgba(249,168,37,0.25)", line=dict(width=0), name="80% range")
fig.add_scatter(x=fc.index, y=fc.values, name=f"Forecast ({res['best']})", line=dict(color="#f9a825", dash="dash"))
fig.add_scatter(x=fc.index, y=net, name="Net of storage costs", line=dict(color="#c62828", dash="dot"))
fig.update_layout(yaxis_title=f"USD per {unit}", height=420, margin=dict(t=20), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model comparison (12-month backtest, lower is better)"):
    st.dataframe(pd.DataFrame({"Model": res["scores"].keys(), "MAPE %": [round(v, 1) for v in res["scores"].values()]}),
                 hide_index=True)
    st.caption("The best model is selected automatically. Prices are in USD to avoid ZWL/ZiG currency distortions. "
               "Forecasts are estimates, not guarantees.")

