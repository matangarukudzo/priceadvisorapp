import os
import base64
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from forecasting import load_prices, monthly_series, compare_models

ICON = Image.open("assets/favicon.png") if os.path.exists("assets/favicon.png") else "🌽"
st.set_page_config(page_title="AgriPrice Advisor Zimbabwe", page_icon=ICON, layout="wide")

DATA_FILE = "data/wfp_food_prices_zwe.csv"

# ---------- Translations (have a native speaker review Shona/Ndebele before publishing) ----------
T = {
    "English": dict(
        tag="Know when to sell your maize, sorghum, wheat and more, and earn more from every harvest.",
        data="1. Data", upload="Upload CSV", crop_h="2. Crop & market", commodity="Commodity", market="Market",
        all="All markets", sit="3. Your situation", qty="Quantity to sell (kg)", horizon="Forecast horizon (months)",
        storage="Storage + loss cost (% per month)", hold="hold and sell in", sell="sell now",
        hold_sub="Expected net gain of about {p:.1f}% after storage costs.",
        sell_sub="Waiting does not beat storage costs for this crop and market.",
        latest="Latest price (USD)", best="Best net price", extra="Est. extra income (USD)",
        err="Best model error", risk="Risk: in the pessimistic case, holding could earn less than selling now.",
        share="Share on WhatsApp", dl="Download summary", models="Model comparison (12-month backtest)"),
    "Shona": dict(
        tag="Ziva nguva yekutengesa chibage, mapfunde, gorosi nezvimwe, uwane mari yakawanda pachirimwa chimwe nechimwe.",
        data="1. Data", upload="Isa faira reCSV", crop_h="2. Chirimwa nemusika", commodity="Chirimwa", market="Musika",
        all="Misika yese", sit="3. Mamiriro ako", qty="Huwandu hwekutengesa (kg)", horizon="Mwedzi yekufanotaura",
        storage="Mutengo wekuchengeta (% pamwedzi)", hold="chengeta utengese muna", sell="tengesa zvino",
        hold_sub="Mari inofungidzirwa kuwedzera ndeye {p:.1f}% mushure mekuchengeta.",
        sell_sub="Kumirira hakuzokunda mutengo wekuchengeta.",
        latest="Mutengo wazvino (USD)", best="Mutengo wakanakisa", extra="Mari yekuwedzera (USD)",
        err="Kukanganisa kwemodhi", risk="Njodzi: kana mitengo yakaderera, kuchengeta kunogona kuwana mari shoma kudarika kutengesa zvino.",
        share="Tumira paWhatsApp", dl="Dhawunirodha pfupiso", models="Kuenzanisa kwemodhi"),
    "Ndebele": dict(
        tag="Yazi isikhathi sokuthengisa ummbila, amabele lokunye, uzuze imali eyengeziweyo kuvuno ngalunye.",
        data="1. Idatha", upload="Layisha ifayela le-CSV", crop_h="2. Isitshalo lemakethe", commodity="Isitshalo",
        market="Imakethe", all="Wonke amakethe", sit="3. Isimo sakho", qty="Inani lokuthengisa (kg)",
        horizon="Inyanga zokuqagela", storage="Izindleko zokulondoloza (% ngenyanga)", hold="gcina uthengise ngo",
        sell="thengisa manje", hold_sub="Inzuzo elindelekileyo ngu-{p:.1f}% emva kwezindleko zokulondoloza.",
        sell_sub="Ukulinda akudluli izindleko zokulondoloza.",
        latest="Intengo yamuva (USD)", best="Intengo engcono kakhulu", extra="Imali eyengeziweyo (USD)",
        err="Iphutha lemodeli", risk="Ingozi: ukugcina kungazuzisa kancane kulokuthengisa manje.",
        share="Thumela ngeWhatsApp", dl="Landa isifinyezo", models="Ukuqhathanisa kwemodeli"),
}

# ---------- Top controls ----------
lang = st.sidebar.selectbox("🌍 Language / Mutauro / Ulimi", list(T))
THEMES = {
    "🌅 Sunset": dict(page="linear-gradient(180deg,#fff8e1,#ffe9c7)", side="#fff3e0", card="#ffffff", border="#f3d9a8",
                     txt="#3e2f1c", hero="linear-gradient(135deg,#ef6c00,#f9a825 55%,#7cb342)", dark=False),
    "🌿 Forest": dict(page="linear-gradient(180deg,#f4faf0,#e3f1dc)", side="#e8f5e9", card="#ffffff", border="#cfe3c8",
                     txt="#1f2d1f", hero="linear-gradient(135deg,#1b5e20,#43a047 60%,#9ccc65)", dark=False),
    "🌙 Night": dict(page="linear-gradient(180deg,#0f1a12,#14231a)", side="#0b140e", card="#1b2b20", border="#2c4636",
                    txt="#e8f0e6", hero="linear-gradient(135deg,#0d3b1e,#1f7a3a 60%,#4c8a2b)", dark=True),
}
theme = THEMES[st.sidebar.selectbox("🎨 Theme", list(THEMES))]
dark = theme["dark"]
t = T[lang]

st.markdown(f"""<style>
.stApp {{background: {theme['page']} !important;}}
header[data-testid="stHeader"] {{background: transparent !important;}}
section[data-testid="stSidebar"] > div {{background: {theme['side']} !important;}}
.block-container {{padding-top: 1.2rem; max-width: 1100px;}}
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp span,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{color: {theme['txt']};}}
[data-testid="stImage"] img {{border-radius: 16px;}}
[data-testid="stMetric"] {{background: {theme['card']}; border: 1px solid {theme['border']}; border-radius: 14px;
       padding: 14px 16px; box-shadow: 0 2px 8px rgba(0,0,0,.10);}}
.hero {{background: {theme['hero']}; border-radius: 16px; padding: 20px 24px; margin: 10px 0 18px;}}
.hero h1 {{margin: 0; font-size: 1.7rem; color: #fff !important;}}
.hero p {{margin: 6px 0 0; font-size: 1rem; color: #fff !important;}}
.rec {{border-radius: 16px; padding: 18px 22px; margin: 6px 0 16px; font-size: 1.15rem; border-left: 8px solid;}}
.rec.hold {{background: #e8f5e9; border-color: #2e7d32; color: #1b5e20 !important;}}
.rec.sell {{background: #fff3e0; border-color: #ef6c00; color: #8a3f00 !important;}}
.rec small {{display: block; font-size: .9rem; margin-top: 4px;}}
</style>""", unsafe_allow_html=True)

if os.path.exists("assets/maize_bg.jpg"):
    st.image("assets/maize_bg.jpg", use_container_width=True)
st.markdown(f"""<div class="hero"><h1>🌽 AgriPrice Advisor · Zimbabwe</h1><p>{t['tag']}</p></div>""",
            unsafe_allow_html=True)


@st.cache_data
def demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-01-01", periods=96, freq="MS")
    rows = []
    for crop, base in [("Maize (demo)", 0.40), ("Sorghum (demo)", 0.45), ("Wheat (demo)", 0.60)]:
        p = base * (1 + 0.15 * np.sin((dates.month - 4) / 12 * 2 * np.pi)) + rng.normal(0, 0.015, len(dates))
        rows.append(pd.DataFrame({"date": dates, "commodity": crop, "market": "All markets", "unit": "KG", "price": p}))
    return pd.concat(rows)


# ---------- Data ----------
st.sidebar.header(t["data"])
upload = st.sidebar.file_uploader(t["upload"], type="csv")
if upload:
    df = load_prices(upload)
elif os.path.exists(DATA_FILE):
    df = load_prices(DATA_FILE)
else:
    df = demo_data()
    st.info("Demo data shown. Add the WFP Zimbabwe file to `data/` (see data/README.md) or upload a CSV. "
            "For tobacco, upload auction prices (e.g. TIMB), since WFP does not track it.")

st.sidebar.header(t["crop_h"])
crop = st.sidebar.selectbox(t["commodity"], sorted(df.commodity.unique()))
markets = ["All markets"] + sorted(df[df.commodity == crop].market.unique())
market = st.sidebar.selectbox(t["market"], markets, format_func=lambda m: t["all"] if m == "All markets" else m)

st.sidebar.header(t["sit"])
qty = st.sidebar.number_input(t["qty"], 100, 10_000_000, 5000, step=100)
horizon = st.sidebar.slider(t["horizon"], 3, 12, 8)
storage = st.sidebar.slider(t["storage"], 0.0, 5.0, 1.5, 0.1) / 100

series = monthly_series(df, crop, market)
if len(series) < 36:
    st.error(f"Only {len(series)} usable months for this selection; need at least 36.")
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
holding = net[i] > now
pct = (net[i] - now) / now * 100

# ---------- Recommendation ----------
if holding:
    head = f"{crop}: {t['hold']} {fc.index[i]:%B %Y}"
    sub, cls, icon = t["hold_sub"].format(p=pct), "hold", "✅"
else:
    head, sub, cls, icon = f"{crop}: {t['sell']}", t["sell_sub"], "sell", "⚡"
st.markdown(f'<div class="rec {cls}">{icon} <b>{head}</b><small>{sub}</small></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric(t["latest"], f"{now:,.3f}")
c2.metric(t["best"], f"{net[i]:,.3f}", f"{pct:+.1f}%")
c3.metric(t["extra"], f"{gain:,.0f}")
c4.metric(t["err"], f"{res['scores'][res['best']]:.1f}%")
if net_lo[i] < now:
    st.warning(t["risk"])

# ---------- Chart ----------
fig = go.Figure()
fig.add_scatter(x=series.index, y=series.values, name="Historical", line=dict(color="#2e7d32"))
fig.add_scatter(x=fc.index, y=up, line=dict(width=0), showlegend=False)
fig.add_scatter(x=fc.index, y=lo, fill="tonexty", fillcolor="rgba(249,168,37,0.25)", line=dict(width=0), name="80% range")
fig.add_scatter(x=fc.index, y=fc.values, name=f"Forecast ({res['best']})", line=dict(color="#f9a825", dash="dash"))
fig.add_scatter(x=fc.index, y=net, name="Net of storage costs", line=dict(color="#ef5350", dash="dot"))
fig.update_layout(yaxis_title=f"USD per {unit}", height=420, margin=dict(t=10, l=10, r=10),
                  legend=dict(orientation="h", y=-0.2), template="plotly_dark" if dark else "plotly_white",
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ---------- Share & download ----------
summary = (f"AgriPrice Advisor Zimbabwe\n{head}\n{sub}\n"
           f"{t['latest']}: {now:,.3f}\n{t['best']}: {net[i]:,.3f}\n{t['extra']}: {gain:,.0f}\n"
           f"Quantity: {qty:,} kg | Storage cost: {storage * 100:.1f}%/month")
s1, s2 = st.columns(2)
s1.link_button("📲 " + t["share"], "https://wa.me/?text=" + urllib.parse.quote(summary), use_container_width=True)
s2.download_button("⬇️ " + t["dl"], summary, file_name=f"agriprice_{crop}.txt", use_container_width=True)

with st.expander(t["models"]):
    st.dataframe(pd.DataFrame({"Model": res["scores"].keys(), "MAPE %": [round(v, 1) for v in res["scores"].values()]}),
                 hide_index=True)
    st.caption("Prices are in USD to avoid ZWL/ZiG currency distortions. Forecasts are estimates, not guarantees.")
