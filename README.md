# 🌽 AgriPrice Advisor

A decision-support tool that forecasts crop prices and tells smallholder farmers **when to sell** to maximise income after storage costs.

## Problem
Smallholder farmers often sell right after harvest, when prices are lowest, because they lack price information and storage options. [Add 1-2 real statistics for your country/crop, with sources.]

## Users
- Smallholder farmers and cooperatives
- Agri-traders and extension officers

## Solution
1. Ingests monthly market price data (CSV)
2. Forecasts prices with Holt-Winters seasonal exponential smoothing
3. Subtracts storage/loss costs to find the best net selling month
4. Shows estimated extra income vs. selling now

## Impact metric
Extra income per season = (best net price - current price) x quantity. Model quality is measured by a 12-month backtest (MAPE).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Business model (sustainability)
- Free for farmers via SMS/WhatsApp summaries
- Paid dashboards for cooperatives, buyers, and agri-lenders
- Partnerships: ministries of agriculture, NGOs, mobile operators

## Roadmap
- [ ] Replace demo data with real market data (FAO GIEWS / FEWS NET / national source)
- [ ] Compare against ARIMA/Prophet
- [ ] SMS/USSD alerts
- [ ] Pilot with a cooperative and record feedback

## Lessons learned
[Fill in after testing: data gaps, forecast limits, adoption barriers.]
