import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from math import log10, sin, pi

st.set_page_config(page_title="BTC Wave Model & RSI Shading", layout="wide")

# 1) FRED fetch helper
API_KEY  = "ba5525cc9cf29a46525360eb07c0a7cc"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
@st.cache_data(ttl=3600)
def fetch_fred(series_id, start="2010-07-24", end=None):
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": start
    }
    if end:
        params["observation_end"] = end
    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"]

# 2) Halving dates & avg ms/block
H0_33, H0_66 = datetime(2010,7,24,17,50,42), datetime(2011,8,7,19,49,38)
H1 = datetime(2012,11,28,16,24,38)
H2 = datetime(2016,7,9,18,46,13)
H3 = datetime(2020,5,11,21,23,43)
H4 = datetime(2028,3,26)  # next halving

bm0    = 699768
bm0_33 = 467896
bm0_66 = 591047
bm1    = 542715
bm2    = 576868
bm3    = 593781

# 3) Convert calendar date → halving-time h
def halving_time(t):
    t_ms = t.timestamp() * 1000
    if t < H0_33:
        return t_ms / (bm0 * 70000)
    elif t < H0_66:
        return 0.33 + (t_ms - H0_33.timestamp()*1000) / (bm0_33 * 70000)
    elif t < H1:
        return 0.66 + (t_ms - H0_66.timestamp()*1000) / (bm0_66 * 70000)
    elif t < H2:
        return 1.0 + (t_ms - H1.timestamp()*1000) / (bm1 * 210000)
    elif t < H3:
        return 2.0 + (t_ms - H2.timestamp()*1000) / (bm2 * 210000)
    else:
        return 3.0 + (t_ms - H3.timestamp()*1000) / (bm3 * 210000)

# 4) Trend & envelope
a, b = 1.48, 5.44
def btc_trend(h):
    return 10**(a + b * np.log10(h))

def wave_envelope(h, tr):
    # protect against h<=0
    if h <= 0:
        h = 1e-6
    wcr, width = 0.25, 0.75
    phase = 0.75 / h
    osc   = sin(2*pi*h - phase)
    up    = min(width, osc)
    dn    = max(-width, osc)
    decay = (1 - wcr)**h
    upper  = tr * 10**(decay * (up + width))
    lower  = tr * 10**(decay * (dn - width))
    middle = tr * 10**(decay * osc)
    return upper, lower, middle

# 5) Fetch real BTC data & build index
today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
btc_real  = fetch_fred("CBBTCUSD", "2010-07-24", today_str).resample("D").ffill()
full_idx  = pd.date_range("2010-07-24", H4, freq="D")
price     = btc_real.reindex(full_idx).ffill()

# 6) Compute RSI (14-day)
delta = price.diff()
gain  = delta.where(delta>0,0.0)
loss  = -delta.where(delta<0,0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain/avg_loss
rsi = 100 - (100/(1+rs))

# 7) Compute wave/trend arrays
trends, uppers, lowers, middles = [], [], [], []
for t in full_idx:
    h = halving_time(t)
    tr = btc_trend(h)
    up, dn, mid = wave_envelope(h, tr)
    trends.append(tr)
    uppers.append(up)
    lowers.append(dn)
    middles.append(mid)
trends, uppers, lowers, middles = map(np.array, (trends, uppers, lowers, middles))

# 8) Compute trailing stop (10% under rolling peak)
rolling_peak = price.cummax()
trail_stop   = rolling_peak * 0.90

# 9) Build dynamic RSI shading shapes
thresholds = [
    (80, "yellow",    0.3),
    (85, "orange",    0.25),
    (90, "darkorange",0.2),
    (95, "red",       0.15),
]
shapes = []
for level, color, opacity in thresholds:
    mask = rsi > level
    in_span = False
    for dt, is_hot in zip(rsi.index, mask):
        if is_hot and not in_span:
            span_start = dt
            in_span = True
        elif not is_hot and in_span:
            span_end = prev_dt
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=span_start, x1=span_end,
                y0=0, y1=1,
                fillcolor=color, opacity=opacity,
                layer="below", line_width=0
            ))
            in_span = False
        prev_dt = dt
    if in_span:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=span_start, x1=prev_dt,
            y0=0, y1=1,
            fillcolor=color, opacity=opacity,
            layer="below", line_width=0
        ))

# 10) Build Plotly figure (single y-axis)
fig = go.Figure()
fig.update_layout(shapes=shapes)

# Wave envelope shading
fig.add_trace(go.Scatter(
    x=full_idx, y=uppers, line=dict(color='rgba(0,0,0,0)'), showlegend=False
))
fig.add_trace(go.Scatter(
    x=full_idx, y=lowers, fill='tonexty',
    fillcolor='rgba(0,0,200,0.2)', line=dict(color='rgba(0,0,0,0)'),
    name='Wave Envelope'
))

# Trend & wave middle
fig.add_trace(go.Scatter(
    x=full_idx, y=trends, mode='lines',
    line=dict(color='red', dash='dash'), name='Power‐Law Trend'
))
fig.add_trace(go.Scatter(
    x=full_idx, y=middles, mode='lines',
    line=dict(color='blue'), name='Wave Middle'
))

# BTC price
fig.add_trace(go.Scatter(
    x=full_idx, y=price, mode='lines',
    line=dict(color='black'), name='BTC Price'
))

# Trailing stop shading
fig.add_trace(go.Scatter(
    x=full_idx, y=rolling_peak, mode='lines',
    line=dict(color='rgba(0,0,0,0)'), showlegend=False
))
fig.add_trace(go.Scatter(
    x=full_idx, y=trail_stop, mode='lines',
    line=dict(color='teal'),
    fill='tonexty', fillcolor='rgba(0,128,128,0.2)',
    name='10% Trailing Stop'
))

# Halving & midpoint lines
halvings  = [H1, H2, H3, H4]
midpts    = [halvings[i] + (halvings[i+1] - halvings[i]) / 2 for i in range(3)]
for d in halvings:
    fig.add_vline(x=d, line=dict(color='gray', dash='dot'))
for d in midpts:
    fig.add_vline(x=d, line=dict(color='gray', dash='dash'))

# Layout
y0, y1 = log10(100), log10(200000)
fig.update_layout(
    title="BTC Price vs. Wave Model & RSI Overbought Shading",
    xaxis=dict(title='Date', rangeslider=dict(visible=True), type='date',
               range=['2022-01-01','2026-12-31']),
    yaxis=dict(title='Price (USD, log)', type='log', range=[y0,y1]),
    height=650, hovermode='x unified', dragmode='zoom'
)

st.plotly_chart(fig, use_container_width=True)



