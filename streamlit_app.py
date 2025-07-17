# Streamlit App: BTC Wave Model + RSI Subplot with RSI-Only Range Slider & Linear Price Axis

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from math import sin, pi

st.set_page_config(page_title="BTC & RSI Overview", layout="wide")

# 1) FRED helper
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
    df = pd.DataFrame(r.json()["observations"])
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"]

# 2) Halving & wave definitions
H0_33 = datetime(2010,7,24,17,50,42)
H0_66 = datetime(2011,8,7,19,49,38)
H1    = datetime(2012,11,28,16,24,38)
H2    = datetime(2016,7,9,18,46,13)
H3    = datetime(2020,5,11,21,23,43)
H4    = datetime(2028,3,26)
bm_vals = [699768,467896,591047,542715,576868,593781]
a, b = 1.48, 5.44

def halving_time(t):
    ms = t.timestamp()*1000
    if t < H0_33:
        return ms/(bm_vals[0]*70000)
    elif t < H0_66:
        return 0.33 + (ms - H0_33.timestamp()*1000)/(bm_vals[1]*70000)
    elif t < H1:
        return 0.66 + (ms - H0_66.timestamp()*1000)/(bm_vals[2]*70000)
    elif t < H2:
        return 1.0 + (ms - H1.timestamp()*1000)/(bm_vals[3]*210000)
    elif t < H3:
        return 2.0 + (ms - H2.timestamp()*1000)/(bm_vals[4]*210000)
    else:
        return 3.0 + (ms - H3.timestamp()*1000)/(bm_vals[5]*210000)

def btc_trend(h):
    return 10**(a + b * np.log10(max(h, 1e-6)))

def wave_envelope(h, tr):
    if h <= 0: 
        h = 1e-6
    wcr, wd = 0.25, 0.75
    phase = 0.75 / h
    osc   = sin(2*pi*h - phase)
    up    = min(wd, osc)
    dn    = max(-wd, osc)
    decay = (1-wcr)**h
    return (
        tr * 10**(decay*(up+wd)),
        tr * 10**(decay*(dn-wd)),
        tr * 10**(decay*osc)
    )

# 3) Fetch BTC data
today = pd.Timestamp.today().strftime("%Y-%m-%d")
btc   = fetch_fred("CBBTCUSD", "2010-07-24", today).resample("D").ffill()
idx   = pd.date_range("2010-07-24", H4, freq="D")
price = btc.reindex(idx).ffill()

# 4) Compute RSI (14-day)
delta    = price.diff()
gain     = delta.where(delta>0, 0.0)
loss     = -delta.where(delta<0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs       = avg_gain / avg_loss
rsi      = 100 - (100 / (1 + rs))

# 5) Compute model arrays
tr_arr, up_arr, dn_arr, mid_arr = [], [], [], []
for t in idx:
    h  = halving_time(t)
    tr = btc_trend(h)
    up, dn, mid = wave_envelope(h, tr)
    tr_arr.append(tr)
    up_arr.append(up)
    dn_arr.append(dn)
    mid_arr.append(mid)
tr_arr, up_arr, dn_arr, mid_arr = map(np.array, (tr_arr, up_arr, dn_arr, mid_arr))

# 6) Trailing stop
peak = price.cummax()
stop = peak * 0.90

# 7) Identify RSI spans
thresholds = [(80,"yellow",0.2),(85,"orange",0.15),(90,"darkorange",0.1),(95,"red",0.05)]
rsi_spans = []
for lvl, col, op in thresholds:
    mask = rsi > lvl
    in_span = False
    for dt, hot in zip(rsi.index, mask):
        if hot and not in_span:
            start, in_span = dt, True
        elif not hot and in_span:
            rsi_spans.append((start, prev_dt, col, op))
            in_span = False
        prev_dt = dt
    if in_span:
        rsi_spans.append((start, prev_dt, col, op))

# 8) Build subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75,0.25], vertical_spacing=0.02)

# Main chart: vertical RSI shading + wave + price + stop
for s, e, col, op in rsi_spans:
    fig.add_vrect(x0=s, x1=e, y0=0, y1=1,
                  fillcolor=col, opacity=op,
                  layer="below", row=1, col=1, yref="paper")
fig.add_trace(go.Scatter(x=idx, y=up_arr,
                         line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=dn_arr, fill='tonexty',
                         fillcolor='rgba(0,0,200,0.2)', line=dict(color='rgba(0,0,0,0)'),
                         name='Wave Env'), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=tr_arr, mode='lines',
                         line=dict(color='red', dash='dash'), name='Trend'), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=mid_arr, mode='lines',
                         line=dict(color='blue'), name='Wave Mid'), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=price, mode='lines',
                         line=dict(color='black'), name='BTC Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=stop, mode='lines', line=dict(color='teal'),
                         fill='tonexty', fillcolor='rgba(0,128,128,0.2)',
                         name='10% Stop'), row=1, col=1)

# Halving & midpoints
halvings  = [H1, H2, H3, H4]
midpoints = [halvings[i] + (halvings[i+1] - halvings[i]) / 2 for i in range(3)]
for d in halvings:
    fig.add_vline(x=d, line=dict(color='gray', dash='dot'))
for m in midpoints:
    fig.add_vline(x=m, line=dict(color='gray', dash='dash'))

# RSI subplot with its own rangeslider
fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines',
                         line=dict(color='black'), name='RSI'), row=2, col=1)

# Layout: disable main slider, enable RSI slider
fig.update_layout(
    height=700,
    hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=False), type='date'),
    xaxis2=dict(rangeslider=dict(visible=True), type='date'),
    yaxis=dict(title='Price (USD)', fixedrange=False),
    yaxis2=dict(title='RSI', range=[0,100], fixedrange=False),
    dragmode='zoom',
    legend=dict(orientation='h', y=1.03, x=0)
)

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom':True})
