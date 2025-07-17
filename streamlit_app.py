# Streamlit App: BTC Wave Model + RSI‐Proportional Vertical Shading (Merged Spans)

# Save as app.py and run: streamlit run app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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

# 2) Halving & wave parameters
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
    return 10**(a + b * np.log10(max(h,1e-6)))

def wave_envelope(h,tr):
    h = max(h,1e-6)
    wcr, wd = 0.25, 0.75
    phase = 0.75/h
    osc   = sin(2*pi*h - phase)
    up, dn = min(wd,osc), max(-wd,osc)
    decay  = (1-wcr)**h
    return (
        tr * 10**(decay*(up+wd)),
        tr * 10**(decay*(dn-wd)),
        tr * 10**(decay*osc)
    )

# 3) Fetch BTC price & build index
today = pd.Timestamp.today().strftime("%Y-%m-%d")
btc   = fetch_fred("CBBTCUSD","2010-07-24",today).resample("D").ffill()
idx   = pd.date_range("2010-07-24", H4, freq="D")
price = btc.reindex(idx).ffill()

# 4) Compute RSI (14-day)
delta    = price.diff()
gain     = delta.where(delta>0,0.0)
loss     = -delta.where(delta<0,0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs       = avg_gain / avg_loss
rsi      = 100 - (100/(1+rs))

# 5) Compute wave model arrays
trs, ups, dns, mids = [], [], [], []
for t in idx:
    h  = halving_time(t)
    tr = btc_trend(h)
    up, dn, mid = wave_envelope(h, tr)
    trs.append(tr); ups.append(up)
    dns.append(dn); mids.append(mid)
trs, ups, dns, mids = map(np.array,(trs,ups,dns,mids))

# 6) Trailing stop (10% below peak)
peak = price.cummax()
stop = peak * 0.90

# 7) Merge RSI>50 into spans for vertical shading
max_opacity = 0.3
mask = rsi > 70
spans = []
in_span = False
for dt, hot in zip(rsi.index, mask):
    if hot and not in_span:
        start = dt
        in_span = True
    elif not hot and in_span:
        end = prev_dt + timedelta(days=1)
        mean_rsi = rsi[start:prev_dt].mean()
        spans.append((start, end, mean_rsi))
        in_span = False
    prev_dt = dt
if in_span:
    end = prev_dt + timedelta(days=1)
    mean_rsi = rsi[start:prev_dt].mean()
    spans.append((start, end, mean_rsi))

# 8) Build subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75,0.25], vertical_spacing=0.03)

# 8a) Main chart shading + wave + price + stop
for start, end, mean_rsi in spans:
    op = ((mean_rsi - 50)/50) * max_opacity
    fig.add_vrect(x0=start, x1=end, y0=0, y1=1,
                  fillcolor="red", opacity=op,
                  layer="below", row=1, col=1, yref="paper")

fig.add_trace(go.Scatter(x=idx, y=ups,
                         line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=dns, fill='tonexty',
                         fillcolor='rgba(0,0,200,0.2)', line=dict(color='rgba(0,0,0,0)'),
                         name='Wave Env'), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=trs, mode='lines',
                         line=dict(color='red', dash='dash'), name='Trend'), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=mids, mode='lines',
                         line=dict(color='blue'), name='Wave Mid'), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=price, mode='lines',
                         line=dict(color='black'), name='BTC Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=stop, mode='lines',
                         line=dict(color='teal'),
                         fill='tonexty', fillcolor='rgba(0,128,128,0.2)',
                         name='10% Stop'), row=1, col=1)

fig.add_vline(x="2025-09-30", line=dict(color='green'), name='2025-09-30')
fig.add_hline(y="138000", line=dict(color='green'), name='$138,000')

# 8b) Halving & midpoints
halvings  = [H1, H2, H3, H4]
midpoints = [halvings[i] + (halvings[i+1] - halvings[i]) / 2 for i in range(3)]
for d in halvings:
    fig.add_vline(x=d, line=dict(color='gray', dash='dot'))
for m in midpoints:
    fig.add_vline(x=m, line=dict(color='gray', dash='dash'))

# 8c) RSI subplot
fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines',
                         line=dict(color='black'), name='RSI'), row=2, col=1)



# 9) Layout: no sliders, log y-axis
fig.update_layout(
    height=700,
    hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=False), type='date'),
    xaxis2=dict(rangeslider=dict(visible=False), type='date', range=['2018-01-01','2026-12-31']),
    yaxis=dict(title='Price (USD)', type='log', fixedrange=False, range=[1000,200000]),
    yaxis2=dict(title='RSI', range=[50,100], fixedrange=False),
    dragmode='zoom',
    legend=dict(orientation='h', y=1.03, x=0)
)

st.plotly_chart(fig, use_container_width=True)



