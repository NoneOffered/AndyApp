import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from math import log10, sin, pi

st.set_page_config(page_title="BTC Wave Model", layout="wide")

API_KEY  = "ba5525cc9cf29a46525360eb07c0a7cc"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

@st.cache_data(ttl=3600)
def fetch_fred(series_id, start="2010-07-24", end=None):
    params = {
        "series_id": series_id,
        "api_key":    API_KEY,
        "file_type":  "json",
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

# Halving dates & block-times
H0_33, H0_66 = datetime(2010,7,24,17,50,42), datetime(2011,8,7,19,49,38)
H1 = datetime(2012,11,28,16,24,38)
H2 = datetime(2016,7,9,18,46,13)
H3 = datetime(2020,5,11,21,23,43)
H4 = datetime(2028,3,26)
bm0, bm0_33, bm0_66, bm1, bm2, bm3 = 699768, 467896, 591047, 542715, 576868, 593781

def halving_time(t):
    t_ms = t.timestamp() * 1000
    if t < H0_33:
        return    t_ms       / (bm0   * 70000)
    elif t < H0_66:
        return 0.33 + (t_ms - H0_33.timestamp()*1000) / (bm0_33 * 70000)
    elif t < H1:
        return 0.66 + (t_ms - H0_66.timestamp()*1000) / (bm0_66 * 70000)
    elif t < H2:
        return 1.0  + (t_ms -  H1.timestamp()*1000)   / (bm1   * 210000)
    elif t < H3:
        return 2.0  + (t_ms -  H2.timestamp()*1000)   / (bm2   * 210000)
    else:
        return 3.0  + (t_ms -  H3.timestamp()*1000)   / (bm3   * 210000)

a, b = 1.48, 5.44

def btc_trend(h):
    return 10**(a + b * np.log10(h))

def wave_envelope(h, tr):
    wcr, width = 0.25, 0.75
    phase       = 0.75 / h
    osc         = sin(2*pi*h - phase)
    up          = min(width, osc)
    dn          = max(-width, osc)
    decay       = (1 - wcr)**h
    upper  = tr * 10**(decay * (up + width))
    lower  = tr * 10**(decay * (dn - width))
    middle = tr * 10**(decay * osc)
    return upper, lower, middle

# Fetch real data
today_str   = pd.Timestamp.today().strftime("%Y-%m-%d")
btc_real    = fetch_fred("CBBTCUSD", "2010-07-24", today_str).resample("D").ffill()
m2_real     = fetch_fred("M2SL",     "2010-07-24", today_str).resample("D").ffill() / 1e3

# Full index & shift
full_idx    = pd.date_range("2010-07-24", H4, freq="D")
m2_ext      = m2_real.reindex(full_idx).ffill()
m2_shifted  = m2_ext.shift(77)

# Compute model
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

# Halving/ midpoints
halvings  = [H1, H2, H3, H4]
midpoints = [halvings[i] + (halvings[i+1]-halvings[i])/2 for i in range(3)]

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=full_idx, y=uppers, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
fig.add_trace(go.Scatter(x=full_idx, y=lowers, fill='tonexty', fillcolor='rgba(0,0,200,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Wave Envelope'))
fig.add_trace(go.Scatter(x=full_idx, y=trends,  mode='lines', line=dict(color='red', dash='dash'), name='Trend'))
fig.add_trace(go.Scatter(x=full_idx, y=middles, mode='lines', line=dict(color='blue'), name='Wave Middle'))
fig.add_trace(go.Scatter(x=btc_real.index, y=btc_real.values, mode='lines', line=dict(color='black'), name='BTC Price'))
#fig.add_trace(go.Scatter(x=full_idx, y=m2_shifted, mode='lines', line=dict(color='orange', dash='dot'), name='US M2 shifted 77d', yaxis='y2'))
for d in halvings:  fig.add_vline(x=d, line=dict(color='gray', dash='dot'))
for d in midpoints: fig.add_vline(x=d, line=dict(color='gray', dash='dash'))
fig.add_vline(x='2025-09-31', line=dict(color='black'))
fig.add_hline(y=138000, line=dict(color='black'))

y0, y1 = log10(10000), log10(200000)
fig.update_layout(
    title="BTC Price & Wave Model Extrapolated to Mar 26 2028",
    xaxis=dict(title='Date', range=['2022-01-01','2026-12-31'], rangeslider=dict(visible=True), type='date'),
    yaxis=dict(title='Price (USD, log)', type='log', range=[y0,y1]),
    #yaxis2=dict(title='US M2 (USD bln)', overlaying='y', side='right'),
    height=700, hovermode='x unified', dragmode='zoom'
)

st.plotly_chart(fig, use_container_width=True)
