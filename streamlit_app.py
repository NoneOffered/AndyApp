# Complete Streamlit App: BTC Wave Model + M2 Shift + RSI Bands + 10% Trailing Stop
# should be fine
# Save this as `app.py` and run with:
#   streamlit run app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from math import log10, sin, pi

st.set_page_config(page_title="BTC Wave Model & RSI", layout="wide")

# --- 1) FRED fetch helper ---
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

# --- 2) Halving params & wave model funcs ---
H0_33, H0_66 = datetime(2010,7,24,17,50,42), datetime(2011,8,7,19,49,38)
H1 = datetime(2012,11,28,16,24,38)
H2 = datetime(2016,7,9,18,46,13)
H3 = datetime(2020,5,11,21,23,43)
H4 = datetime(2028,3,26)
bm0, bm0_33, bm0_66, bm1, bm2, bm3 = 699768, 467896, 591047, 542715, 576868, 593781
a, b = 1.48, 5.44

def halving_time(t):
    t_ms = t.timestamp()*1000
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

# --- 3) Fetch BTC & M2 data ---
today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
btc = fetch_fred("CBBTCUSD", "2010-07-24", today_str).resample("D").ffill()
m2  = fetch_fred("M2SL",     "2010-07-24", today_str).resample("D").ffill() / 1e3

# --- 4) Compute RSI ---
window = 14
delta = btc.diff()
gain  = delta.where(delta > 0, 0.0)
loss  = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(window).mean()
avg_loss = loss.rolling(window).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# --- 5) Build full calendar index & M2 shift slider ---
full_idx = pd.date_range("2010-07-24", H4, freq="D")
m2_ext   = m2.reindex(full_idx).ffill()
shift    = st.sidebar.slider("Shift US M2 (days)", 0, 90, 77)

# --- 6) Compute wave model arrays ---
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

# --- 7) Compute trailing stop (10% under rolling peak) ---
price        = btc.reindex(full_idx).ffill()
rolling_peak = price.cummax()
trail_stop   = rolling_peak * 0.90

# --- 8) Halving & midpoints ---
halvings  = [H1, H2, H3, H4]
midpts    = [halvings[i] + (halvings[i+1]-halvings[i])/2 for i in range(3)]

# --- 9) Plot main chart ---
fig = go.Figure()
# Wave envelope
fig.add_trace(go.Scatter(x=full_idx, y=uppers, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
fig.add_trace(go.Scatter(x=full_idx, y=lowers, fill='tonexty',
                         fillcolor='rgba(0,0,200,0.2)', line=dict(color='rgba(0,0,0,0)'), 
                         name='Wave Envelope'))
# Trend & middle
fig.add_trace(go.Scatter(x=full_idx, y=trends,  mode='lines',
                         line=dict(color='red',  dash='dash'), name='Trend'))
fig.add_trace(go.Scatter(x=full_idx, y=middles, mode='lines',
                         line=dict(color='blue'), name='Wave Middle'))
# BTC price
fig.add_trace(go.Scatter(x=btc.index, y=btc.values, mode='lines',
                         line=dict(color='black'), name='BTC Price'))
# Trailing stop shading
fig.add_trace(go.Scatter(x=full_idx, y=rolling_peak,
                         mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
fig.add_trace(go.Scatter(x=full_idx, y=trail_stop,
                         mode='lines', line=dict(color='blue'),
                         fill='tonexty', fillcolor='rgba(0,0,255,0.2)',
                         name='10% Trailing Stop'))
# US M2 shifted
fig.add_trace(go.Scatter(x=full_idx, y=m2_ext.shift(shift), mode='lines',
                         line=dict(color='orange', dash='dot'),
                         name=f'US M2 shifted {shift}d', yaxis='y2'))
# Halving & midpoints lines
for d in halvings:  fig.add_vline(x=d, line=dict(color='gray', dash='dot'))
for d in midpts:    fig.add_vline(x=d, line=dict(color='gray', dash='dash'))

# Layout main
y0, y1 = log10(100), log10(200000)
fig.update_layout(
    title="BTC Price & Wave Model Extrapolated to Mar 26 2028",
    xaxis=dict(title='Date', range=['2022-01-01','2026-12-31'], rangeslider=dict(visible=True), type='date'),
    yaxis=dict(title='Price (USD, log)', type='log', range=[y0,y1]),
    yaxis2=dict(title='US M2 (USD bln)', overlaying='y', side='right'),
    height=600, hovermode='x unified', dragmode='zoom'
)

# --- 10) Plot RSI with shaded overbought zones ---
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines', name='RSI (14)', line=dict(color='black')))
fig_rsi.add_hrect(y0=80, y1=85, fillcolor="yellow", opacity=0.3, line_width=0)
fig_rsi.add_hrect(y0=85, y1=90, fillcolor="orange", opacity=0.25, line_width=0)
fig_rsi.add_hrect(y0=90, y1=95, fillcolor="darkorange", opacity=0.2, line_width=0)
fig_rsi.add_hrect(y0=95, y1=100, fillcolor="red", opacity=0.15, line_width=0)
fig_rsi.update_layout(title="Bitcoin 14-day RSI with Overbought Zones",
                      yaxis=dict(range=[0,100]), xaxis=dict(rangeslider=dict(visible=True)),
                      height=300)

# --- 11) Display in Streamlit ---
st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig_rsi, use_container_width=True)