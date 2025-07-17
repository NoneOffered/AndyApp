# Streamlit App: BTC Wave Model + RSI Subplot with Shared Range Slider & Custom Y Zoom

# Save as app.py and run: streamlit run app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from math import log10, sin, pi

st.set_page_config(page_title="BTC & RSI Overview", layout="wide")

# 1) FRED helper
API_KEY  = "ba5525cc9cf29a46525360eb07c0a7cc"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
@st.cache_data(ttl=3600)
def fetch_fred(series_id, start="2010-07-24", end=None):
    params = {"series_id": series_id,"api_key":API_KEY,"file_type":"json","observation_start":start}
    if end: params["observation_end"]=end
    r=requests.get(BASE_URL,params=params,timeout=10); r.raise_for_status()
    df=pd.DataFrame(r.json()["observations"])
    df["date"]=pd.to_datetime(df["date"]); df["value"]=pd.to_numeric(df["value"],errors="coerce")
    return df.set_index("date")["value"]

# 2) Halving & wave definitions
H0_33, H0_66 = datetime(2010,7,24,17,50,42), datetime(2011,8,7,19,49,38)
H1, H2 = datetime(2012,11,28,16,24,38), datetime(2016,7,9,18,46,13)
H3, H4 = datetime(2020,5,11,21,23,43), datetime(2028,3,26)
bm_vals = [699768,467896,591047,542715,576868,593781]
a,b=1.48,5.44
def halving_time(t):
    ms=t.timestamp()*1000; bounds=[H0_33,H0_66,H1,H2,H3,H4]
    for i in range(len(bounds)-1):
        if t< bounds[i+1]:
            start_ms=bounds[i].timestamp()*1000
            blocks=210000 if i>=1 else 70000
            return i+(ms-start_ms)/(bm_vals[i]*blocks)
    start_ms=bounds[-2].timestamp()*1000
    return len(bounds)-2+(ms-start_ms)/(bm_vals[-1]*210000)
def btc_trend(h): return 10**(a+b*np.log10(h))
def wave_envelope(h,tr):
    if h<=0: h=1e-6
    wcr,wd=0.25,0.75; phase=0.75/h; osc=sin(2*pi*h-phase)
    up,dn=min(wd,osc),max(-wd,osc); decay=(1-wcr)**h
    return (tr*10**(decay*(up+wd)),tr*10**(decay*(dn-wd)),tr*10**(decay*osc))

# 3) Fetch data
today_str=pd.Timestamp.today().strftime("%Y-%m-%d")
btc = fetch_fred("CBBTCUSD","2010-07-24",today_str).resample("D").ffill()
idx = pd.date_range("2010-07-24",H4,freq="D")
price = btc.reindex(idx).ffill()

# 4) RSI
delta=price.diff(); gain=delta.where(delta>0,0.0); loss=-delta.where(delta<0,0.0)
avg_g=gain.rolling(14).mean(); avg_l=loss.rolling(14).mean()
rs=avg_g/avg_l; rsi=100-(100/(1+rs))

# 5) Model arrays
tr, up_arr, dn_arr, mid_arr = [], [], [], []
for t in idx:
    h=halving_time(t); tr_val=btc_trend(h)
    up,dn,mid=wave_envelope(h,tr_val)
    tr.append(tr_val); up_arr.append(up); dn_arr.append(dn); mid_arr.append(mid)
tr, up_arr, dn_arr, mid_arr = map(np.array,(tr,up_arr,dn_arr,mid_arr))

# 6) Trailing stop
peak=price.cummax(); stop=peak*0.90

# 7) Subplots: price+model & RSI
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.8,0.2], vertical_spacing=0.02,
    subplot_titles=("BTC Price & Wave Model","RSI (14)"))

# Price & model (row1)
fig.add_trace(go.Scatter(x=idx,y=up_arr, line=dict(color='rgba(0,0,0,0)'), showlegend=False),row=1,col=1)
fig.add_trace(go.Scatter(x=idx,y=dn_arr, fill='tonexty', fillcolor='rgba(0,0,200,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Wave Env'),row=1,col=1)
fig.add_trace(go.Scatter(x=idx,y=tr,mode='lines',line=dict(color='red',dash='dash'),name='Trend'),row=1,col=1)
fig.add_trace(go.Scatter(x=idx,y=mid_arr,mode='lines',line=dict(color='blue'),name='Wave Mid'),row=1,col=1)
fig.add_trace(go.Scatter(x=idx,y=price,mode='lines',line=dict(color='black'),name='BTC Price'),row=1,col=1)
fig.add_trace(go.Scatter(x=idx,y=peak,mode='lines',line=dict(color='rgba(0,0,0,0)'),showlegend=False),row=1,col=1)
fig.add_trace(go.Scatter(x=idx,y=stop,mode='lines',line=dict(color='teal'),fill='tonexty',fillcolor='rgba(0,128,128,0.2)',name='10% Stop'),row=1,col=1)
for d in [H1,H2,H3,H4]: fig.add_vline(x=d,line=dict(color='gray',dash='dot'))
for i in range(3):
    m=( [H1,H2,H3,H4][i] + [H1,H2,H3,H4][i+1] )/2
    fig.add_vline(x=m,line=dict(color='gray',dash='dash'))

# RSI (row2)
fig.add_trace(go.Scatter(x=rsi.index,y=rsi,mode='lines',line=dict(color='black'),name='RSI'),row=2,col=1)
# Overbought rectangles on RSI subplot
for low,color,op in [(80,"yellow",0.3),(85,"orange",0.25),(90,"darkorange",0.2),(95,"red",0.15)]:
    fig.add_hrect(y0=low, y1=100, line_width=0, fillcolor=color, opacity=op, row=2, col=1)

# Layout with customizable Y zoom
y0,y1 = log10(1000), log10(200000)
fig.update_layout(
    height=700, hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=True),type='date',range=['2022-01-01','2026-12-31']),
    yaxis=dict(type='log',title='Price (USD)',range=[y0,y1],fixedrange=False),
    yaxis2=dict(title='RSI',range=[0,100],fixedrange=False),
    dragmode='zoom', legend=dict(orientation='h',y=1.05,x=0)
)

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
