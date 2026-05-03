import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="QuantLab | Financial Engineering",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════
# GLOBAL CSS — GLASSMORPHISM + NEON THEME
# ══════════════════════════════════════════════════
st.markdown("""
<style>  
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');  
  
:root{  
  --bg:#050914;  
  --bg2:#080d1e;  
  --blue:#00d4ff;  
  --green:#00ff88;  
  --gold:#ffd700;  
  --red:#ff4d6d;  
  --text:#c8d6f0;  
  --muted:#4a5568;  
  --glass:rgba(0,212,255,0.04);  
  --border:rgba(0,212,255,0.12);  
}  
  
html,body,[class*="css"],.stApp{  
  background-color:var(--bg)!important;  
  color:var(--text)!important;  
  font-family:'Inter',sans-serif;  
}  
  
.stApp{  
  background:  
    radial-gradient(ellipse 80% 60% at 0% 0%,rgba(0,212,255,0.07) 0%,transparent 55%),  
    radial-gradient(ellipse 70% 50% at 100% 90%,rgba(0,255,136,0.05) 0%,transparent 55%),  
    var(--bg)!important;  
}  

/* ── METRICS ── */  
[data-testid="metric-container"]{  
  background:var(--glass)!important;  
  border:1px solid var(--border)!important;  
  border-radius:12px!important;  
  padding:16px!important;  
  backdrop-filter:blur(20px)!important;  
}  

/* ── BUTTONS ── */  
.stButton>button{  
  background:transparent!important;  
  border:1px solid var(--blue)!important;color:var(--blue)!important;  
  font-family:'JetBrains Mono',monospace!important;  
  letter-spacing:2px!important;text-transform:uppercase!important;  
  width: 100%;
}
</style>  """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# PLOTLY HELPER & FUNCTIONS
# ══════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#8892b0", size=11),
    margin=dict(l=0, r=0, t=40, b=0),
)

def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 1e-6: return max(S - K, 0) if opt == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def bs_greeks(S, K, T, r, sigma, opt="call"):
    if T <= 1e-6: return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if opt == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {"delta": delta, "gamma": gamma, "theta": 0, "vega": vega, "rho": 0}

def build_bs_surface(K, T_max, r, sigma, opt):
    S_vals = np.linspace(max(K * 0.4, 10), K * 1.8, 40)
    T_vals = np.linspace(0.01, T_max, 40)
    S_grid, T_grid = np.meshgrid(S_vals, T_vals)
    vec = np.vectorize(lambda s, t: bs_price(s, K, t, r, sigma, opt))
    Z = vec(S_grid, T_grid)
    return S_vals, T_vals, Z

# ══════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════
def render_ticker():
    st.markdown('<div style="color:#00ff88; font-family:monospace; font-size:12px; text-align:center;">'
                'BTC +2.34% | S&P500 +0.84% | NVDA -0.47% | AAPL +1.23%</div>', unsafe_allow_html=True)

def section_header(title, subtitle, badge=None):
    st.markdown(f"## {title} <small style='color:#4a5568;'>{badge if badge else ''}</small>", unsafe_allow_html=True)
    st.caption(subtitle)

# ══════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════
def render_bs_tab():
    with st.sidebar:
        st.header("Parameters")
        S0 = st.slider("Stock Price (S)", 10.0, 200.0, 100.0)
        K = st.slider("Strike Price (K)", 10.0, 200.0, 100.0)
        T_max = st.slider("Time to Expiry (Y)", 0.1, 2.0, 1.0)
        sigma = st.slider("Volatility (σ)", 0.05, 1.0, 0.2)
        r = st.slider("Risk-free Rate (r)", 0.01, 0.15, 0.05)
        opt_type = st.selectbox("Option Type", ["Call", "Put"])

    section_header("Black-Scholes Engine", "Real-time derivatives pricing & 3D Visualization", "Live")
    
    price = bs_price(S0, K, T_max, r, sigma, opt_type.lower())
    greeks = bs_greeks(S0, K, T_max, r, sigma, opt_type.lower())
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${price:.2f}")
    c2.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
    c3.metric("Vega (ν)", f"{greeks['vega']:.4f}")

    S_vals, T_vals, Z = build_bs_surface(K, T_max, r, sigma, opt_type.lower())
    fig = go.Figure(data=[go.Surface(z=Z, x=S_vals, y=T_vals, colorscale="Plasma")])
    fig.update_layout(**PLOT_LAYOUT, height=600, scene=dict(
        xaxis_title="Stock Price", yaxis_title="Time", zaxis_title="Option Price"
    ))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════
render_ticker()
tab1, tab2, tab3 = st.tabs(["Black-Scholes", "Portfolio", "Monte Carlo"])

with tab1:
    render_bs_tab()

with tab2:
    st.info("Portfolio Optimization Module: Select tickers to begin analysis.")

with tab3:
    st.info("Monte Carlo Simulation: Predicting price paths using Geometric Brownian Motion.")
