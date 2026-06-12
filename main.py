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
# MAIN RENDER FUNCTIONS FOR TABS
# ══════════════════════════════════════════════════
def render_bs_tab():
    with st.sidebar:
        st.header("Black-Scholes Params")
        S0 = st.slider("Stock Price (S)", 10.0, 200.0, 100.0, key="bs_s")
        K = st.slider("Strike Price (K)", 10.0, 200.0, 100.0, key="bs_k")
        T_max = st.slider("Time to Expiry (Y)", 0.1, 2.0, 1.0, key="bs_t")
        sigma = st.slider("Volatility (σ)", 0.05, 1.0, 0.2, key="bs_sig")
        r = st.slider("Risk-free Rate (r)", 0.01, 0.15, 0.05, key="bs_r")
        opt_type = st.selectbox("Option Type", ["Call", "Put"], key="bs_opt")

    section_header("Black-Scholes Engine", "Real-time derivatives pricing & 3D Visualization", "Live")
    
    price = bs_price(S0, K, T_max, r, sigma, opt_type.lower())
    greeks = bs_greeks(S0, K, T_max, r, sigma, opt_type.lower())
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${price:.2f}")
    c2.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
    c3.metric("Vega (ν)", f"{greeks['vega']:.4f}")

    S_vals, T_vals, Z = build_bs_surface(K, T_max, r, sigma, opt_type.lower())
    fig = go.Figure(data=[go.Surface(z=Z, x=S_vals, y=T_vals, colorscale="Plasma")])
    fig.update_layout(**PLOT_LAYOUT, height=500, scene=dict(
        xaxis_title="Stock Price", yaxis_title="Time", zaxis_title="Option Price"
    ))
    st.plotly_chart(fig, use_container_width=True)

def render_portfolio_tab():
    with st.sidebar:
        st.header("Portfolio Params")
        tickers_input = st.text_input("Stocks Tickers (Comma separated)", "AAPL,MSFT,GOOGL,AMZN")
        rf_rate = st.slider("Risk-free Rate (Sharpe)", 0.0, 0.1, 0.02, step=0.01)
    
    section_header("Markowitz Portfolio Optimization", "Modern Portfolio Theory (MPT) & Efficient Frontier", "Optimizer")
    
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        
        closing_prices = pd.DataFrame()
        
        for t in tickers:
            df = yf.download(t, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # لتفادي الـ MultiIndex الناجم عن تحديث yfinance، نقوم بتسوية الأعمدة وتبسيط الأسماء
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # استخراج أسعار الإغلاق بأمان وإضافتها لـ DataFrame الرئيسي
                if 'Adj Close' in df.columns:
                    closing_prices[t] = df['Adj Close']
                elif 'Close' in df.columns:
                    closing_prices[t] = df['Close']
        
        if closing_prices.empty:
            st.error("No data found for the provided tickers.")
            return
            
        closing_prices = closing_prices.ffill().bfill()
                
        returns = closing_prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        num_portfolios = 1000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            p_return = np.sum(weights * mean_returns)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            results[0,i] = p_std
            results[1,i] = p_return
            results[2,i] = (p_return - rf_rate) / p_std
            
        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
        best_weights = weights_record[max_sharpe_idx]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results[0], y=results[1], mode='markers',
                marker=dict(color=results[2], colorscale='Viridis', showscale=True, size=5),
                name='Portfolios'
            ))
            fig.add_trace(go.Scatter(
                x=[sdp], y=[rp], mode='markers',
                marker=dict(color='red', size=12, symbol='star'),
                name='Max Sharpe Ratio'
            ))
            fig.update_layout(**PLOT_LAYOUT, title="Efficient Frontier Surface", height=450, xaxis_title="Volatility", yaxis_title="Expected Return")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### Optimal Allocation")
            df_weights = pd.DataFrame({'Asset': tickers, 'Weight': best_weights})
            df_weights['Weight'] = df_weights['Weight'].apply(lambda x: f"{x*100:.2f}%")
            st.table(df_weights)
            st.metric("Expected Return", f"{rp*100:.2f}%")
            st.metric("Portfolio Volatility", f"{sdp*100:.2f}%")
            
    except Exception as e:
        st.error(f"Please check tickers or connectivity: {e}")

def render_monte_carlo_tab():
    with st.sidebar:
        st.header("Monte Carlo Params")
        S_mc = st.number_input("Initial Price", value=100.0)
        mu_mc = st.slider("Expected Return (μ)", -0.2, 0.5, 0.1)
        sigma_mc = st.slider("Volatility (σ)", 0.05, 1.0, 0.2, key="mc_sig")
        days_mc = st.slider("Simulation Horizon (Days)", 30, 365, 252)
        sim_count = st.slider("Simulations Count", 10, 200, 50)
        
    section_header("Geometric Brownian Motion", "Stochastic Simulation for asset price paths", "Simulation")
    
    dt = 1 / 252
    time_series = np.arange(days_mc)
    fig = go.Figure()
    
    final_prices = []
    for i in range(sim_count):
        price_path = [S_mc]
        for t in range(1, days_mc):
            drift = (mu_mc - 0.5 * sigma_mc**2) * dt
            shock = sigma_mc * np.random.normal() * np.sqrt(dt)
            price_path.append(price_path[-1] * np.exp(drift + shock))
        
        final_prices.append(price_path[-1])
        fig.add_trace(go.Scatter(y=price_path, mode='lines', opacity=0.4, line=dict(width=1.5), showlegend=False))
        
    fig.update_layout(**PLOT_LAYOUT, title="Simulated Geometric Brownian Motion Paths", height=450, xaxis_title="Timeline (Days)", yaxis_title="Asset Value ($)")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Path Analytics")
        st.metric("Expected Ending Price", f"${np.mean(final_prices):.2f}")
        st.metric("95% Value at Risk (VaR)", f"${(S_mc - np.percentile(final_prices, 5)):.2f}")

# ══════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════
render_ticker()
tab1, tab2, tab3 = st.tabs(["Black-Scholes", "Portfolio Optimization", "Monte Carlo Simulation"])

with tab1:
    render_bs_tab()

with tab2:
    render_portfolio_tab()

with tab3:
    render_monte_carlo_tab()
    
# ══════════════════════════════════════════════════
# MICROSOFT FOUNDRY IQ — AI FINANCIAL AGENT
# ══════════════════════════════════════════════════
from groq import Groq

st.markdown("---")
st.subheader("💼 QuantLab Smart Financial Agent (Foundry IQ)")
st.write("Ask the AI agent about financial engineering, analysis, or market calculations. / اسأل الوكيل الذكي عن التحليلات والعمليات المالية.")

# سحب الـ API Key من الـ Secrets الخاصة بـ Streamlit بأمان لتفادي الأخطاء الكودية المباشرة
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

try:
    if not GROQ_API_KEY:
        st.info("Please add GROQ_API_KEY in Streamlit Secrets to enable the AI Agent. / يرجى إضافة مفتاح GROQ_API_KEY في إعدادات المنصة لتفعيل الوكيل المالي.")
    else:
        client = Groq(api_key=GROQ_API_KEY)
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Welcome! I am your smart financial agent powered by Microsoft Foundry IQ. You can now also upload context files or sheets directly. \n\nمرحباً بك! أنا وكيلك المالي الذكي المدعوم بـ Microsoft Foundry IQ. يمكنك الآن أيضاً تحميل ملفات البيانات أو الجداول مباشرة لتحليلها."}
            ]

        uploaded_file = st.file_uploader("Upload financial data sheet or image context (CSV, XLSX, PDF, PNG, JPG)", type=["csv", "xlsx", "pdf", "png", "jpg"])
        
        file_context = ""
        if uploaded_file is not None:
            st.success(f"📎 Attached file: {uploaded_file.name}")
            if uploaded_file.name.endswith('.csv'):
                df_preview = pd.read_csv(uploaded_file).head(5)
                file_context = f"\n[User uploaded a data sheet preview:\n{df_preview.to_string()}]"
            elif uploaded_file.name.endswith('.xlsx'):
                df_preview = pd.read_excel(uploaded_file).head(5)
                file_context = f"\n[User uploaded a spreadsheet preview:\n{df_preview.to_string()}]"
            else:
                file_context = f"\n[User uploaded an image or doc asset: {uploaded_file.name}]"

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_input := st.chat_input("Ask a financial question... / اكتب سؤالك المالي هنا..."):
            full_prompt = user_input + file_context
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Analyzing via Foundry IQ layer... / جاري التفكير والتحليل..."):
                    completion = client.chat.completions.create(
                        model="llama-3.1-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are an expert AI Financial Agent specialized in Financial Engineering and Quantitative Analysis for the Microsoft SkillsBuild Agents League Hackathon. Respond fluently in the language used by the user. If the system includes structured data text from a user's uploaded document, parse and analyze it accurately."}
                        ] + [{"role": "user" if m["role"]=="user" else "assistant", "content": m["content"]} for m in st.session_state.messages[-4:]] + [{"role": "user", "content": full_prompt}]
                    )
                    ai_response = completion.choices[0].message.content
                    st.write(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.rerun()
                    
except Exception as e:
    st.error(f"Connection Error: {str(e)}")        
