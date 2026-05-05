import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_logic import prepare_data, build_lstm_model, predict_future

# 1. Page Configuration
st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")

# 2. Updated Styling
st.markdown("""
    <style>
    .stApp { background-color: #FAF9F6 !important; }
    section[data-testid="stSidebar"] { background-color: #EFEBE3 !important; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 2px solid #A0522D !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #A0522D !important; }
    p, div, label, .stMetricLabel, .stMetricValue, .stMarkdown { color: #2F4F4F !important; }
    div.stButton > button { 
        background-color: #A0522D !important; 
        color: white !important; 
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - TSLA set as Default
st.sidebar.title("🛠️ Control Panel")
# .strip().upper() ensures no casing or space issues
ticker_input = st.sidebar.text_input("Stock Symbol", "TSLA").strip().upper()
seq_len = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
pred_len = st.sidebar.slider("Prediction Horizon", 1, 14, 7)

# 4. Optimized Data Fetching
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(symbol):
    try:
        # download is faster for single calls than Ticker().history()
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()

        # FIX: Flatten MultiIndex (This is why it says "No Data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Select columns and clean
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# 5. Main UI Logic
st.title("🚀 AI-Powered Stock Forecasting")
st.markdown("### 🧠 Multistep-Multivariate LSTM Neural Network")

# Load Data
with st.spinner(f"Loading {ticker_input} market data..."):
    data = get_data(ticker_input)

if data.empty:
    st.error(f"❌ No data found for '{ticker_input}'. Please check the symbol or internet connection.")
    # Fallback to TSLA button if search fails
    if st.button("Reload Default (TSLA)"):
        st.rerun()
else:
    # 6. Faster Dashboard Rendering
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Market Overview")
        last_price = data['Close'].iloc[-1].item()
        st.metric(f"Latest {ticker_input} Close", f"${last_price:.2f}")
        # Only show last 5 rows to keep session message small and fast
        st.write(data.tail(5))
        
    with col2:
        st.subheader("📈 Historical Trend")
        fig = go.Figure()
        # Only plot last 120 days to speed up Plotly rendering
        plot_data = data.tail(120)
        fig.add_trace(go.Scatter(
            x=plot_data.index, 
            y=plot_data['Close'], 
            line=dict(color='#A0522D'),
            name="Historical"
        ))
        fig.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # 7. Optimized Prediction Execution
    if st.button("🚀 Run AI Prediction", type="primary"):
        if len(data) < seq_len:
            st.error(f"Need {seq_len} days of data, found {len(data)}.")
        else:
            with st.spinner("AI Brain Training (Free Tier Optimized)..."):
                # Use only recent history for training to save speed/memory
                train_data = data.tail(300) 
                X, y, scaler = prepare_data(train_data, seq_len, pred_len)
                model = build_lstm_model(seq_len, pred_len)
                
                # Low epochs for fast execution in Streamlit environment
                model.fit(X, y, epochs=5, batch_size=16, verbose=0)
                
                last_seq = data.values[-seq_len:]
                prediction = predict_future(model, last_seq, scaler)
                
                st.success("✅ Prediction generated!")
                
                # Plot results
                pred_dates = pd.date_range(start=data.index[-1], periods=pred_len+1)[1:]
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=pred_dates, y=prediction, mode='lines+markers', line=dict(color='#6B8E23')))
                fig_pred.update_layout(template="plotly_white", title="AI Forecasted Path")
                st.plotly_chart(fig_pred, use_container_width=True)