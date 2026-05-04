import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_logic import prepare_data, build_lstm_model, predict_future

st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")

# Earthy Styling
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
    div.stButton > button { background-color: #A0522D !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🛠️ Control Panel")
ticker_input = st.sidebar.text_input("Stock Symbol", "AAPL")
seq_len = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
pred_len = st.sidebar.slider("Prediction Horizon", 1, 14, 7)

@st.cache_data(ttl=3600)
def get_data(symbol):
    try:
        clean_symbol = symbol.strip().upper()
        ticker_obj = yf.Ticker(clean_symbol)
        df = ticker_obj.history(period="2y", interval="1d", auto_adjust=True)
        
        if df.empty:
            df = yf.download(clean_symbol, period="2y", progress=False, auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()

        # FIX: Flatten MultiIndex columns (common for AAPL/TSLA)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'High', 'Low', 'Close']]
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

st.title("🚀 AI-Powered Stock Forecasting")
st.markdown("### 🧠 Multistep-Multivariate LSTM Neural Network")

try:
    with st.spinner(f"Fetching data for {ticker_input}..."):
        data = get_data(ticker_input)
    
    if data.empty:
        st.error(f"❌ No data found for '{ticker_input}'.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📊 Market Overview")
            last_price = data['Close'].iloc[-1].item()
            st.metric(f"Latest {ticker_input} Close", f"${last_price:.2f}")
            st.write(data.tail(5))
            
        with col2:
            st.subheader("📈 Historical Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[-100:], y=data['Close'][-100:], line=dict(color='#A0522D')))
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Run AI Prediction", type="primary"):
            if len(data) < seq_len:
                st.error(f"Need {seq_len} days of data, found {len(data)}.")
            else:
                with st.spinner("Training (Free Tier Optimized)..."):
                    X, y, scaler = prepare_data(data, seq_len, pred_len)
                    model = build_lstm_model(seq_len, pred_len)
                    # Small subset (50 samples) and 2 epochs to save Render RAM
                    model.fit(X[-50:], y[-50:], epochs=2, batch_size=16, verbose=0)
                    
                    last_seq = data.values[-seq_len:]
                    prediction = predict_future(model, last_seq, scaler)
                    
                    st.success("✅ Prediction generated!")
                    pred_dates = pd.date_range(start=data.index[-1], periods=pred_len+1)[1:]
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=pred_dates, y=prediction, mode='lines+markers', line=dict(color='#6B8E23')))
                    st.plotly_chart(fig_pred, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {e}")