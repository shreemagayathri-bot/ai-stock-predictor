import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_logic import prepare_data, build_lstm_model, predict_future

st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")

# 2. Updated Styling: Dark Text / Original Earthy Background
st.markdown("""
    <style>
    /* Main Background: Tan/Beige */
    .stApp { background-color: #FAF9F6 !important; }
    
    /* Sidebar Background: Light Earthy Grey */
    section[data-testid="stSidebar"] { background-color: #EFEBE3 !important; }
    
    /* Metric Card: White background with Brown border */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 2px solid #A0522D !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    /* Headers: Brown */
    h1, h2, h3, h4, h5, h6 { color: #A0522D !important; }

    /* ALL OTHER TEXT: Dark Charcoal for legibility */
    p, div, label, .stMetricLabel, .stMetricValue, .stMarkdown, section[data-testid="stSidebar"] .stSlider label {
        color: #2F4F4F !important;
    }
    
    /* Button: Brown background with White text */
    div.stButton > button { 
        background-color: #A0522D !important; 
        color: white !important; 
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🛠️ Control Panel")
ticker_input = st.sidebar.text_input("Stock Symbol", "AAPL").strip().upper()
seq_len = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
pred_len = st.sidebar.slider("Prediction Horizon", 1, 14, 7)

@st.cache_data(ttl=3600)
def get_data(symbol):
    try:
        # 1. Force uppercase and strip whitespace
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            return pd.DataFrame()

        # 2. Fetch using Ticker object
        ticker_obj = yf.Ticker(clean_symbol)
        
        # 3. Use period="2y" to ensure we have enough data for the lookback
        df = ticker_obj.history(period="2y", interval="1d", auto_adjust=True)
        
        # 4. Fallback: If history() fails, try download()
        if df is None or df.empty:
            df = yf.download(clean_symbol, period="2y", progress=False, auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()

        # 5. Handle MultiIndex columns (sometimes happens with yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 6. Select core columns and remove timezone info for Plotly/Model compatibility
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df.index = df.index.tz_localize(None)
        
        return df
    except Exception as e:
        # Printing to console helps you debug during development
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()
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
            # Use light template for light background
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Run AI Prediction", type="primary"):
            if len(data) < seq_len:
                st.error(f"Need {seq_len} days of data, found {len(data)}.")
            else:
                with st.spinner("Training (Free Tier Optimized)..."):
                    X, y, scaler = prepare_data(data, seq_len, pred_len)
                    model = build_lstm_model(seq_len, pred_len)
                    model.fit(X[-50:], y[-50:], epochs=2, batch_size=16, verbose=0)
                    
                    last_seq = data.values[-seq_len:]
                    prediction = predict_future(model, last_seq, scaler)
                    
                    st.success("✅ Prediction generated!")
                    pred_dates = pd.date_range(start=data.index[-1], periods=pred_len+1)[1:]
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=pred_dates, y=prediction, mode='lines+markers', line=dict(color='#6B8E23')))
                    fig_pred.update_layout(template="plotly_white")
                    st.plotly_chart(fig_pred, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {e}")