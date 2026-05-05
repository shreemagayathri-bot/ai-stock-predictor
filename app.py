import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_logic import prepare_data, build_lstm_model, predict_future

# 1. Page Configuration
st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")

# 2. THEME RESTORED: Original Earthy / Light Background
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

# 3. Sidebar - TSLA as Default
st.sidebar.title("🛠️ Control Panel")
ticker_input = st.sidebar.text_input("Stock Symbol", "TSLA").strip().upper()
seq_len = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
pred_len = st.sidebar.slider("Prediction Horizon", 1, 14, 7)

# 4. Optimized Data Fetching (Fixed NoneType Error)
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(symbol):
    try:
        # threads=False prevents the 'NoneType object is not subscriptable' error
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True, threads=False)
        
        if df is None or df.empty:
            return pd.DataFrame()

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# 5. Main UI Logic
st.title("🚀 AI-Powered Stock Forecasting")
st.markdown("### 🧠 Multistep-Multivariate LSTM Neural Network")

with st.spinner(f"Fetching data for {ticker_input}..."):
    data = get_data(ticker_input)

if data.empty:
    st.error(f"❌ No data found for '{ticker_input}'. Please check the symbol.")
else:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📊 Market Overview")
        last_price = data['Close'].iloc[-1].item()
        st.metric(f"Latest {ticker_input} Close", f"${last_price:.2f}")
        st.write(data.tail(5)) # Small table to prevent "Bad Message"
        
    with col2:
        st.subheader("📈 Historical Trend")
        fig = go.Figure()
        # Limit plot points for speed
        plot_df = data.tail(120)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], line=dict(color='#A0522D')))
        fig.update_layout(template="plotly_white", height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # 6. Optimized AI Prediction
    if st.button("🚀 Run AI Prediction", type="primary"):
        if len(data) < seq_len:
            st.error(f"Need {seq_len} days of data, found {len(data)}.")
        else:
            # Container helps prevent buffer issues
            with st.container():
                with st.spinner("AI Brain Training (Optimized Speed)..."):
                    # Use last 300 days for training context
                    train_data = data.tail(300)
                    X, y, scaler = prepare_data(train_data, seq_len, pred_len)
                    model = build_lstm_model(seq_len, pred_len)
                    
                    # 3 Epochs = Fast & Responsive for Web App
                    model.fit(X, y, epochs=3, batch_size=32, verbose=0)
                    
                    last_seq = data.values[-seq_len:]
                    prediction = predict_future(model, last_seq, scaler)
                    
                    st.success("✅ Prediction generated!")
                    
                    # Forecast Chart
                    pred_dates = pd.date_range(start=data.index[-1], periods=pred_len+1)[1:]
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=pred_dates, y=prediction, mode='lines+markers', line=dict(color='#6B8E23')))
                    fig_pred.update_layout(template="plotly_white", title="7-Day Forecast")
                    st.plotly_chart(fig_pred, use_container_width=True)