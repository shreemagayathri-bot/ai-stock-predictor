import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_logic import prepare_data, build_lstm_model, predict_future

# 1. Page Config
st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")

# 2. Earthy Professional Styling
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
    div, p, label, .stMetricLabel, .stMetricValue { color: #2F4F4F !important; }
    div.stButton > button {
        background-color: #A0522D !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
st.sidebar.title("🛠️ Control Panel")
ticker = st.sidebar.text_input("Stock Symbol", "AAPL")
seq_len = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
pred_len = st.sidebar.slider("Prediction Horizon", 1, 14, 7)

# 4. Data Logic
@st.cache_data
# 4. Data Logic
@st.cache_data
def get_data(ticker_symbol):
    try:
        # Create Ticker object
        ticker_obj = yf.Ticker(ticker_symbol.strip().upper())
        
        # history() is more robust for cloud deployments than download()
        df = ticker_obj.history(period="5y")
        
        if df.empty:
            return pd.DataFrame()
            
        # Standardize columns: yfinance history() returns Title Case (Close, Open)
        # We ensure they match your model's expected lowercase or title case logic
        df = df[['Open', 'High', 'Low', 'Close']]
        
        # Drop timezone info to prevent Plotly/Pandas date errors
        df.index = df.index.tz_localize(None)
        
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker_symbol}: {e}")
        return pd.DataFrame()

# 5. UI Layout
st.title("🚀 AI-Powered Stock Forecasting")
st.markdown("### 🧠 Multistep-Multivariate LSTM Neural Network")

try:
    data = get_data(ticker)
    
    if data.empty:
        st.error("❌ No data found. Check ticker.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Market Overview")
            last_price = data['Close'].iloc[-1].item()
            st.metric("Latest Close", f"${last_price:.2f}")
            st.write(data.tail(5))
            
        with col2:
            st.subheader("📈 Historical Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[-100:], y=data['Close'][-100:], 
                                     name="Close Price", line=dict(color='#A0522D')))
            fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Run AI Prediction", type="primary"):
            # SAFETY CHECK: Prevent the reshape error
            if len(data) < seq_len:
                st.error(f"❌ Not enough data! You need at least {seq_len} days of data for this Lookback.")
            else:
                with st.spinner("Training LSTM..."):
                    X, y, scaler = prepare_data(data, seq_len, pred_len)
                    model = build_lstm_model(seq_len, pred_len)
                    model.fit(X[-50:], y[-50:], epochs=1, batch_size=16, verbose=0)
                    
                    last_seq = data.values[-seq_len:]
                    # Pass the dynamic seq_len to the prediction function
                    prediction = predict_future(model, last_seq, scaler)
                    
                    st.success("✅ Prediction generated!")
                    st.subheader("🔮 Forecast Results")
                    
                    pred_dates = pd.date_range(start=data.index[-1], periods=pred_len+1)[1:]
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=pred_dates, y=prediction, 
                                                  mode='lines+markers', name="Forecast",
                                                  line=dict(color='#6B8E23', width=3)))
                    fig_pred.update_layout(template="plotly_white")
                    st.plotly_chart(fig_pred, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {e}")