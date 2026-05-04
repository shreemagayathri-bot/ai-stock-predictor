import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_logic import prepare_data, build_lstm_model, predict_future

# 1. Page Config
st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")

# 2. Midnight Blue Theme with White Text
st.markdown("""
    <style>
    /* Main background - Midnight Blue */
    .stApp { 
        background-color: #0A192F !important; 
    }
    
    /* Sidebar background - Slightly lighter blue */
    section[data-testid="stSidebar"] { 
        background-color: #112240 !important; 
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #172A45 !important;
        border: 1px solid #64FFDA !important; /* Cyan accent border */
        border-radius: 10px !important;
        padding: 15px !important;
    }

    /* Force all text to white */
    h1, h2, h3, h4, h5, h6, p, div, label, .stMetricLabel, .stMetricValue, .stMarkdown {
        color: #FFFFFF !important;
    }

    /* Sidebar specific text and sliders */
    section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] p {
        color: #FFFFFF !important;
    }

    /* Button Styling - Cyan/Teal accent */
    div.stButton > button {
        background-color: #64FFDA !important;
        color: #0A192F !important; /* Dark text on bright button */
        border: none !important;
        font-weight: bold !important;
    }
    
    div.stButton > button:hover {
        background-color: #52d1b2 !important;
        color: #0A192F !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
st.sidebar.title("🛠️ Control Panel")
ticker_input = st.sidebar.text_input("Stock Symbol", "AAPL")
seq_len = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
pred_len = st.sidebar.slider("Prediction Horizon", 1, 14, 7)

# 4. Data Logic
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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'High', 'Low', 'Close']]
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# 5. UI Layout
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
            fig.add_trace(go.Scatter(x=data.index[-100:], y=data['Close'][-100:], name="Close", line=dict(color='#64FFDA', width=2)))
            # Using 'plotly_dark' as it fits blue themes better than 'plotly_white'
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Run AI Prediction", type="primary"):
            if len(data) < seq_len:
                st.error(f"Need {seq_len} days of data, found {len(data)}.")
            else:
                with st.spinner("Training AI..."):
                    X, y, scaler = prepare_data(data, seq_len, pred_len)
                    model = build_lstm_model(seq_len, pred_len)
                    model.fit(X[-50:], y[-50:], epochs=2, batch_size=16, verbose=0)
                    
                    last_seq = data.values[-seq_len:]
                    prediction = predict_future(model, last_seq, scaler)
                    
                    st.success("✅ Prediction generated!")
                    pred_dates = pd.date_range(start=data.index[-1], periods=pred_len+1)[1:]
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=pred_dates, y=prediction, mode='lines+markers', name="Forecast", line=dict(color='#ADFF2F', width=3)))
                    fig_pred.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_pred, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {e}")