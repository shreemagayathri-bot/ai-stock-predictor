import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_logic import prepare_data, build_lstm_model, predict_future

# 1. Page Config & Force Light Theme for Speed
st.set_page_config(page_title="AI Predictor", layout="wide")

# 2. Sidebar - TSLA Default
st.sidebar.header("🛠️ Settings")
ticker_input = st.sidebar.text_input("Ticker", "TSLA").strip().upper()
# Smaller lookback = Faster training
seq_len = st.sidebar.slider("Lookback", 30, 60, 45) 
pred_len = 7 

# 3. Fast Data Fetching (Limited to 1 year for speed)
@st.cache_data(ttl=600)
def get_clean_data(symbol):
    try:
        # Download only 1 year to keep the 'message' small
        df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        
        # Flatten MultiIndex immediately
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Close']].copy() # Only use Close price for maximum speed
        df.index = df.index.tz_localize(None)
        return df
    except:
        return None

# 4. Main App Logic
st.title("📈 AI Stock Forecast")

data = get_clean_data(ticker_input)

if data is None:
    st.error("Waiting for data... (Check ticker or internet)")
else:
    # Quick Metric
    last_val = data['Close'].iloc[-1]
    st.metric(f"{ticker_input} Price", f"${last_val:.2f}")

    # Plotly Optimization: Use thin lines, no heavy markers
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:], line=dict(width=2)))
    fig.update_layout(height=300, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)

    # 5. The "Run" Section - Designed not to crash
    if st.button("🚀 Predict Next 7 Days"):
        # We wrap this in a container to prevent "Bad Message" errors
        with st.container():
            with st.spinner("Processing..."):
                # Fast Training: Only use the last 200 days
                train_df = data.tail(200)
                
                # Model Logic
                X, y, scaler = prepare_data(train_df, seq_len, pred_len)
                model = build_lstm_model(seq_len, pred_len)
                
                # ONLY 3 EPOCHS: Enough for a demo, fast enough not to time out
                model.fit(X, y, epochs=3, batch_size=32, verbose=0)
                
                # Predict
                last_seq = train_df.values[-seq_len:]
                prediction = predict_future(model, last_seq, scaler)
                
                # Success UI
                st.success(f"Forecast for {ticker_input} complete!")
                
                # Simple Prediction Chart
                pdf = pd.DataFrame(prediction, columns=['Price'])
                st.line_chart(pdf)