import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.stock_functions import *

# Page config
st.set_page_config(
    page_title="Stock Market Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Sidebar
st.sidebar.title("Stock Analysis Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
days = st.sidebar.slider("Days of Historical Data", 30, 365, 180)

# Date range
end_date = datetime.now()
start_date = end_date - timedelta(days=days)

# Main page
st.title("Stock Market Analysis Dashboard")
st.write("Analyze stock market data with interactive visualizations")

# Load data
try:
    df = get_stock_data(ticker, start_date, end_date)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${df['Close'][-1]:.2f}")
    with col2:
        price_change = df['Close'][-1] - df['Close'][-2]
        st.metric("Price Change", f"${price_change:.2f}")
    with col3:
        st.metric("Volume", f"{df['Volume'][-1]:,.0f}")
    with col4:
        high_low = f"${df['High'][-1]:.2f} / ${df['Low'][-1]:.2f}"
        st.metric("High/Low", high_low)

    # Candlestick chart
    st.plotly_chart(create_candlestick_chart(df), use_container_width=True)
    
    # Moving averages
    df = calculate_moving_averages(df)
    
    # Technical indicators
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20-day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='200-day MA'))
    
    fig.update_layout(
        title='Price and Moving Averages',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    st.plotly_chart(create_volume_chart(df), use_container_width=True)
    
    # Data table
    st.subheader("Historical Data")
    st.dataframe(df.sort_index(ascending=False).head())

except Exception as e:
    st.error(f"Error: {e}")
    st.write("Please check the stock ticker and try again.")