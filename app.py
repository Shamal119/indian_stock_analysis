import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from src.stock_functions import *

st.set_page_config(page_title="Indian Stock Analysis", layout="wide")

st.title("Indian Stock Market Analysis Dashboard")

# Sidebar
st.sidebar.title("Analysis Parameters")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Market Overview", "Individual Stock Analysis"]
)

if analysis_type == "Market Overview":
    st.header("Top Performing Stocks")
    days = st.slider("Select Time Period (Days)", 7, 365, 30)
    
    top_performers = get_best_performers(days)
        
    st.dataframe(top_performers)
    
    # Plot top 5 performers
    fig = go.Figure()
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    for symbol in top_performers['Symbol'][:5]:
        df = get_stock_data(symbol, start_date, end_date)
        if df is not None:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=symbol))
    
    fig.update_layout(title="Top 5 Performers Price Movement")
    st.plotly_chart(fig)

else:
    ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")
    days = st.sidebar.slider("Days of Historical Data", 30, 365, 180)
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    df = get_stock_data(ticker, start_date, end_date)
        
    if df is not None:
        # Current Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"₹{df['Close'][-1]:.2f}")
        with col2:
            price_change = df['Close'][-1] - df['Close'][-2]
            st.metric("Price Change", f"₹{price_change:.2f}")
        with col3:
            st.metric("Volume", f"{df['Volume'][-1]:,.0f}")
        with col4:
            st.metric("Volatility", f"{df['Volatility'][-1]:.2%}")

        # Technical Analysis Chart
        st.plotly_chart(create_analysis_chart(df), use_container_width=True)

        # Price Prediction
        st.subheader("Price Prediction (Next 30 Days)")
        predictions = predict_stock_price(df)
        future_dates = pd.date_range(start=df.index[-1], periods=31)[1:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical'))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), name='Predicted'))
        fig.update_layout(title="Price Prediction", template='plotly_dark')
        st.plotly_chart(fig)

        # Latest News
        st.subheader("Latest News")
        news_df = get_stock_news(ticker)
        if not news_df.empty:
            for _, news in news_df.iterrows():
                st.write(f"**{news['title']}**")
                st.write(f"Publisher: {news['publisher']} | [Read More]({news['link']})")
                st.write("---")
        else:
            st.write("No recent news available")

        # Historical Data Table
        st.subheader("Historical Data")
        st.dataframe(df.sort_index(ascending=False).head())
        else:
            st.error("Error fetching data. Please check the stock symbol.")