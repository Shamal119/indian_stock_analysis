import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

def get_nifty50_symbols():
    """Get list of current Nifty 50 stocks"""
    symbols = [
        'ADANIPORTS.NS',  # Adani Ports
        'ASIANPAINT.NS',  # Asian Paints
        'AXISBANK.NS',    # Axis Bank
        'BAJFINANCE.NS',  # Bajaj Finance
        'BAJAJFINSV.NS',  # Bajaj Finserv
        'BHARTIARTL.NS',  # Bharti Airtel
        'BRITANNIA.NS',   # Britannia Industries
        'CIPLA.NS',       # Cipla
        'COALINDIA.NS',   # Coal India
        'DIVISLAB.NS',    # Divi's Laboratories
        'DRREDDY.NS',     # Dr. Reddy's Laboratories
        'EICHERMOT.NS',   # Eicher Motors
        'GRASIM.NS',      # Grasim Industries
        'HCLTECH.NS',     # HCL Technologies
        'HDFCBANK.NS',    # HDFC Bank
        'HDFC.NS',        # HDFC
        'HEROMOTOCO.NS',  # Hero MotoCorp
        'HINDALCO.NS',    # Hindalco Industries
        'HINDUNILVR.NS',  # Hindustan Unilever
        'ICICIBANK.NS',   # ICICI Bank
        'INDUSINDBK.NS',  # IndusInd Bank
        'INFY.NS',        # Infosys
        'ITC.NS',         # ITC
        'JSWSTEEL.NS',    # JSW Steel
        'KOTAKBANK.NS',   # Kotak Mahindra Bank
        'LT.NS',          # Larsen & Toubro
        'M&M.NS',         # Mahindra & Mahindra
        'MARUTI.NS',      # Maruti Suzuki
        'NESTLEIND.NS',   # Nestlé India
        'NTPC.NS',        # NTPC
        'ONGC.NS',        # ONGC
        'POWERGRID.NS',   # Power Grid Corporation of India
        'RELIANCE.NS',    # Reliance Industries
        'SBIN.NS',        # State Bank of India
        'SUNPHARMA.NS',   # Sun Pharmaceutical
        'TATAMOTORS.NS',  # Tata Motors
        'TATASTEEL.NS',   # Tata Steel
        'TECHM.NS',       # Tech Mahindra
        'TITAN.NS',       # Titan Company
        'ULTRACEMCO.NS',  # UltraTech Cement
        'UPL.NS',         # UPL
        'WIPRO.NS'        # Wipro
    ]
    return symbols

def get_stock_data(ticker, start_date, end_date):
    """Get stock data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            print(f"Failed to get data for {ticker}: No data available.")
            return None
        
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def get_best_performers(days=30):
    """Get best performing stocks in the last n days"""
    symbols = get_nifty50_symbols()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    performance = []
    
    for symbol in symbols:
        df = get_stock_data(symbol, start_date, end_date)
        
        if df is not None and not df.empty and 'Close' in df.columns:
            try:
                returns = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                performance.append({
                    'Symbol': symbol,
                    'Returns': returns,
                    'Current_Price': df['Close'].iloc[-1],
                    'Volume': df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
                })
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

    if performance:
        return pd.DataFrame(performance).sort_values('Returns', ascending=False)
    else:
        return pd.DataFrame(columns=['Symbol', 'Returns', 'Current_Price', 'Volume'])  # Return empty DF if no data


def predict_stock_price(df, days_to_predict=30):
    """Simple prediction using moving average"""
    last_price = df['Close'].iloc[-1]
    ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
    ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
    
    # Simple trend-based prediction
    trend = (ma20 - ma50) / ma50
    predictions = []
    current_price = last_price
    
    for _ in range(days_to_predict):
        next_price = current_price * (1 + trend)
        predictions.append(next_price)
        current_price = next_price
    
    return np.array(predictions).reshape(-1, 1)


def get_stock_news(ticker):
    """Get latest news about the stock"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return pd.DataFrame(news)[['title', 'publisher', 'link', 'providerPublishTime']]
    except:
        return pd.DataFrame()

def create_analysis_chart(df):
    """Create comprehensive analysis chart"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add Moving Averages
    for window in [20, 50, 200]:
        ma = df['Close'].rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ma,
            name=f'{window}-day MA',
            line=dict(width=1)
        ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Stock Analysis',
        yaxis_title='Price (INR)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        template='plotly_dark'
    )
    
    return fig