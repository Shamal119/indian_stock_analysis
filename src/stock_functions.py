import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def get_nifty50_symbols():
    """Get list of current Nifty 50 stocks"""
    # In a real-world scenario, this list should be fetched dynamically
    # from a reliable source, as the Nifty 50 constituents can change.
    symbols = [
        'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
        'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS',
        'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFC.NS',
        'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
        'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS',
        'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBIN.NS',
        'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS',
        'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS'
    ]
    return symbols

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Get stock data with error handling"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            print(f"Failed to get data for {ticker}: No data available.")
            return None
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_data
def get_best_performers(days=30):
    """Get best performing stocks in the last n days using a single batch download."""
    symbols = get_nifty50_symbols()
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    # Fetch all data in a single batch request
    try:
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', progress=False)
        if data.empty:
            return pd.DataFrame(columns=['Symbol', 'Returns', 'Current_Price', 'Volume'])
    except Exception as e:
        print(f"Error during batch download: {e}")
        return pd.DataFrame(columns=['Symbol', 'Returns', 'Current_Price', 'Volume'])

    performance = []
    for symbol in symbols:
        df = data[symbol]
        df = df.dropna(subset=['Close']) # Ensure 'Close' prices are available
        if not df.empty:
            try:
                start_price = df['Close'].iloc[0]
                end_price = df['Close'].iloc[-1]
                returns = ((end_price / start_price) - 1) * 100
                performance.append({
                    'Symbol': symbol,
                    'Returns': returns,
                    'Current_Price': end_price,
                    'Volume': df['Volume'].iloc[-1]
                })
            except IndexError:
                print(f"Not enough data for {symbol} to calculate performance.")
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

    if performance:
        return pd.DataFrame(performance).sort_values('Returns', ascending=False)
    else:
        return pd.DataFrame(columns=['Symbol', 'Returns', 'Current_Price', 'Volume'])

@st.cache_data
def predict_stock_price(df, days_to_predict=30, window_size=60):
    """
    Predict future stock prices using an LSTM model that incorporates additional factors.
    In addition to 'Close' prices, we include 'Volume' and 'Volatility' as input features.
    """
    if len(df) < window_size:
        raise ValueError("Not enough data to train the LSTM model. Increase the historical data range.")
    
    features = ['Close', 'Volume', 'Volatility']
    df_features = df[features].dropna()
    data = df_features.values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X_train, y_train = [], []
    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i-window_size:i])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, len(features))),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    predictions_scaled = []
    current_sequence = scaled_data[-window_size:].copy()
    
    for _ in range(days_to_predict):
        current_sequence_reshaped = np.reshape(current_sequence, (1, window_size, len(features)))
        next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
        predictions_scaled.append(next_pred_scaled)
        
        last_observed = current_sequence[-1].copy()
        next_day = last_observed.copy()
        next_day[0] = next_pred_scaled
        
        current_sequence = np.append(current_sequence[1:], [next_day], axis=0)
    
    close_min = scaler.data_min_[0]
    close_range = scaler.data_range_[0]
    predictions = np.array(predictions_scaled) * close_range + close_min
    
    return predictions

@st.cache_data
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