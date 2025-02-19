import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
import warnings

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


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


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def predict_stock_price(df, days_to_predict=30, window_size=60):
    """
    Predict future stock prices using an LSTM model that incorporates additional factors.
    In addition to 'Close' prices, we include 'Volume' and 'Volatility' as input features.
    
    Parameters:
        df (pd.DataFrame): Historical stock data containing at least the columns:
                           'Close', 'Volume', and 'Volatility'.
        days_to_predict (int): Number of future days to predict.
        window_size (int): Number of past days to use as input.
    
    Returns:
        np.array: Array of predicted prices (original scale) for the next `days_to_predict` days.
    """
    # Ensure there's enough data
    if len(df) < window_size:
        raise ValueError("Not enough data to train the LSTM model. Increase the historical data range.")
    
    # Select and clean the features
    features = ['Close', 'Volume', 'Volatility']
    df_features = df[features].dropna()  # Remove rows with missing values
    data = df_features.values  # Shape: (n_samples, 3)
    
    # Scale the features (all at once) using MinMaxScaler.
    # Note: We use one scaler so we can later invert the prediction for 'Close'
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create training sequences
    X_train, y_train = [], []
    for i in range(window_size, len(scaled_data)):
        # Each input sample is a sequence of vectors over the window_size period
        X_train.append(scaled_data[i-window_size:i])
        # The target is the 'Close' price (first feature) at time i
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape X_train if necessary (it should be of shape: [samples, window_size, num_features])
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, len(features))))
    model.add(LSTM(50))
    model.add(Dense(1))  # Output: predicted scaled 'Close' price
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Begin iterative forecasting
    predictions_scaled = []
    # Start with the last window from the available data
    current_sequence = scaled_data[-window_size:].copy()
    
    for _ in range(days_to_predict):
        # Reshape current sequence to (1, window_size, num_features)
        current_sequence_reshaped = np.reshape(current_sequence, (1, window_size, len(features)))
        # Predict the next day's scaled 'Close' price
        next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
        predictions_scaled.append(next_pred_scaled)
        
        # Prepare the next input vector:
        # For the additional features (Volume and Volatility), we'll assume they remain as in the last observed day.
        last_observed = current_sequence[-1].copy()  # Contains [Close, Volume, Volatility]
        next_day = last_observed.copy()
        next_day[0] = next_pred_scaled  # Update the 'Close' price with the prediction
        
        # Slide the window: drop the first day and append the new predicted day
        current_sequence = np.append(current_sequence[1:], [next_day], axis=0)
    
    # Inverse transform the scaled 'Close' predictions back to the original scale.
    # Since 'Close' is the first feature, we use the corresponding min and range.
    close_min = scaler.data_min_[0]
    close_range = scaler.data_range_[0]
    predictions = np.array(predictions_scaled) * close_range + close_min
    
    return predictions


# def predict_stock_price(df, days_to_predict=30, window_size=60):
#     """
#     Predict future stock prices using an LSTM model.
    
#     Parameters:
#         df (pd.DataFrame): Historical stock data with a 'Close' column.
#         days_to_predict (int): Number of future days to predict.
#         window_size (int): Number of past days to use for each prediction input.
        
#     Returns:
#         np.array: Array of predicted prices for the next `days_to_predict` days.
#     """
#     # Ensure there's enough data
#     if len(df) < window_size:
#         raise ValueError("Not enough data to train the LSTM model. Increase the historical data range.")
    
#     # Prepare the dataset using the 'Close' prices
#     close_prices = df['Close'].values.reshape(-1, 1)
    
#     # Scale the data to the [0, 1] range
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(close_prices)
    
#     # Create the training sequences
#     X_train, y_train = [], []
#     for i in range(window_size, len(scaled_data)):
#         X_train.append(scaled_data[i - window_size:i, 0])
#         y_train.append(scaled_data[i, 0])
#     X_train, y_train = np.array(X_train), np.array(y_train)
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
#     # Build the LSTM model
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
    
#     # Train the model (adjust epochs and batch_size as needed)
#     model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
#     # Predict future prices using iterative forecasting
#     predictions = []
#     last_sequence = scaled_data[-window_size:]
#     current_sequence = last_sequence.copy()
    
#     for _ in range(days_to_predict):
#         current_sequence_reshaped = np.reshape(current_sequence, (1, window_size, 1))
#         next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)
#         predictions.append(next_pred_scaled[0, 0])
#         # Append the prediction and slide the window
#         current_sequence = np.append(current_sequence[1:], [[next_pred_scaled[0, 0]]], axis=0)
    
#     # Inverse transform the scaled predictions back to original prices
#     predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
#     return predictions



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