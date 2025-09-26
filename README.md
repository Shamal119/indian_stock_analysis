# Indian Stock Market Analysis Dashboard

A powerful, interactive dashboard built with Streamlit for analyzing the Indian stock market, featuring real-time data, technical analysis, and price predictions.

## Features

- **Market Overview**: Get a snapshot of the top-performing stocks from the Nifty 50 index over a customizable time period (7 to 365 days).
- **Individual Stock Analysis**: Dive deep into any publicly listed Indian stock by providing its ticker symbol (e.g., `RELIANCE.NS`).
- **Interactive Candlestick Charts**: Visualize price movements with detailed candlestick charts that support zooming and panning for in-depth analysis.
- **Technical Indicators**: Overlay popular technical indicators, including 20-day, 50-day, and 200-day Simple Moving Averages (SMA).
- **Volume Analysis**: Analyze trading volume with an integrated bar chart to gauge market sentiment and confirm trends.
- **Price Prediction**: Leverage a Long Short-Term Memory (LSTM) neural network to forecast the stock's price for the next 30 days.
- **Latest News**: Stay informed with the latest news headlines and links to articles for the selected stock, fetched in real-time.
- **Historical Data**: View and sort a table of the most recent historical data for any stock.

## Technologies Used

- **Framework**: [Streamlit](https://streamlit.io/)
- **Data Retrieval**: [yfinance](https://pypi.org/project/yfinance/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization**: [Plotly](https://plotly.com/)
- **Machine Learning**: [TensorFlow](https://www.tensorflow.org/) (with Keras), [Scikit-learn](https://scikit-learn.org/)

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.