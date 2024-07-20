- Student ID: 20120602
- Fullname: Nguyễn Minh Trí

### Overview

This project involves fetching cryptocurrency price data, training Long Short-Term Memory (LSTM) models for price prediction, and visualizing the results using a web application built with Dash. The project covers three cryptocurrencies: Bitcoin (BTC), Ethereum (ETH), and Cardano (ADA).

### File Structure

-   `fetch_crypto_data.py`: Script to fetch historical price data from Binance API and save it to CSV files.
-   `stock_prediction.ipynb`: Jupyter Notebook to train LSTM models for cryptocurrency price prediction.
-   `stock_app.py`: Dash application to visualize actual and predicted prices.

### Requirements

-   Python 3.8+
-   Libraries: pandas, numpy, requests, matplotlib, keras, sklearn, dash, plotly

### Setup and Usage

#### 1\. Fetch Cryptocurrency Data

1.  Run `fetch_crypto_data.py` to fetch historical price data for BTC, ETH, and ADA from Binance API.

    `python fetch_crypto_data.py`

2.  This script saves the data to CSV files:

    -   `./data/BTC-USD.csv`
    -   `./data/ETH-USD.csv`
    -   `./data/ADA-USD.csv`

#### 2\. Train LSTM Models

1.  Open and run the `stock_prediction.ipynb` notebook.
2.  This notebook will:
    -   Load the CSV data.
    -   Preprocess the data.
    -   Train LSTM models for each cryptocurrency.
    -   Save the trained models to:
        -   `./model/BTC-USD_lstm_model.h5`
        -   `./model/ETH-USD_lstm_model.h5`
        -   `./model/ADA-USD_lstm_model.h5`

#### 3\. Run Dash Application

1.  Run `stock_app.py` to start the Dash application.

    `python stock_app.py`

2.  The application will display:

    -   Actual and predicted closing prices for the selected cryptocurrency.
    -   Interactive visualizations using Plotly.
