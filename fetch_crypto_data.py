import requests
import pandas as pd

def fetch_binance_data(symbol, interval, start_date, end_date):
    url = f'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(pd.to_datetime(start_date).timestamp() * 1000),
        'endTime': int(pd.to_datetime(end_date).timestamp() * 1000),
        'limit': 1000
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 
        'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 
        'Taker_buy_quote_asset_volume', 'Ignore'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Adj Close'] = df['Close']  # For simplicity, we can set Adj Close to Close
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Fetch data for BTC-USD, ETH-USD, ADA-USD
btc_data = fetch_binance_data('BTCUSDT', '1d', '2022-01-01', '2024-01-01')
eth_data = fetch_binance_data('ETHUSDT', '1d', '2022-01-01', '2024-01-01')
ada_data = fetch_binance_data('ADAUSDT', '1d', '2022-01-01', '2024-01-01')

# Save to CSV
btc_data.to_csv('./data/BTC-USD.csv', index=False)
eth_data.to_csv('./data/ETH-USD.csv', index=False)
ada_data.to_csv('./data/ADA-USD.csv', index=False)
