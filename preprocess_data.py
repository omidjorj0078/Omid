import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the raw data
data = pd.read_csv("eurusd_1m_historical_data.csv")

# Convert 'Datetime' column to datetime objects
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Set 'Datetime' as index
data = data.set_index('Datetime')

# Calculate spread (assuming Bid and Ask are not available, using High-Low as proxy for range)
data['Spread'] = data['High'] - data['Low']

# Calculate volatility (e.g., using rolling standard deviation of returns)
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=20).std()

# Fill NaN values created by rolling window (e.g., with 0 or forward fill)
data['Volatility'] = data['Volatility'].fillna(0)

# Select features for scaling and model input
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'Volatility']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the features
data[features] = scaler.fit_transform(data[features])

# Save the preprocessed data
data.to_csv("eurusd_1m_preprocessed_data.csv")

print("Data preprocessed and saved to eurusd_1m_preprocessed_data.csv")

