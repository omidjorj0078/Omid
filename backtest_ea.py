import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Positional Embedding for Transformer (re-definition for loading)
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-2]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

# Transformer Block (re-definition for loading)
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# Load the preprocessed data
data = pd.read_csv("eurusd_1m_preprocessed_data.csv")

# Load the trained models with custom objects
custom_objects_transformer = {
    "PositionalEmbedding": PositionalEmbedding,
    "TransformerBlock": TransformerBlock,
    "mse": tf.keras.losses.MeanSquaredError() # Explicitly define mse
}

custom_objects_rl = {
    "Adam": Adam,
    "mse": tf.keras.losses.MeanSquaredError() # Explicitly define mse for RL model
}

transformer_model = tf.keras.models.load_model("transformer_forex_predictor.h5", custom_objects=custom_objects_transformer)
rl_agent_model = tf.keras.models.load_model("rl_trading_agent.h5", custom_objects=custom_objects_rl)

# Define trading parameters
initial_balance = 10000.0
lot_size = 0.01 # Fixed lot size for simplicity in backtesting
leverage = 500 # Example leverage

# Initialize trading state
balance = initial_balance
position = 0 # 0: no position, 1: long, -1: short
entry_price = 0.0
profit_loss = 0.0
trades = []

lookback_window = 60 # Same as used in training

# Function to simulate trade execution
def execute_trade(signal, current_price, dynamic_sl, dynamic_tp):
    global balance, position, entry_price, profit_loss

    if signal == 1: # Buy
        if position == 0:
            position = 1
            entry_price = current_price
            print(f"BUY at {current_price:.5f}, SL: {current_price - dynamic_sl:.5f}, TP: {current_price + dynamic_tp:.5f}")
            trades.append({"type": "BUY", "entry_price": entry_price, "sl": current_price - dynamic_sl, "tp": current_price + dynamic_tp})
        else:
            print("Already in a position, skipping BUY.")
    elif signal == 2: # Sell
        if position == 0:
            position = -1
            entry_price = current_price
            print(f"SELL at {current_price:.5f}, SL: {current_price + dynamic_sl:.5f}, TP: {current_price - dynamic_tp:.5f}")
            trades.append({"type": "SELL", "entry_price": entry_price, "sl": current_price + dynamic_sl, "tp": current_price - dynamic_tp})
        else:
            print("Already in a position, skipping SELL.")
    elif signal == 0: # Close position
        if position != 0:
            if position == 1: # Close long
                pnl = (current_price - entry_price) * 100000 * lot_size # Assuming 100,000 units per lot
            else: # Close short
                pnl = (entry_price - current_price) * 100000 * lot_size
            balance += pnl
            profit_loss += pnl
            print(f"CLOSE at {current_price:.5f}, PnL: {pnl:.2f}, New Balance: {balance:.2f}")
            trades[-1]["exit_price"] = current_price
            trades[-1]["pnl"] = pnl
            position = 0
            entry_price = 0.0
        else:
            print("No open position to close.")

# Backtesting loop
for i in range(lookback_window, len(data) - 1):
    current_data_slice = data.iloc[i - lookback_window : i]
    current_market_data = current_data_slice[["Open", "High", "Low", "Close", "Volume", "Spread", "Volatility"]].values
    current_price = data["Close"].iloc[i]

    # Reshape for Transformer model
    transformer_input = np.reshape(current_market_data, (1, lookback_window, 7))

    # Get future candle prediction (not directly used for trading, but for display)
    future_candles = transformer_model.predict(transformer_input)

    # Prepare state for RL agent
    # The state size in train_rl_agent.py is `lookback_window * 7 + 3` (60 * 7 + 3 = 423)
    state_for_rl = np.append(current_market_data.flatten(), [balance, position, current_price])
    state_for_rl = np.reshape(state_for_rl, (1, lookback_window * 7 + 3))

    # Get action, SL, TP from RL agent
    rl_output = rl_agent_model.predict(state_for_rl)[0]
    action = np.argmax(rl_output[:3]) # First 3 elements are actions
    dynamic_sl = rl_output[3] # SL value
    dynamic_tp = rl_output[4] # TP value

    # Simulate trade execution
    execute_trade(action, current_price, dynamic_sl, dynamic_tp)

    # Simulate SL/TP hit (simplified)
    if position == 1: # Long position
        sl_price = trades[-1]["sl"]
        tp_price = trades[-1]["tp"]
        if current_price <= sl_price:
            print(f"SL hit for BUY at {current_price:.5f}")
            pnl = (current_price - entry_price) * 100000 * lot_size
            balance += pnl
            profit_loss += pnl
            trades[-1]["exit_price"] = current_price
            trades[-1]["pnl"] = pnl
            position = 0
            entry_price = 0.0
        elif current_price >= tp_price:
            print(f"TP hit for BUY at {current_price:.5f}")
            pnl = (current_price - entry_price) * 100000 * lot_size
            balance += pnl
            profit_loss += pnl
            trades[-1]["exit_price"] = current_price
            trades[-1]["pnl"] = pnl
            position = 0
            entry_price = 0.0
    elif position == -1: # Short position
        sl_price = trades[-1]["sl"]
        tp_price = trades[-1]["tp"]
        if current_price >= sl_price:
            print(f"SL hit for SELL at {current_price:.5f}")
            pnl = (entry_price - current_price) * 100000 * lot_size
            balance += pnl
            profit_loss += pnl
            trades[-1]["exit_price"] = current_price
            trades[-1]["pnl"] = pnl
            position = 0
            entry_price = 0.0
        elif current_price <= tp_price:
            print(f"TP hit for SELL at {current_price:.5f}")
            pnl = (entry_price - current_price) * 100000 * lot_size
            balance += pnl
            profit_loss += pnl
            trades[-1]["exit_price"] = current_price
            trades[-1]["pnl"] = pnl
            position = 0
            entry_price = 0.0

# Final results
print("\n--- Backtesting Results ---")
print(f"Initial Balance: {initial_balance:.2f}")
print(f"Final Balance: {balance:.2f}")
print(f"Total PnL: {profit_loss:.2f}")
print(f"Number of Trades: {len(trades)}")

# Calculate win rate
winning_trades = [trade for trade in trades if "pnl" in trade and trade["pnl"] > 0]
win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
print(f"Win Rate: {win_rate:.2f}%")

# Save trades to CSV for detailed analysis
trades_df = pd.DataFrame(trades)
trades_df.to_csv("backtest_trades.csv", index=False)
print("Detailed trades saved to backtest_trades.csv")

