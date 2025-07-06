import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model

# Positional Embedding for Transformer
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

# Transformer Block
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

# Load preprocessed data
data = pd.read_csv("eurusd_1m_preprocessed_data.csv")

# Prepare data for Transformer model
# Features: Open, High, Low, Close, Volume, Spread, Volatility
features = ["Open", "High", "Low", "Close", "Volume", "Spread", "Volatility"]

# Use a lookback window for sequence input
lookback_window = 60 # 60 minutes of data
prediction_horizon = 10 # Predict 10 future candles (Close prices)

X, y = [], []
for i in range(len(data) - lookback_window - prediction_horizon):
    X.append(data[features].iloc[i : i + lookback_window].values)
    y.append(data["Close"].iloc[i + lookback_window : i + lookback_window + prediction_horizon].values)

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets (simple split for demonstration)
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define Transformer model parameters
embed_dim = X.shape[-1]  # Number of features
num_heads = 2
ff_dim = 32

inputs = Input(shape=(lookback_window, embed_dim))
x = PositionalEmbedding(lookback_window, embed_dim)(inputs)
x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(prediction_horizon, activation="linear")(x) # Output 10 future close prices

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("transformer_forex_predictor.h5")
print("Transformer model trained and saved to transformer_forex_predictor.h5")

