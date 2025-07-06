import socket
import numpy as np
import tensorflow as tf

# Load the trained models
transformer_model = tf.keras.models.load_model("transformer_forex_predictor.h5")
rl_agent_model = tf.keras.models.load_model("rl_trading_agent.h5")

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    try:
        while True:
            # Receive data from MQL5
            data = conn.recv(4096).decode("utf-8") # Increased buffer size
            if not data:
                break

            # Process the received data (this is a placeholder, real implementation will be more complex)
            # The data from MQL5 should be preprocessed in the same way as the training data
            # For now, let\"s assume we receive a numpy array of the right shape
            market_data = np.fromstring(data, sep=",")
            # Reshape for the model (60 bars * 7 features)
            market_data = np.reshape(market_data, (1, 60, 7)) 

            # Get prediction from Transformer model
            future_candles = transformer_model.predict(market_data)
            # Flatten the prediction and convert to string for sending
            future_candles_str = ",".join(map(str, future_candles.flatten()))

            # Get trading signal, SL, and TP from RL agent
            # The state for the RL agent needs to be constructed from the received data
            # This is a simplified example
            # The RL agent expects a flattened state, let\"s use the last market data point and some dummy values
            # This needs to be aligned with the state definition in train_rl_agent.py
            # For now, let\"s use a simplified state for demonstration
            # The state size in train_rl_agent.py is `env._get_state().shape[0]` which is `lookback_window * 5 + 3`
            # (60 * 5 + 3 = 303)
            state_for_rl = np.append(market_data.flatten(), [0, 0, 0]) # Placeholder for balance, position, current_price
            state_for_rl = np.reshape(state_for_rl, (1, 303)) # Reshape for the RL model

            action, sl, tp = rl_agent_model.predict(state_for_rl)[0]
            action = np.argmax(action) # Get the action from the RL agent output

            # Combine signal, future candles prediction, SL, and TP
            response_to_mql5 = f"{action}|{future_candles_str}|{sl}|{tp}"
            conn.sendall(response_to_mql5.encode("utf-8"))

    finally:
        print(f"[CONNECTION CLOSED] {addr}")
        conn.close()

def main():
    host = "127.0.0.1"
    port = 5000

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"[LISTENING] Server is listening on {host}:{port}")

    while True:
        conn, addr = server.accept()
        handle_client(conn, addr)

if __name__ == "__main__":
    main()

