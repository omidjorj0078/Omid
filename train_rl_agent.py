import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Load preprocessed data
data = pd.read_csv("eurusd_1m_preprocessed_data.csv")

# Define the trading environment
class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, lookback_window=60):
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.lookback_window = lookback_window
        self.current_step = lookback_window
        self.n_features = data.shape[1] - 1 # Exclude Datetime column
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.lookback_window
        self.position = 0
        self.entry_price = 0
        return self._get_state()

    def _get_state(self):
        history = self.data.iloc[self.current_step - self.lookback_window : self.current_step][["Open", "High", "Low", "Close", "Volume", "Spread", "Volatility"]].values.flatten()
        current_price = self.data["Close"].iloc[self.current_step]
        state = np.append(history, [self.balance, self.position, current_price])
        return state

    def step(self, action):
        # Actions: 0: hold, 1: buy, 2: sell
        self.current_step += 1
        if self.current_step >= len(self.data):
            return self._get_state(), 0, True, {}

        current_price = self.data["Close"].iloc[self.current_step]
        reward = 0
        done = False

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                reward = 0.01 # Small reward for taking an action

        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                reward = 0.01 # Small reward for taking an action

        elif action == 0: # Close position if open, otherwise hold
            if self.position == 1: # Close long
                pnl = (current_price - self.entry_price) * 100000 * 0.01 # Assuming 0.01 lot size
                self.balance += pnl
                reward = pnl # Reward is the PnL
                self.position = 0
                self.entry_price = 0
            elif self.position == -1: # Close short
                pnl = (self.entry_price - current_price) * 100000 * 0.01
                self.balance += pnl
                reward = pnl # Reward is the PnL
                self.position = 0
                self.entry_price = 0

        # End episode if balance is too low or end of data
        if self.balance <= self.initial_balance * 0.9 or self.current_step >= len(self.data) -1:
            done = True
            if self.position != 0: # Close any open position at the end of episode
                if self.position == 1:
                    self.balance += (current_price - self.entry_price) * 100000 * 0.01
                else:
                    self.balance += (self.entry_price - current_price) * 100000 * 0.01

        return self._get_state(), reward, done, {}

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # Replay memory
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(self.state_size,)))
        model.add(Dense(32, activation="relu"))
        # Output layer: 3 actions (hold, buy, sell) + 2 for SL/TP (e.g., percentage of entry price)
        model.add(Dense(self.action_size + 2, activation="linear")) 
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action, random SL/TP (for exploration)
            # SL/TP values are now in pips (e.g., 10-100 pips)
            return random.randrange(self.action_size), random.uniform(0.0010, 0.0100), random.uniform(0.0010, 0.0100)
        
        act_values = self.model.predict(state)
        action = np.argmax(act_values[0][:self.action_size])
        sl_tp_values = act_values[0][self.action_size:]
        
        # Ensure SL/TP are positive and within reasonable bounds (e.g., 10-100 pips)
        # Assuming 1 pip = 0.0001 for EURUSD
        sl = max(0.0010, min(0.0100, abs(sl_tp_values[0])))
        tp = max(0.0010, min(0.0100, abs(sl_tp_values[1])))
        
        return action, sl, tp

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Consider the SL/TP in the reward calculation for the target
                # This is a simplified approach, a more complex reward function would be needed
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0][:self.action_size])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Update SL/TP targets as well (this is a very basic approach)
            # A more sophisticated approach would involve a separate value function for SL/TP
            target_f[0][self.action_size] = target_f[0][self.action_size] # Keep current SL target
            target_f[0][self.action_size + 1] = target_f[0][self.action_size + 1] # Keep current TP target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main training loop
env = TradingEnvironment(data)
# Corrected state_size calculation
state_size = env.lookback_window * 7 + 3 # 7 features (Open, High, Low, Close, Volume, Spread, Volatility) + 3 (balance, position, current_price)
action_size = 3 # Hold, Buy, Sell
agent = DQNAgent(state_size, action_size)

batch_size = 32
epochs = 50 # Increased number of epochs

for e in range(epochs):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    for time in range(env.lookback_window, len(data) -1):
        action, sl, tp = agent.act(state) # Get action, SL, TP
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode: {e}/{epochs}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}, Balance: {env.balance:.2f}")
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Save the RL model
agent.model.save("rl_trading_agent.h5")
print("Reinforcement Learning agent trained and saved to rl_trading_agent.h5")

