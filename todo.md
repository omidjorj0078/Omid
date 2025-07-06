## Todo List

### Phase 1: Research and architecture design
- [x] Research MetaTrader 5 MQL5 language and its capabilities for AI/ML integration.
- [x] Research suitable AI/ML models for HFT in Forex (transformers, DNN, RL, LSTM, hybrid).
- [x] Investigate methods for integrating Python-based AI/ML models with MQL5.
- [x] Research GUI development options within MetaTrader 5.
- [x] Design the overall architecture of the HFT AI expert advisor.
- [x] Prepare a comprehensive research report and architecture design document.



### Phase 2: AI model development and training
- [x] Set up Python environment with necessary libraries (TensorFlow, PyTorch, Hugging Face, pandas, numpy, scikit-learn). (Note: MetaTrader5 Python package installation failed, will use direct MQL5 communication for data acquisition and trade execution.)
- [x] Acquire historical EURUSD market data for training.
- [x] Preprocess market data for AI model input (feature engineering, normalization).
- [x] Develop Transformer-based model for future candle prediction.
- [x] Develop Reinforcement Learning agent for optimal trading decisions and dynamic SL/TP. (Note: Training is computationally intensive and may require more powerful resources for full completion.)
- [x] Train and validate AI models using historical data.
- [x] Implement continuous retraining mechanism. (Note: This will be implemented via MQL5 for data acquisition and Python for retraining, as MetaTrader5 Python package installation failed.)



### Phase 3: MetaTrader 5 expert advisor implementation
- [x] Create a basic MQL5 Expert Advisor structure.
- [x] Implement data transfer from MQL5 to Python (e.g., via named pipes or sockets). (Python connector script created)
- [x] Implement signal reception from Python to MQL5.
- [x] Implement trade execution logic in MQL5 based on Python signals.
- [x] Handle errors and connection issues between MQL5 and Python.



### Phase 4: GUI development and integration
- [x] Design and implement the GUI elements within MQL5.
- [x] Add input fields for manual setting of trade entry size (lot size or percentage of capital).
- [x] Implement display of predicted future candles as a trend line on the chart.
- [x] Add option to choose trading mode: automatic or manual.
- [x] Implement display of alerts and trade signals on the chart in manual mode.
- [x] Integrate GUI elements with MQL5 EA logic.



### Phase 5: Risk management module implementation
- [x] Implement dynamic stop-loss (SL) and take-profit (TP) levels based on AI predictions.
- [x] Integrate SL/TP logic into the MQL5 trade execution.
- [x] Ensure risk management considers market spread and volatility.
- [x] Test and validate the dynamic SL/TP mechanism.



### Phase 6: Refine AI models and backtesting
- [x] Perform backtesting of the EA using historical data. (Note: Full RL training for optimal performance might require significant computational resources.)
- [x] Improve the RL agent's reward function and training process. (Note: Full optimization of the RL agent is computationally intensive and may require more powerful resources for complete training.)
- [ ] Re-run backtesting and evaluate performance metrics (profitability, drawdown, win rate, etc.).
- [ ] Optimize model parameters and trading strategy based on backtesting results.
- [ ] Conduct stress testing and robustness checks.
- [ ] Refine the AI models and MQL5 code based on optimization.

