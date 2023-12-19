import os

import time
from binaryapi.stable_api import Binary
from rich.console import Console
import ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import numpy as numpy
from collections import defaultdict
from datetime import timedelta

class MarketAnalyzer:
    def __init__(self):
        self.binary_token = os.environ.get('BINARY_TOKEN', 'NGysXImgLMM9Qzv')
        self.console = Console(log_path=False)
        self.binary = None  # Initialize Binary as None initially
        self.tick_df = None
        self.best_rf = None
        self.logged_in = False  # Flag to track login status

    def login(self):
        if not self.logged_in:  # Check if not already logged in
            self.binary = Binary(token=self.binary_token)
            self.logged_in = True
            self.console.log('Logged in')

    def fetch_tick_history(self, symbol='R_100', style='candles', count=0):
        # self.login()  # Ensure login before fetching data

        close = []
        epoch = []
        high = []
        low = []
        open_ = []  # Renamed 'open' to 'open_' to avoid conflict with reserved keyword

        def message_handler(message):
            nonlocal close, epoch, high, low, open_
            msg_type = message.get('msg_type')

            if msg_type in ['candles', 'ohlc']:
                # Accumulate candle data
                candles_data = message['candles']
                for candle in candles_data:
                    close.append(candle['close'])
                    epoch.append(candle['epoch'])
                    high.append(candle['high'])
                    low.append(candle['low'])
                    open_.append(candle['open'])

        self.binary = Binary(token=self.binary_token, message_callback=message_handler)

        self.binary.api.ticks_history(
            ticks_history=symbol,
            style=style,
            count=count,
            end='latest'
        )
        time.sleep(1)

        tick_data = {'close': close, 'epoch': epoch, 'high': high, 'low': low, 'open': open_}
        self.tick_df = pd.DataFrame(tick_data)

        return self.tick_df




    def establish_relationship(self, data1, data2):
        if data1 is None or data1.empty or data2 is None or data2.empty:
            return "Insufficient data for analysis"

        # Extract the close prices from data1 and data2
        close_prices_data1 = data1['close']
        close_prices_data2 = data2['close']

        # Calculate cross-correlation between the two datasets
        cross_corr = self.cross_correlation(close_prices_data1, close_prices_data2)

        # Analyze the cross-correlation results
        correlation_threshold = 0.5  # Placeholder threshold for significant correlation
        if any(abs(corr) > correlation_threshold for corr in cross_corr):
            relationship = "The datasets show significant correlation at certain lags"
        else:
            relationship = "The datasets do not exhibit significant correlation"

        return relationship

    def cross_correlation(self, series1, series2):
        # Calculate cross-correlation between two series at different lags
        max_lag = min(len(series1), len(series2)) // 4  # Adjust the max lag based on your data characteristics
        cross_corr = [series1.corr(series2.shift(lag)) for lag in range(-max_lag, max_lag + 1)]
        return cross_corr

    def calculate_correlation(self, data1, data2):
        if data1 is None or data1.empty or data2 is None or data2.empty:
            return "Insufficient data for analysis"

        # Extract the close prices from data1 and data2
        close_prices_data1 = data1['close']
        close_prices_data2 = data2['close']

        # Calculate Pearson correlation coefficient
        correlation = close_prices_data1.corr(close_prices_data2)

        return correlation



    def develop_strategy(self, data1, data2, lag=5, delay=15):
        if data1 is None or data1.empty or data2 is None or data2.empty:
            return "Insufficient data for strategy development"

        # Extract the close prices from data1 and data2
        close_prices_data1 = data1['close']
        close_prices_data2 = data2['close']

        # Create lagged features from data2
        lagged_data2 = pd.DataFrame()
        for i in range(1, lag + 1):
            lagged_data2[f'lag_{i}'] = close_prices_data2.shift(i)

        # Combine lagged features with data1 for prediction
        combined_data = pd.concat([close_prices_data1, lagged_data2], axis=1).dropna()

        # Split the data into features and target
        X = combined_data.drop('close', axis=1)
        y = combined_data['close']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict data1 movements after a delay
        predicted_data1 = model.predict(X_test)
        mse = mean_squared_error(y_test, predicted_data1)

        return mse


    def backtesting(self, data1, data2, strategy, lag=5, delay=15):
        if data1 is None or data1.empty or data2 is None or data2.empty:
            return "Insufficient data for backtesting"

        # Extract the close prices from data1 and data2
        close_prices_data1 = data1['close']
        close_prices_data2 = data2['close']

        # Create lagged features from data2
        lagged_data2 = pd.DataFrame()
        for i in range(1, lag + 1):
            lagged_data2[f'lag_{i}'] = close_prices_data2.shift(i)

        # Combine lagged features with data1 for prediction
        combined_data = pd.concat([close_prices_data1, lagged_data2], axis=1).dropna()

        # Split the data into features and target
        X = combined_data.drop('close', axis=1)
        y = combined_data['close']

        # Train a Random Forest Regressor on historical data
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Prepare data for testing
        test_data = pd.concat([close_prices_data1, lagged_data2], axis=1).dropna().tail(len(X))

        # Predict data1 movements after a delay
        X_test = test_data.drop('close', axis=1)
        y_test = test_data['close']
        predicted_data1 = model.predict(X_test)
        mse = mean_squared_error(y_test, predicted_data1)

        return mse



    def update_strategy(self, strategy, new_data):
        if strategy is None or new_data is None or len(new_data) == 0:
            return strategy  # Return the original strategy if no new data is available

        # Extract the new close prices from the new_data
        new_close_prices_data = pd.DataFrame(new_data)['close']

        series_from_df = new_close_prices_data

        strategy = {
            'lag_1': [1, 2, 3, 4, 5],
            'lag_2': [2, 3, 4, 5, 6],
            'lag_3': [3, 4, 5, 6, 7],
            'lag_4': [4, 5, 6, 7, 8],
            'lag_5': [5, 6, 7, 8, 9],
            'target_variable': [0.01, -0.02, 0.03, -0.01, 0.02]  # Example target variable for prediction
        }
        # Update the strategy with new real-time data as additional lagged features
        lag = 5  # Assuming the strategy used a lag of 5 for features
        updated_strategy1 = strategy.copy()  # Create a copy of the original strategy

        # print(f"{new_close_prices_data}")
        # print(f"{series_from_df}")
        # Append new lagged features based on the new real-time data

        for i in range(1, lag + 1):
            lag_key = f'lag_{i}'



            # Ensure the index to access is within the valid range

            if i <= len(new_close_prices_data):
                lagged_value = series_from_df.iloc[-i]
                if isinstance(lagged_value, (int, float)):
                    lagged_value_str = str(lagged_value).replace('.', '')
                    lagged_digits = [int(digit) for digit in lagged_value_str]

                    # Check if the lag_key exists and initialize it as a list if not



                    # Append the individual digits to the updated_strategy[lag_key] list
                    updated_strategy1[lag_key].extend(lagged_digits)

        return updated_strategy1

    def risk_management(self, data1, data2, strategy, backtest_error_threshold=0.1):
        if strategy is None:
            return "No strategy available for risk management"

        # Placeholder for risk management based on backtesting results
        # Analyze the backtesting results to assess strategy performance
        if backtest_error_threshold is not None:
            if strategy > backtest_error_threshold:
                risk_assessment = "High backtesting error, consider re-evaluating the strategy"
            else:
                risk_assessment = "Acceptable backtesting error, continue monitoring the strategy"

        return risk_assessment


    def monitor_and_adjust(self, updated_strategy, data1):
        if updated_strategy is None or data1 is None or data1.empty:
            return False  # No strategy or insufficient data to monitor

        # Placeholder for more sophisticated monitoring and adjustment strategies
        # Example: Evaluate strategy performance using statistical analysis or models

        # Calculate mean and standard deviation of actual data1 movements
        actual_movements = data1['close'].pct_change().dropna()
        mean_actual_movements = actual_movements.mean()
        std_actual_movements = actual_movements.std()

        # Calculate mean and standard deviation of predicted data1 movements using the strategy
        # For the sake of example, assume predicted movements are generated from the strategy
        predicted_movements = self.generate_predicted_movements(updated_strategy)
        mean_predicted_movements = predicted_movements.mean()
        std_predicted_movements = predicted_movements.std()

        # Compare means and standard deviations to decide if adjustments are needed
        adjustment_needed = False

        if abs(mean_actual_movements - mean_predicted_movements) > 0.01 * mean_actual_movements:
            adjustment_needed = True

        if abs(std_actual_movements - std_predicted_movements) > 0.01 * std_actual_movements:
            adjustment_needed = True

        return adjustment_needed

    def continuous_monitoring(self, data1, data2, strategy):
            if data1 is None or data1.empty or data2 is None or data2.empty:
                return "Insufficient data for continuous monitoring"

            if strategy is None:
                return "No strategy available for continuous monitoring"

            # Placeholder for continuous monitoring and adjustments based on real-time data
            # Here, simulate real-time data updates and strategy adjustments
            simulated_real_time_data_update = self.fetch_tick_history(symbol='R_100', count=1)  # Simulated real-time data update
            simulated_data1_update = simulated_real_time_data_update['close']


            # Assuming the strategy was based on lagged features, update the features in the strategy
            updated_strategy = self.update_strategy(strategy, simulated_data1_update)

            # Monitor strategy performance with updated features and make adjustments if necessary
            if updated_strategy:
                adjustment_needed = self.monitor_and_adjust(updated_strategy, data1)
                if adjustment_needed:
                    return "Adjustment made in the strategy based on real-time data"

            return "No adjustments made in the strategy"

    def generate_predicted_movements(self, updated_strategy):
        if updated_strategy is None:
            return None  # No strategy available to generate predictions

        # Extract lagged features from the updated strategy
        lagged_features = [updated_strategy[f'lag_{i}'] for i in range(1, 6)]  # Assuming 5 lagged features

        # Find the maximum length of sequences in lagged_features
        max_length = max(len(seq) for seq in lagged_features)

        # Pad sequences with zeros to make them uniform
        padded_features = [seq + [0] * (max_length - len(seq)) for seq in lagged_features]
        # Prepare input features and target variable for the model
        X = np.array(padded_features)  # Transpose for sklearn format (samples x features)
        y = np.array(updated_strategy['target_variable'])  # Assuming 'target_variable' is the predicted variable

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict movements using the trained model
        predicted_movements = model.predict(X)  # Replace X with new data for real-time predictions

        return predicted_movements


    def analyze_relationship(self, data1, data2, delay=15):
        if data1 is None or data1.empty or data2 is None or data2.empty:
            return "Insufficient data for analysis"

        # Step 1: Establish Relationship
        relationship = self.establish_relationship(data1, data2)

        # Step 2: Calculate Correlation
        correlation = self.calculate_correlation(data1, data2)





        # Optionally return or use analysis results for decision-making
        return relationship, correlation

    def backtest_strategy(self, data, strategy_function):
        signals = strategy_function(data)  # Generate signals using the provided strategy function

        position = 0  # Start with no position
        pnl = []  # List to track profit/loss at each trade
        entry_time = None  # Initialize entry time
        consecutive_losses = 0  # Track consecutive losses
        max_consecutive_losses = 0  # Track maximum consecutive losses

        for i in range(len(data)):
            current_time = data.index[i]  # Get the current timestamp

            if signals['signal'].iloc[i] == 1:
                if position == 0:  # Check if no position is open
                    position = 1  # Enter long position
                    entry_price = data['close'].iloc[i]
                    entry_time = current_time  # Record entry time

            elif signals['signal'].iloc[i] == -1:
                if position == 0:  # Check if no position is open
                    position = -1  # Enter short position
                    entry_price = data['close'].iloc[i]
                    entry_time = current_time  # Record entry time

            # Add time-based conditions for exit (e.g., if holding for a certain duration)
            if position != 0 and entry_time is not None:
                holding_duration = current_time - entry_time  # Calculate holding duration
                holding_duration_timedelta = timedelta(minutes=holding_duration)

                if holding_duration_timedelta >= timedelta(minutes=1):  # Example: Exit after 1 minute
                    exit_price = data['close'].iloc[i]
                    trade_result = exit_price - entry_price
                    pnl.append(trade_result)  # Calculate profit/loss

                    # Update consecutive losses count
                    if trade_result < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0

                    position = 0  # Reset position
                    entry_time = None  # Reset entry time

        total_trades = len(pnl)
        win_trades = sum([1 for trade in pnl if trade > 0])
        loss_trades = total_trades - win_trades
        win_ratio = win_trades / total_trades if total_trades > 0 else 0
        average_pnl = sum(pnl) / total_trades if total_trades > 0 else 0

        return {
            'total_trades': total_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_ratio': win_ratio,
            'average_pnl': average_pnl,
            'max_consecutive_losses': max_consecutive_losses
        }


if __name__ == "__main__":
    analyzer = MarketAnalyzer()

    fetched_data1 = analyzer.fetch_tick_history(symbol='R_100', count=100)  # Fetching more data for analysis
    fetched_data2 = analyzer.fetch_tick_history(symbol='1HZ100V', count=100)  # Fetching data for the second symbol



    # Assume initial strategy with lagged features
    initial_strategy = {
        'lag_1': [1, 2, 3, 4, 5],
        'lag_2': [2, 3, 4, 5, 6],
        'lag_3': [3, 4, 5, 6, 7],
        'lag_4': [4, 5, 6, 7, 8],
        'lag_5': [5, 6, 7, 8, 9],
        'target_variable': [0.01, -0.02, 0.03, -0.01, 0.02]  # Example target variable for prediction
    }

    # Simulated real-time data update
    simulated_real_time_data_update = analyzer.fetch_tick_history(symbol='R_100', count=1)
    simulated_data1_update = simulated_real_time_data_update['close']

    # Update strategy with new real-time data
    updated_strategy = analyzer.update_strategy(initial_strategy, simulated_real_time_data_update)

    # Perform analysis using fetched data and updated strategy
    fetched_data1 = analyzer.fetch_tick_history(symbol='R_100', count=100)
    fetched_data2 = analyzer.fetch_tick_history(symbol='1HZ100V', count=100)

    # # Perform analysis using fetched data and updated strategy
    # relationship, correlation, strategy, backtest_results, risk_analysis, continuous_monitoring_result = analyzer.analyze_relationship(fetched_data1, fetched_data2)
    # predicted_movements = analyzer.generate_predicted_movements(
    #     updated_strategy)  # Use updated_strategy or the relevant strategy
    # # Optionally use or print the analysis results for decision-making
    # print(f"Relationship: {relationship}")
    # print(f"Correlation: {correlation}")
    # print(f"Backtest Results: {backtest_results}")
    # print(f"Risk Analysis: {risk_analysis}")
    # print(f"Continuous Monitoring Result: {continuous_monitoring_result}")








