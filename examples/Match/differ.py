import os
import pandas as pd
import numpy as np
from rich.console import Console
from binaryapi.stable_api import Binary
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import ta

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import ast  # Import this to handle the string representation of the dictionary

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MarketAnalyzer:
    def __init__(self):
        self.binary_token = os.environ.get('BINARY_TOKEN', 'NGysXImgLMM9Qzv')
        self.console = Console(log_path=False)
        self.binary = Binary(token=self.binary_token)
        self.tick_df = None
        self.best_rf = None

    def fetch_tick_history(self, symbol='R_100', style='candles', count=100):
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
        self.console.log('Logged in')

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

    def prepare_data(self):

        self.tick_df['price_change'] = self.tick_df['close'].shift(-1) - self.tick_df['close']

        # Define thresholds for magnitude labeling
        thresholds = {
            'strong_increase': 0.02,  # Adjust thresholds as needed
            'slight_increase': 0.005,
            'no_change': -0.005,
            'slight_decrease': -0.02,
            'strong_decrease': float('-inf')
        }

        # Function to label magnitude
        def label_magnitude(price_change):
            if price_change > thresholds['strong_increase']:
                return 'Strong Increase'
            elif thresholds['slight_increase'] < price_change <= thresholds['strong_increase']:
                return 'Slight Increase'
            elif thresholds['no_change'] < price_change <= thresholds['slight_increase']:
                return 'No Change'
            elif thresholds['slight_decrease'] < price_change <= thresholds['no_change']:
                return 'Slight Decrease'
            else:
                return 'Strong Decrease'

        # Apply labeling function to create magnitude labels
        self.tick_df['magnitude_label'] = self.tick_df['price_change'].apply(label_magnitude)

        # Encode labels as numerical values
        label_mapping = {
            'Strong Increase': 0,
            'Slight Increase': 1,
            'No Change': 2,
            'Slight Decrease': 3,
            'Strong Decrease': 4
        }

        self.tick_df['magnitude_label_encoded'] = self.tick_df['magnitude_label'].map(label_mapping)
        # Feature Engineering using 'ta' library
        # Example indicators: SMA, RSI, MACD, Bollinger Bands
        # Feel free to add more indicators or adjust parameters
        self.tick_df['sma'] = ta.trend.sma_indicator(close=self.tick_df['close'], window=5)
        self.tick_df['rsi'] = ta.momentum.rsi(self.tick_df['close'], window=14)
        self.tick_df['macd'] = ta.trend.macd(self.tick_df['close'])
        self.tick_df['bb_bbm'] = ta.volatility.bollinger_mavg(self.tick_df['close'])

        # Drop rows with NaN values (if any)
        self.tick_df.dropna(inplace=True)

        # Selecting features for classification and regression
        selected_features = ['sma', 'rsi', 'macd', 'bb_bbm']

        x_classification = self.tick_df[selected_features]

        x_regression = self.tick_df[selected_features]


        # Perform imputation for missing values



        x_classification = np.nan_to_num(x_classification, nan=np.nanmean(x_classification))
        x_regression = np.nan_to_num(x_regression, nan=np.nanmean(x_regression))


        y_classification = self.tick_df['magnitude_label_encoded'].values.ravel()
        y_regression = self.tick_df['price_change'].shift(-1).ffill().values.ravel()

        return x_classification, y_classification, x_regression, y_regression

        # Function to train Multi-Output Random Forest

    def train_multi_output_model(self, x_cls, y_cls, x_reg, y_reg):
        # Multi-Class Classification (Random Forest Classifier)
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(x_cls, y_cls)

        # Regression (Random Forest Regressor)
        regressor = RandomForestRegressor(n_estimators=100)
        regressor.fit(x_reg, y_reg)

        return classifier, regressor

    def evaluate_models(self, classifier, regressor, x_cls_test, y_cls_test, x_reg_test, y_reg_test):
        # Classification Evaluation
        y_cls_pred = classifier.predict(x_cls_test)
        accuracy = accuracy_score(y_cls_test, y_cls_pred)
        precision = precision_score(y_cls_test, y_cls_pred, average='weighted', zero_division=0)
        recall = recall_score(y_cls_test, y_cls_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_cls_test, y_cls_pred, average='weighted', zero_division=0)

        print("Classification Metrics:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")

        # Regression Evaluation
        y_reg_pred = regressor.predict(x_reg_test)
        mse = mean_squared_error(y_reg_test, y_reg_pred)
        rmse = mean_squared_error(y_reg_test, y_reg_pred, squared=False)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)

        print("\nRegression Metrics:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        return {
            "classification_accuracy": accuracy,
            "classification_precision": precision,
            "classification_recall": recall,
            "classification_f1": f1,
            "regression_mse": mse,
            "regression_rmse": rmse,
            "regression_mae": mae
        }

    def predict_and_trade(self, classifier, regressor, features):
        # Predict movement magnitude
        magnitude_prediction = classifier.predict(features.reshape(1, -1))[0]

        # Predict trend direction
        trend_prediction = self.predict_trend(classifier, features)

        # Predict price change
        price_change_prediction = regressor.predict(features.reshape(1, -1))[0]

        # Trading Strategy based on predictions
        if magnitude_prediction == 0 and trend_prediction == 'Upwards' and price_change_prediction > 0.5:
            # Example: Potential buy signal for strong increase, upwards trend, and high positive price change
            return "Buy"
        # You can define more conditions based on your predictions for different trading signals
        if magnitude_prediction != 0 and trend_prediction == 'Downwards' and price_change_prediction < -0.5:
            # Example: Potential sell signal for significant decrease, downwards trend, and high negative price change
            return "Sell"

        return "Hold"  # Default action if no specific conditions are met

        # Helper function to predict trend direction

    def predict_trend(self, classifier, features):
        # Use a subset of features relevant to trend prediction
        trend_features = features  # Adjust features based on trend prediction needs
        trend_prediction = classifier.predict(trend_features.reshape(1, -1))[0]
        return trend_prediction
    #
    #
    #
    # def train_model(self):
    # def validate_unseen_data(self):
    #
    # def retrain_model(self):


if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    fetched_data = analyzer.fetch_tick_history(count=200)
    analyzer.tick_df = fetched_data
    x_cls_train, y_cls_train, x_reg_train, y_reg_train = analyzer.prepare_data()

    # Split data into training and test sets for both classification and regression
    x_cls_train, x_cls_test, y_cls_train, y_cls_test = train_test_split(x_cls_train, y_cls_train, test_size=0.2,
                                                                        random_state=42)
    x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x_reg_train, y_reg_train, test_size=0.2,
                                                                        random_state=42)

    # Train the multi-output model
    classifier, regressor = analyzer.train_multi_output_model(x_cls_train, y_cls_train, x_reg_train, y_reg_train)

    # Evaluate models on test data
    evaluation_results = analyzer.evaluate_models(classifier, regressor, x_cls_test, y_cls_test, x_reg_test, y_reg_test)

    # This snippet calculates the technical indicators for the next candle
    last_5_close = fetched_data['close'].iloc[-40:]  # Selecting the last 5 close prices
    next_candle_close = fetched_data['close'].iloc[-1]  # Close price of the next candle

    # Calculate SMA (Simple Moving Average) for the next candle
    sma_value = ta.trend.sma_indicator(close=last_5_close, window=5).iloc[-1]

    # Calculate RSI (Relative Strength Index) for the next candle
    rsi_value = ta.momentum.rsi(last_5_close, window=14).iloc[-1]

    # Calculate MACD (Moving Average Convergence Divergence) for the next candle
    macd_value = ta.trend.macd(last_5_close).iloc[-1]

    # Calculate Bollinger Bands for the next candle
    bb_bbm_value = ta.volatility.bollinger_mavg(last_5_close).iloc[-1]

    # Form the feature vector for prediction
    features_for_prediction = np.array([sma_value, rsi_value, macd_value, bb_bbm_value])
    # Check for NaN values in features_for_prediction
    nan_values = np.isnan(features_for_prediction)



    # Reshape the 1D array to a 2D array with a single column
    features_for_prediction = features_for_prediction.reshape(1, -1)


    features_for_prediction = np.nan_to_num(features_for_prediction, nan=np.nanmean(features_for_prediction))


    # Drop rows with NaN values
    # features_for_prediction = features_for_prediction[~np.isnan(features_for_prediction).any(axis=1)]
    print(f"Previous close: {next_candle_close}")
    print(f"Predict Features: {features_for_prediction}")

    action = analyzer.predict_and_trade(classifier, regressor, features_for_prediction)



    print(f"Predicted Action: {action}")

