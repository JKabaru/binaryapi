import os
import time
from datetime import datetime, timedelta
from rich import print
from rich.console import Console
import traceback

from sklearn.model_selection import train_test_split

from binaryapi.constants import CONTRACT_TYPE, DURATION
from binaryapi.stable_api import Binary
from multi_Output import MarketAnalyzer  # Assuming this is your Market Analyzer class
import ta
import numpy as np

token = os.environ.get('BINARY_TOKEN', 'NGysXImgLMM9Qzv')
# NGysXImgLMM9Qzv   1XvdgsTHGAgItnC
console = Console(log_path=False)
prev_balance = 0
loss_count = 0
win_count = -1
amount = 0.35

count_losses = 0
count_win = -1
consecutive_losses = 0

def message_handler(message):
    global prev_balance, loss_count, amount, win_count, count_win, count_losses, consecutive_losses
    msg_type = message.get('msg_type')

    try:
        # if msg_type in ['buy', 'proposal']:
        #     print(message)

        # if msg_type == 'proposal_open_contract':  # When subscribe=true
        #     poc = message['proposal_open_contract']
        #     print(f'{poc["contract_id"]} {poc["status"]}: {poc["longcode"]}')

        if msg_type == 'balance':
            try:
                balance = message["balance"]["balance"]
                currency = binary.api.profile.currency

                console.log('Balance: {} {}'.format(balance, currency))

                # Check if there was a change in balance (win or loss)
                if balance > prev_balance:
                    console.log('You won!')
                    win_count += 1
                    if loss_count > consecutive_losses:
                        consecutive_losses = loss_count
                    loss_count = 0
                    count_win += 1
                    amount = 0.35
                    console.log(f"win count: {count_win}")
                    console.log(f"loss count: {count_losses}")
                    console.log(f"CONSECUTIVE LOSSES: {consecutive_losses}")

                elif balance < prev_balance:


                    console.log('')


                else:
                    loss_count += 1
                    win_count = 0
                    count_losses += 1
                    console.log(f"loss count: {count_losses}")
                    console.log(f"win count: {count_win}")
                    console.log(f"CONSECUTIVE LOSSES: {consecutive_losses}")
                    if loss_count >= 1:  # Adjust amount after 3 consecutive losses
                        amount *= 1  # Double the amount after 3 losses
                        console.log(f"Increasing amount to: {amount}")


                prev_balance = balance
            except KeyError:
                console.print_exception()


    except Exception as e:
        traceback.print_exc()

    # print("MSG", message)


if __name__ == '__main__':
    binary = Binary(token=token, message_callback=message_handler)
    print('Logged in')

    subscribe = False
    end_time = datetime.now() + timedelta(minutes=700)  # Set end time for 5 minutes from now

    # Before the loop for trading
    analyzer = MarketAnalyzer()
    fetched_data = analyzer.fetch_tick_history(count=500)
    analyzer.tick_df = fetched_data
    x_cls_train, y_cls_train, x_reg_train, y_reg_train = analyzer.prepare_data()

    # Split data into training and test sets for both classification and regression
    x_cls_train, x_cls_test, y_cls_train, y_cls_test = train_test_split(x_cls_train, y_cls_train, test_size=0.2,
                                                                        random_state=42)
    x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x_reg_train, y_reg_train, test_size=0.2,
                                                                        random_state=42)
    default_classifier, default_regressor = analyzer.train_multi_output_model(x_cls_train, y_cls_train, x_reg_train,
                                                                              y_reg_train)
    best_classifier = default_classifier
    best_regressor = default_regressor

    while datetime.now() < end_time:


        # Your MarketAnalyzer code to get the majority prediction
        analyzer = MarketAnalyzer()
        fetched_data = analyzer.fetch_tick_history(count=500)
        analyzer.tick_df = fetched_data
        x_cls_train, y_cls_train, x_reg_train, y_reg_train = analyzer.prepare_data()

        if len(x_cls_train) < 5:  # Adjust threshold based on your dataset size
            test_size = 0.1  # Set a smaller test size if the dataset is too small
        else:
            test_size = 0.2  # Use the original test size

        # Perform train-test split only if the dataset size is sufficient
        if len(x_cls_train) >= 5:
            x_cls_train, x_cls_test, y_cls_train, y_cls_test = train_test_split(
                x_cls_train, y_cls_train, test_size=test_size, random_state=42)
            x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(
                x_reg_train, y_reg_train, test_size=test_size, random_state=42)

        best_classifier, best_regressor = analyzer.optimize_hyperparameters_random_feedback(
            x_cls_train, y_cls_train, x_reg_train, y_reg_train, win_count, loss_count)



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

        action = analyzer.predict_and_trade(best_classifier, best_regressor, features_for_prediction)

        print(f"Predicted Action: {action}")

        # Calculate the start of the next minute
        current_time = datetime.now()
        seconds_to_next_minute = (50 - current_time.second) % 60  # Calculate remaining seconds until the next minute

        print(f"Time left: {seconds_to_next_minute}")

        time.sleep(seconds_to_next_minute)


        if action != "Hold":
            success, contract_id, req_id = binary.buy(
                contract_type=CONTRACT_TYPE.CALL if action == "Sell" else CONTRACT_TYPE.PUT,
                amount=amount,
                symbol='R_100',
                duration=1,
                duration_unit=DURATION.MINUTE,
                subscribe=subscribe
            )
            print({'success': success, 'contract_id': contract_id, 'req_id': req_id})

        time.sleep(1)  # Wait for 20 seconds before the next iteration
