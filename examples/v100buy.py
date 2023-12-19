import os
import time
from datetime import datetime, timedelta
from rich import print
from rich.console import Console
import traceback

from sklearn.model_selection import train_test_split

from binaryapi.constants import CONTRACT_TYPE, DURATION
from binaryapi.stable_api import Binary
from Candles import MarketAnalyzer  # Assuming this is your Market Analyzer class
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
    def generate_signals(data):
        data['macd'] = ta.trend.macd_diff(data['close'], window_fast=12, window_slow=26, window_sign=9)

        # Compute Relative Strength Index (RSI)
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        # Calculate Stochastic Oscillator
        data['stoch'] = ta.momentum.stoch(data['high'], data['low'], data['close'], window=14, smooth_window=3)
        # Compute On-Balance Volume (OBV)
        # data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])

        # Compute Average True Range (ATR)
        # data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])

        # Generate buy and sell signals based on a combination of indicators
        conditions_buy = ((data['rsi'] < 35) & (data['stoch'] < 20))

        conditions_sell = ((data['rsi'] > 65) & (data['stoch'] > 80))

        data['signal'] = np.where(conditions_buy, 1, np.where(conditions_sell, -1, 0))
        return data


    symbols = ['1HZ10V', 'R_10', '1HZ25V', 'R_25', '1HZ50V', 'R_50', '1HZ75V', 'R_75', '1HZ100V', 'R_100']
    while datetime.now() < end_time:

        analyzer = MarketAnalyzer()
        for symbol in symbols:
            fetched_data = analyzer.fetch_tick_history(symbol=symbol, count=50)  # Fetch data for each symbol
            backtest_results = analyzer.backtest_strategy(fetched_data.copy(), generate_signals)
            data = generate_signals(fetched_data)
            print(f"{backtest_results}")

            action = "Hold"
            if not data.empty and len(data) > 0 and not fetched_data.empty and len(fetched_data) > 1:
                # Check for buy/sell signals
                if data['signal'].iloc[-1] == 1:
                    last_open = fetched_data['open'].iloc[-2]
                    last_close = fetched_data['close'].iloc[-2]
                    last_candle_bearish = last_close < last_open


                    print(f"last_candle_bearish: {last_candle_bearish}")
                    print(f"{backtest_results['average_pnl']}")
                    print(f"signal : BUY")
                    if backtest_results['max_consecutive_losses'] == 0 and backtest_results['average_pnl'] > 0:

                            action = "Buy"

                    else:
                        action = "Hold"  # No clear signal, hold position or wait

                elif data['signal'].iloc[-1] == -1:
                    last_open = fetched_data['open'].iloc[-2]
                    last_close = fetched_data['close'].iloc[-2]
                    last_candle_bullish = last_close > last_open
                    print(f"last_candle_bullish: {last_candle_bullish}")
                    print(f"{backtest_results['average_pnl']}")
                    print(f"signal : SELL")
                    if backtest_results['max_consecutive_losses'] == 0 and backtest_results['average_pnl'] > 0:

                        action = "Sell"

                    else:
                        action = "Hold"  # No clear signal, hold position or wait

            if action != "Hold":
                success, contract_id, req_id = binary.buy(
                    contract_type=CONTRACT_TYPE.CALL if action == "Buy" else CONTRACT_TYPE.PUT,
                    amount=amount,
                    symbol=symbol,
                    duration=1,
                    duration_unit=DURATION.MINUTE,
                    subscribe=subscribe
                )
                print({'success': success, 'contract_id': contract_id, 'req_id': req_id})

            time.sleep(5)  # Wait for 20 seconds before the next iteration
