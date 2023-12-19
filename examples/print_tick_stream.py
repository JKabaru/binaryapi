# Prints ticks data for 60 seconds
import os
import time
from datetime import datetime

from rich.console import Console
from binaryapi.stable_api import Binary

# Binary Token
token = os.environ.get('BINARY_TOKEN', 'NGysXImgLMM9Qzv')

console = Console(log_path=False)


def message_handler(message):
    msg_type = message.get('msg_type')

    if msg_type == 'candle':
        # Print tick data from message
        tick_data = message['candle']
        # console.print(tick_data)
        dt = datetime.fromtimestamp(tick_data['epoch'])
        console.print('{}: {} -> {}'.format(dt, tick_data['symbol'], tick_data['quote']))


if __name__ == '__main__':
    binary = Binary(token=token, message_callback=message_handler)
    console.log('Logged in')

    # Symbol: can be an array of symbols too. eg: ['R_75', 'R_100'] etc.
    symbol = ['R_100']

    # Subscribe to ticks stream
    binary.api.ticks(symbol, subscribe=True)

    # Wait for 60 seconds then exit the script
    time.sleep(60)
