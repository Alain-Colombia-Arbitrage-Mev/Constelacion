import sys
sys.path.insert(0, './lib')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import xgboost as xgbs
from xgboost import plot_importance, plot_tree
import yfinance as yf
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import warnings
import json
import pytz
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import mplfinance as mpf
import argparse

warnings.filterwarnings("ignore")

colombia_tz = pytz.timezone('America/Bogota')

parser = argparse.ArgumentParser(description='Predict forex prices using XGBoost.')
parser.add_argument('symbol', type=str, help='The symbol to predict (e.g., EURUSD=X)')
args = parser.parse_args()

SYMBOL = args.symbol.upper()
INTERVAL = '1h'

def get_historical_data(symbol, interval='1h', days=180):
    end_date = pd.Timestamp.now(tz=pytz.UTC)
    start_date = end_date - pd.Timedelta(days=days)
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if data.empty:
            print(f"Warning: No data available for {symbol}")
            return None
        data.index = pd.to_datetime(data.index, utc=True)
        return data
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None

def feature_engineering(data, DXY, predictions=np.array([None]))->pd.core.frame.DataFrame:
    assert type(data) == pd.core.frame.DataFrame, "data must be a dataframe"
    assert type(DXY) == pd.core.series.Series, "DXY must be a series"
    assert type(predictions) == np.ndarray, "predictions must be an array"

    if predictions.any() == True:
        data = get_historical_data(SYMBOL, interval=INTERVAL)
        DXY = get_historical_data("DX-Y.NYB", interval=INTERVAL)["Close"]
        if data is None or DXY is None:
            raise ValueError("Unable to fetch required data")
        data = features(data, DXY)
        data["Predictions"] = predictions
        data["Close"] = data["Close_y"]
        data.drop("Close_y", 1, inplace=True)
        data.dropna(0, inplace=True)
    data = features(data, DXY)
    return data

def features(data, DXY)->pd.core.frame.DataFrame:
    for i in [2, 3, 4]:
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).mean()
        data[f"Volume{i}"] = data["Volume"].rolling(i).mean()
        data[f"Low_std{i}"] = data["Low"].rolling(i).std()
        data[f"High_std{i}"] = data["High"].rolling(i).std()
        data[f"Close{i}"] = data["Close"].shift(i)

    data["DXY"] = DXY
    data["Hour"] = data.index.hour
    data["Day"] = data.index.day
    data["Month"] = data.index.month
    data["Weekday"] = data.index.weekday
    data["Upper_Shape"] = data["High"] - np.maximum(data["Open"], data["Close"])
    data["Lower_Shape"] = np.minimum(data["Open"], data["Close"]) - data["Low"]
    data["Close_y"] = data["Close"]
    
    # Remove any columns with NaN values
    data = data.dropna(axis=1)
    
    print(f"Number of features after engineering: {data.shape[1]}")
    return data

def windowing(data, WINDOW, PREDICTION_SCOPE):
    assert isinstance(data, np.ndarray), "data must be passed as an array"
    assert isinstance(WINDOW, int), "Window must be an integer"
    assert isinstance(PREDICTION_SCOPE, int), "Prediction scope must be an integer"

    X, y = [], []

    for i in range(len(data) - (WINDOW + PREDICTION_SCOPE)):
        X.append(data[i:i+WINDOW, :-1])
        y.append(data[i+WINDOW+PREDICTION_SCOPE-1, -1])

    return np.array(X), np.array(y)

def train_test_split(data, WINDOW):
    assert type(data) == pd.core.frame.DataFrame, "data must be a dataframe"
    assert type(WINDOW) == int, "Window must be an integer"

    train = data.iloc[:-WINDOW]
    test = data.iloc[-WINDOW:]

    return train, test

def train_validation_split(train, percentage):
    assert type(train) == pd.core.frame.DataFrame, "train must be a dataframe"
    assert type(percentage) == float, "percentage must be a float"

    train_set = np.array(train.iloc[:int(len(train)*percentage)])
    validation_set = np.array(train.iloc[int(len(train)*percentage):])

    return train_set, validation_set

def plotting(y_val, y_test, pred_test, mae, WINDOW, PREDICTION_SCOPE):
    assert type(WINDOW) == int, "Window must be an integer"
    assert type(PREDICTION_SCOPE) == int, "Prediction scope must be an integer"

    ploting_pred = [y_test[-1], pred_test]
    ploting_test = [y_val[-1]]+list(y_test)

    time = (len(y_val)-1)+(len(ploting_test)-1)+(len(ploting_pred)-1)

    x_ticks = list(forex_prices.index[-time:])+[forex_prices.index[-1]+timedelta(hours=PREDICTION_SCOPE+1)]

    _predictprice = round(pred_test, 4)
    _date = x_ticks[-1]
    _hours = PREDICTION_SCOPE+1

    return _predictprice, _date, _hours

def train_xgb_model(X_train, y_train, X_val, y_val, plotting=False):
    print(f"Shape of X_train in train_xgb_model: {X_train.shape}")
    print(f"Shape of X_val in train_xgb_model: {X_val.shape}")

    # Aplanar los datos de entrada
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    print(f"Shape of X_train_flat: {X_train_flat.shape}")
    print(f"Shape of X_val_flat: {X_val_flat.shape}")

    model = xgbs.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train_flat, y_train)

    pred_val = model.predict(X_val_flat)
    mae = mean_absolute_error(y_val, pred_val)

    if plotting:
        plt.figure(figsize=(15, 6))
        sns.set_theme(style="white")
        sns.lineplot(x=range(len(y_val)), y=y_val, color="grey", alpha=.4)
        sns.lineplot(x=range(len(y_val)), y=pred_val, color="red")
        plt.xlabel("Time")
        plt.ylabel(f"{SYMBOL} price")
        plt.title(f"The MAE for this period is: {round(mae, 6)}")

    return mae, model

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

def get_highest_volume_prices(historical_prices, n=4):
    if historical_prices.empty:
        return []
    
    highest_volume_hours = historical_prices.sort_values('Volume', ascending=False).head(n)
    
    highest_volume_prices = [
        {
            'date': date.strftime('%Y-%m-%d %H:%M'),
            'price': price,
            'volume': volume
        }
        for date, price, volume in zip(highest_volume_hours.index, highest_volume_hours['Close'], highest_volume_hours['Volume'])
    ]
    
    return highest_volume_prices

def determine_trade_direction(current_price, predicted_price):
    return "LONG 📈" if predicted_price > current_price else "SHORT 📉"

def find_entry_and_targets(historical_prices, current_price, predicted_price, trade_direction):
    if historical_prices.empty:
        print("Warning: Empty historical data. Using current and predicted prices.")
        highest_price = max(current_price, predicted_price)
        lowest_price = min(current_price, predicted_price)
    else:
        highest_price = historical_prices['High'].max()
        lowest_price = historical_prices['Low'].min()
    
    if "SHORT" in trade_direction:
        entry = highest_price
        tp1 = min(predicted_price, current_price)
        tp2 = lowest_price
        stop_loss = entry * 1.001  # Adjusted for forex
    else:  # LONG
        entry = lowest_price
        tp1 = max(predicted_price, current_price)
        tp2 = highest_price
        stop_loss = entry * 0.999  # Adjusted for forex
    
    return entry, tp1, tp2, stop_loss

def create_chart(symbol, hours=168, entry=None, tp1=None, tp2=None, stop_loss=None, trade_direction=None, interval='1h'):
    print(f"Starting chart creation for {symbol} with {interval} interval")
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(hours=hours)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    
    print(f"Downloaded data for chart: {len(data)} rows")
    print(data.head())
    
    if len(data) < 2:
        print(f"Not enough data to create chart for {symbol}")
        return None
    
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
    
    annotations = []
    if all(v is not None for v in [entry, tp1, tp2, stop_loss, trade_direction]):
        color = 'g' if "LONG" in trade_direction else 'r'
        entry_line = [float(entry)] * len(data)
        tp1_line = [float(tp1)] * len(data)
        tp2_line = [float(tp2)] * len(data)
        stop_loss_line = [float(stop_loss)] * len(data)
        annotations.extend([
            mpf.make_addplot(entry_line, color=color, linestyle='--', label=f'Entry: {entry:.4f}'),
            mpf.make_addplot(tp1_line, color=color, linestyle=':', label=f'TP1: {tp1:.4f}'),
            mpf.make_addplot(tp2_line, color=color, linestyle=':', label=f'TP2: {tp2:.4f}'),
            mpf.make_addplot(stop_loss_line, color='purple', linestyle='-.', label=f'SL: {stop_loss:.4f}')
        ])
    
    if not data['SMA20'].isnull().all():
        annotations.append(mpf.make_addplot(data['SMA20'].astype(float), color='orange', label='SMA20'))
    
    try:
        print("Creating the chart...")
        fig, axes = mpf.plot(data, type='candle', style=s, volume=True, 
                             addplot=annotations if annotations else None, 
                             title=f'\n{symbol} Price Chart (Last {hours} Hours, {interval} interval)',
                             ylabel='Price',
                             ylabel_lower='Volume',
                             figsize=(12, 8),
                             returnfig=True)
        
        axes[0].legend(loc='upper left')
        
        chart_path = os.path.abspath(f'{symbol}_chart.png')
        print(f"Attempting to save chart at: {chart_path}")
        plt.savefig(chart_path)
        plt.close(fig)
        print(f"Chart created and saved successfully at {chart_path}")
        return chart_path
    except Exception as e:
        print(f"Error creating chart: {e}")
        import traceback
        print(traceback.format_exc())
        return None

async def send_to_telegram_async(message, image_path):
    bot_token = '6848512889:AAG2fBYJ-dcblpngnvRB4Pexw19d_E_kkR0'
    chat_id = '-1002225888276'
    
    bot = Bot(token=bot_token)
    
    try:
        print(f"Attempting to send message to Telegram. Chat ID: {chat_id}")
        message_result = await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        print(f"Message sent. Message ID: {message_result.message_id}")
        
        if image_path and os.path.exists(image_path):
            print(f"Attempting to send image from {image_path}...")
            with open(image_path, 'rb') as image_file:
                photo_result = await bot.send_photo(chat_id=chat_id, photo=image_file)
            print(f"Image sent. Photo ID: {photo_result.message_id}")
        elif image_path:
            print(f"Could not find image at {image_path}")
        else:
            print("No image path provided.")
        
        print("Telegram sending process completed.")
    except Exception as e:
        print(f"Error sending message to Telegram: {e}")
        import traceback
        print(traceback.format_exc())

def send_to_telegram(message, image_path):
    print(f"Initiating Telegram send. Image path: {image_path}")
    asyncio.run(send_to_telegram_async(message, image_path))

def predictPrice(interval='1h'):
    global forex_prices, DXY, y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE

    PERCENTAGE = 0.8  # Cambiado de 0.995 a 0.8 para tener suficientes datos de validación
    WINDOW = 24 * 3  # 3 days of hourly data
    PREDICTION_SCOPE = 1  # Predicting 1 hour ahead

    forex_prices = get_historical_data(SYMBOL, interval=INTERVAL, days=90)
    DXY = get_historical_data("DX-Y.NYB", interval=interval, days=90)
    
    if forex_prices is None or DXY is None:
        raise ValueError("Unable to fetch required data")
    
    DXY = DXY["Close"]

    forex_prices = feature_engineering(forex_prices, DXY)

    # Convertir el DataFrame a numpy array
    data_array = forex_prices.values

    # Dividir los datos en entrenamiento y prueba
    train_size = int(len(data_array) * PERCENTAGE)
    train_data = data_array[:train_size]
    test_data = data_array[train_size:]

    # Aplicar windowing a los datos de entrenamiento
    X_train, y_train = windowing(train_data, WINDOW, PREDICTION_SCOPE)

    # Dividir los datos de entrenamiento en entrenamiento y validación
    val_size = int(len(X_train) * 0.2)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")

    mae, xgb_model = train_xgb_model(X_train, y_train, X_val, y_val, plotting=False)

    # Preparar datos de prueba
    X_test, y_test = windowing(test_data, WINDOW, PREDICTION_SCOPE)
    print(f"Shape of X_test: {X_test.shape}")

    pred_test_xgb = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
    print(f"Shape of pred_test_xgb: {pred_test_xgb.shape}")
    print(f"Last predicted value: {pred_test_xgb[-1]}")

    predicted_price, prediction_date, prediction_hours = plotting(y_val, y_test, pred_test_xgb[-1], mae, WINDOW, PREDICTION_SCOPE)
    
    print(f"Predicted price: {predicted_price}")
    print(f"Prediction date: {prediction_date}")
    print(f"Prediction hours: {prediction_hours}")

    return predicted_price, prediction_date, prediction_hours, mae

def main():
    symbol = args.symbol.upper()
    interval = '1h'
    
    try:
        print(f"Starting prediction for {symbol} with {interval} interval")
        current_price = get_current_price(symbol)
        if current_price is None:
            raise ValueError(f"Could not get current price for {symbol}")
        print(f"Current price obtained: ${current_price:.4f}")

        predicted_price, prediction_date, prediction_hours, mae = predictPrice(interval)
        if predicted_price is None:
            raise ValueError(f"Could not get price prediction for {symbol}")
        print(f"Predicted price: ${predicted_price:.4f} for {prediction_date}")
        print(f"MAE (Mean Absolute Error): {mae:.4f}")

        trade_direction = determine_trade_direction(current_price, predicted_price)
        print(f"Trade direction: {trade_direction}")
        
        historical_prices = get_historical_data(symbol, interval=interval)
        print("Historical data obtained:")
        print(historical_prices)

        highest_volume_prices = get_highest_volume_prices(historical_prices, n=4)

        if historical_prices.empty:
            print("Warning: Could not obtain historical data. Using alternative values.")
            highest_price_7d = max(current_price, predicted_price)
            lowest_price_7d = min(current_price, predicted_price)
            entry, tp1, tp2, stop_loss = current_price, predicted_price, predicted_price, current_price * (1.001 if "SHORT" in trade_direction else 0.999)
        else:
            highest_price_7d = historical_prices['High'].max()
            lowest_price_7d = historical_prices['Low'].min()
            entry, tp1, tp2, stop_loss = find_entry_and_targets(historical_prices, current_price, predicted_price, trade_direction)

        print(f"Calculated levels: Entry=${entry:.4f}, TP1=${tp1:.4f}, TP2=${tp2:.4f}, SL=${stop_loss:.4f}")

        now = pd.Timestamp.now(tz=colombia_tz)
        specific_prices = {}
        for hours in range(1, 25, 6):  # Get prices for 1, 7, 13, 19 hours ago
            price_key = f"price_{hours}h_ago"
            price_date = now - pd.Timedelta(hours=hours)
            if historical_prices.empty:
                specific_prices[price_key] = None
            else:
                hour_prices = historical_prices[historical_prices.index.floor('H') == price_date.floor('H')]
                if not hour_prices.empty:
                    if "SHORT" in trade_direction:
                        specific_prices[price_key] = float(hour_prices['High'].max())
                    else:
                        specific_prices[price_key] = float(hour_prices['Low'].min())
                else:
                    specific_prices[price_key] = None
        
        prediction_data = {
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "prediction_date": prediction_date.strftime('%Y-%m-%d %H:%M'),
            "prediction_hours": int(prediction_hours),
            "mae": float(mae),
            "trade_direction": trade_direction,
            "entry_price": float(entry),
            "stop_loss": float(stop_loss),
            "target_price_1": float(tp1),
            "target_price_2": float(tp2),
            "highest_price_7d": float(highest_price_7d),
            "lowest_price_7d": float(lowest_price_7d),
            "token": symbol,
            "highest_volume_prices": highest_volume_prices,
            **specific_prices
        }
        
        print("Saving prediction data...")
        filename = f'../data/prediction-{symbol}.json'
        with open(filename, 'w') as json_file:
            json.dump(prediction_data, json_file, indent=4)
        print(f"Prediction data saved successfully in {filename}")
        
        print(json.dumps(prediction_data, indent=4))
        
        print(f"Values for chart: entry={entry}, tp1={tp1}, tp2={tp2}, stop_loss={stop_loss}, trade_direction={trade_direction}")
        
        print("Verifying historical data:")
        print(historical_prices)
        print(f"Shape of historical data: {historical_prices.shape}")
        
        chart_path = None
        if not historical_prices.empty and len(historical_prices) >= 2:
            print("Attempting to create chart...")
            chart_path = create_chart(symbol, hours=168, entry=entry, tp1=tp1, tp2=tp2, stop_loss=stop_loss, trade_direction=trade_direction, interval=interval)
            if chart_path:
                print(f"Chart created successfully at: {chart_path}")
            else:
                print("Could not create chart")
        else:
            print("Could not create chart due to insufficient historical data.")
            print(f"Available historical data: {len(historical_prices)} periods")

        current_date = datetime.now(colombia_tz).strftime('%Y-%m-%d')
        current_time = datetime.now(colombia_tz).strftime('%H:%M')
        
        message = f"""
*{symbol} Prediction* for {current_date}

Prediction for the next {prediction_hours} hours:

- Current Price: ${prediction_data['current_price']:.4f}
- Predicted Price: ${prediction_data['predicted_price']:.4f}
- Trade Direction: {prediction_data['trade_direction']}
- MAE (Mean Absolute Error): {prediction_data['mae']:.4f}

Trading Levels:
- Entry Price: ${prediction_data['entry_price']:.4f}
- Stop Loss: ${prediction_data['stop_loss']:.4f}
- Target 1 (TP1): ${prediction_data['target_price_1']:.4f}
- Target 2 (TP2): ${prediction_data['target_price_2']:.4f}

Price Range (last 7 days):
- Highest Price: ${prediction_data['highest_price_7d']:.4f}
- Lowest Price: ${prediction_data['lowest_price_7d']:.4f}

Prices with Highest Volume (last 7 days):
{chr(10).join([f"- {price['date']}: ${price['price']:.4f} (Volume: {price['volume']:,.0f})" for price in prediction_data['highest_volume_prices']])}

Generated on {current_date} at {current_time}
"""

        print("Preparing to send to Telegram...")
        if chart_path and os.path.exists(chart_path):
            print(f"Chart file found at {chart_path}")
            send_to_telegram(message, chart_path)
        else:
            print(f"Could not find chart file at {chart_path if chart_path else 'any location'}. Sending message only.")
            send_to_telegram(message, None)

        for hours in range(0, 25, 6):
            specific_date = now - pd.Timedelta(hours=hours)
            if not historical_prices.empty:
                print(f"\nData from {hours} hours ago ({specific_date}):")
                print(historical_prices.loc[historical_prices.index.floor('H') == specific_date.floor('H')])
            else:
                print(f"\nNo data available for {hours} hours ago ({specific_date})")

        print(f"\nDebug Information:")
        print(f"Current Price: {current_price}")
        print(f"Predicted Price: {predicted_price}")
        print(f"Trade Direction: {trade_direction}")
        print(f"Entry Price: {entry}")
        print(f"Stop Loss: {stop_loss}")
        print(f"Target Price 1: {tp1}")
        print(f"Target Price 2: {tp2}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()