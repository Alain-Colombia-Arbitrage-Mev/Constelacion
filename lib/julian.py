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
import argparse  # Añadida esta línea

warnings.filterwarnings("ignore")

colombia_tz = pytz.timezone('America/Bogota')

# Argparse setup
parser = argparse.ArgumentParser(description='Predict cryptocurrency prices using XGBoost.')
parser.add_argument('symbol', type=str, help='The symbol to predict (e.g., BTC-USD, AVAX-USD)')
args = parser.parse_args()

# Global variables
SYMBOL = args.symbol.upper()
INTERVAL = '1d'  # Fixed interval



def feature_engineering(data, SPY, predictions=np.array([None]))->pd.core.frame.DataFrame:
    assert type(data) == pd.core.frame.DataFrame, "data must be a dataframe"
    assert type(SPY) == pd.core.series.Series, "SPY must be a dataframe"
    assert type(predictions) == np.ndarray, "predictions must be an array"

    if predictions.any() == True:
        data = yf.download(SYMBOL, start="2009-11-30")
        SPY = yf.download("SPY", start="2001-11-30")["Close"]
        data = features(data, SPY)
        data["Predictions"] = predictions
        data["Close"] = data["Close_y"]
        data.drop("Close_y", 1, inplace=True)
        data.dropna(0, inplace=True)
    data = features(data, SPY)
    return data

def features(data, SPY)->pd.core.frame.DataFrame:
    for i in [2, 3, 4, 5, 6, 7]:
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).mean()
        data[f"Volume{i}"] = data["Volume"].rolling(i).mean()
        data[f"Low_std{i}"] = data["Low"].rolling(i).std()
        data[f"High_std{i}"] = data["High"].rolling(i).std()
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).std()
        data[f"Close{i}"] = data["Close"].shift(i)
        data[f"Adj_Close{i}_max"] = data["Adj Close"].rolling(i).max()
        data[f"Adj_Close{i}_min"] = data["Adj Close"].rolling(i).min()
        data[f"Adj_Close{i}_quantile"] = data["Adj Close"].rolling(i).quantile(1)

    data["SPY"] = SPY
    data["Day"] = data.index.day
    data["Month"] = data.index.month
    data["Year"] = data.index.year
    data["day_year"] = data.index.day_of_year
    data["Weekday"] = data.index.weekday
    data["Upper_Shape"] = data["High"] - np.maximum(data["Open"], data["Close"])
    data["Lower_Shape"] = np.minimum(data["Open"], data["Close"]) - data["Low"]
    data["Close_y"] = data["Close"]
    return data

def windowing(train, val, WINDOW, PREDICTION_SCOPE):
    assert type(train) == np.ndarray, "train must be passed as an array"
    assert type(val) == np.ndarray, "validation must be passed as an array"
    assert type(WINDOW) == int, "Window must be an integer"
    assert type(PREDICTION_SCOPE) == int, "Prediction scope must be an integer"

    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(len(train)-(WINDOW+PREDICTION_SCOPE)):
        X, y = np.array(train[i:i+WINDOW, :-1]), np.array(train[i+WINDOW+PREDICTION_SCOPE, -1])
        X_train.append(X)
        y_train.append(y)

    for i in range(len(val)-(WINDOW+PREDICTION_SCOPE)):
        X, y = np.array(val[i:i+WINDOW, :-1]), np.array(val[i+WINDOW+PREDICTION_SCOPE, -1])
        X_test.append(X)
        y_test.append(y)

    return X_train, y_train, X_test, y_test

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

    x_ticks = list(stock_prices.index[-time:])+[stock_prices.index[-1]+timedelta(PREDICTION_SCOPE+1)]

    _predictprice = round(ploting_pred[-1][0],2)
    _date = x_ticks[-1]
    _days = PREDICTION_SCOPE+1

    return _predictprice, _date, _days

def train_xgb_model(X_train, y_train, X_val, y_val, plotting=False):
    model = xgbs.XGBRegressor(gamma=1, n_estimators=200)
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)

    if plotting:
        plt.figure(figsize=(15, 6))
        sns.set_theme(style="white")
        sns.lineplot(x=range(len(y_val)), y=y_val, color="grey", alpha=.4)
        sns.lineplot(x=range(len(y_val)), y=pred_val, color="red")
        plt.xlabel("Time")
        plt.ylabel("{symbol} stock price")
        plt.title(f"The MAE for this period is: {round(mae, 3)}")

    return mae, model

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

def get_historical_prices(symbol, days=7, interval='1d'):
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if data.empty:
        print(f"Advertencia: No se pudieron obtener datos históricos para {symbol}")
        return pd.DataFrame(columns=['Close', 'High', 'Low', 'Volume'])
    return data[['Close', 'High', 'Low', 'Volume']]

def get_highest_volume_prices(historical_prices, n=4):
    if historical_prices.empty:
        return []
    
    highest_volume_days = historical_prices.sort_values('Volume', ascending=False).head(n)
    
    highest_volume_prices = [
        {
            'date': date.strftime('%Y-%m-%d %H:%M'),
            'price': price,
            'volume': volume
        }
        for date, price, volume in zip(highest_volume_days.index, highest_volume_days['Close'], highest_volume_days['Volume'])
    ]
    
    return highest_volume_prices

def determine_trade_direction(current_price, predicted_price):
    return "LONG 📈" if predicted_price > current_price else "SHORT 📉"

def find_entry_and_targets(historical_prices, current_price, predicted_price, trade_direction):
    if historical_prices.empty:
        print("Advertencia: Datos históricos vacíos. Usando precios actuales y predichos.")
        highest_price = max(current_price, predicted_price)
        lowest_price = min(current_price, predicted_price)
    else:
        highest_price = historical_prices['High'].max()
        lowest_price = historical_prices['Low'].min()
    
    if "SHORT" in trade_direction:
        entry = highest_price
        tp1 = min(predicted_price, current_price)
        tp2 = lowest_price
        stop_loss = entry * 1.03
    else:  # LONG
        entry = lowest_price
        tp1 = max(predicted_price, current_price)
        tp2 = highest_price
        stop_loss = entry * 0.97
    
    return entry, tp1, tp2, stop_loss

def create_chart(symbol, days=7, entry=None, tp1=None, tp2=None, stop_loss=None, trade_direction=None, interval='1d'):
    print(f"Iniciando creación del gráfico para {symbol} con intervalo {interval}")
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    
    print(f"Datos descargados para el gráfico: {len(data)} filas")
    print(data.head())
    
    if len(data) < 2:
        print(f"No hay suficientes datos para crear el gráfico de {symbol}")
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
            mpf.make_addplot(entry_line, color=color, linestyle='--', label=f'Entry: {entry:.2f}'),
            mpf.make_addplot(tp1_line, color=color, linestyle=':', label=f'TP1: {tp1:.2f}'),
            mpf.make_addplot(tp2_line, color=color, linestyle=':', label=f'TP2: {tp2:.2f}'),
            mpf.make_addplot(stop_loss_line, color='purple', linestyle='-.', label=f'SL: {stop_loss:.2f}')
        ])
    
    if not data['SMA20'].isnull().all():
        annotations.append(mpf.make_addplot(data['SMA20'].astype(float), color='orange', label='SMA20'))
    
    try:
        print("Creando el gráfico...")
        fig, axes = mpf.plot(data, type='candle', style=s, volume=True, 
                             addplot=annotations if annotations else None, 
                             title=f'\n{symbol} Price Chart (Last {days} Days, {interval} interval)',
                             ylabel='Price',
                             ylabel_lower='Volume',
                             figsize=(12, 8),
                             returnfig=True)
        
        axes[0].legend(loc='upper left')
        
        chart_path = os.path.abspath(f'{symbol}_chart.png')
        print(f"Intentando guardar el gráfico en: {chart_path}")
        plt.savefig(chart_path)
        plt.close(fig)
        print(f"Gráfico creado y guardado exitosamente en {chart_path}")
        return chart_path
    except Exception as e:
        print(f"Error al crear el gráfico: {e}")
        import traceback
        print(traceback.format_exc())
        return None

async def send_to_telegram_async(message, image_path):
    bot_token = '6848512889:AAG2fBYJ-dcblpngnvRB4Pexw19d_E_kkR0'
    chat_id = '-1002225888276'
    
    bot = Bot(token=bot_token)
    
    try:
        print(f"Intentando enviar mensaje a Telegram. Chat ID: {chat_id}")
        message_result = await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        print(f"Mensaje enviado. Message ID: {message_result.message_id}")
        
        if image_path and os.path.exists(image_path):
            print(f"Intentando enviar imagen desde {image_path}...")
            with open(image_path, 'rb') as image_file:
                photo_result = await bot.send_photo(chat_id=chat_id, photo=image_file)
            print(f"Imagen enviada. Photo ID: {photo_result.message_id}")
        elif image_path:
            print(f"No se pudo encontrar la imagen en {image_path}")
        else:
            print("No se proporcionó ruta de imagen.")
        
        print("Proceso de envío a Telegram completado.")
    except Exception as e:
        print(f"Error al enviar mensaje a Telegram: {e}")
        import traceback
        print(traceback.format_exc())

def send_to_telegram(message, image_path):
    print(f"Iniciando envío a Telegram. Ruta de imagen: {image_path}")
    asyncio.run(send_to_telegram_async(message, image_path))

def predictPrice(interval='1d'):
    global stock_prices, SPY, y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE

    PERCENTAGE = 0.995
    WINDOW = 2
    PREDICTION_SCOPE = 0

    stock_prices = yf.download(SYMBOL, interval=INTERVAL)
    SPY = yf.download("SPY", interval=interval)["Close"]

    stock_prices = feature_engineering(stock_prices, SPY)

    train, test = train_test_split(stock_prices, WINDOW)
    train_set, validation_set = train_validation_split(train, PERCENTAGE)

    X_train, y_train, X_val, y_val = windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)

    X_train = np.array([x.flatten() for x in X_train])
    y_train = np.array(y_train)
    X_val = np.array([x.flatten() for x in X_val])
    y_val = np.array(y_val)

    mae, xgb_model = train_xgb_model(X_train, y_train, X_val, y_val, plotting=False)

    X_test = np.array(test.iloc[:, :-1]).reshape(1, -1)
    y_test = np.array(test.iloc[:, -1])

    pred_test_xgb = xgb_model.predict(X_test)
    predicted_price, prediction_date, prediction_days = plotting(y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE)
    
    return predicted_price, prediction_date, prediction_days, mae

def main():

    symbol = args.symbol.upper()
    interval = '1d'  # Puedes cambiar esto a '4h' cuando quieras usar velas de 4 horas
    
    try:
        print(f"Iniciando predicción para {symbol} con intervalo {interval}")
        current_price = get_current_price(symbol)
        if current_price is None:
            raise ValueError(f"No se pudo obtener el precio actual para {symbol}")
        print(f"Precio actual obtenido: ${current_price:.2f}")

        predicted_price, prediction_date, prediction_days, mae = predictPrice(interval)
        if predicted_price is None:
            raise ValueError(f"No se pudo obtener la predicción de precio para {symbol}")
        print(f"Precio predicho: ${predicted_price:.2f} para {prediction_date}")
        print(f"MAE (Error Medio Absoluto): {mae:.2f}")

        trade_direction = determine_trade_direction(current_price, predicted_price)
        print(f"Dirección del trade: {trade_direction}")
        
        historical_prices = get_historical_prices(symbol, days=7, interval=interval)
        print("Datos históricos obtenidos:")
        print(historical_prices)

        highest_volume_prices = get_highest_volume_prices(historical_prices, n=4)

        if historical_prices.empty:
            print("Advertencia: No se pudieron obtener datos históricos. Usando valores alternativos.")
            highest_price_4d = max(current_price, predicted_price)
            lowest_price_4d = min(current_price, predicted_price)
            entry, tp1, tp2, stop_loss = current_price, predicted_price, predicted_price, current_price * (1.03 if "SHORT" in trade_direction else 0.97)
        else:
            highest_price_4d = historical_prices['High'].max()
            lowest_price_4d = historical_prices['Low'].min()
            entry, tp1, tp2, stop_loss = find_entry_and_targets(historical_prices, current_price, predicted_price, trade_direction)

        print(f"Niveles calculados: Entry=${entry:.2f}, TP1=${tp1:.2f}, TP2=${tp2:.2f}, SL=${stop_loss:.2f}")

        now = datetime.now(colombia_tz)
        specific_prices = {}
        for days in range(1, 5):
            price_key = f"price_{days}d_ago"
            price_date = now.date() - timedelta(days=days)
            if historical_prices.empty:
                specific_prices[price_key] = None
            else:
                day_prices = historical_prices[historical_prices.index.date == price_date]
                if not day_prices.empty:
                    if "SHORT" in trade_direction:
                        specific_prices[price_key] = float(day_prices['High'].max())
                    else:
                        specific_prices[price_key] = float(day_prices['Low'].min())
                else:
                    specific_prices[price_key] = None
        
        prediction_data = {
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "prediction_date": prediction_date.strftime('%Y-%m-%d'),
            "prediction_days": int(prediction_days),
            "mae": float(mae),
            "trade_direction": trade_direction,
            "entry_price": float(entry),
            "stop_loss": float(stop_loss),
            "target_price_1": float(tp1),
            "target_price_2": float(tp2),
            "highest_price_4d": float(highest_price_4d),
            "lowest_price_4d": float(lowest_price_4d),
            "token": symbol,
            "highest_volume_prices": highest_volume_prices,
            **specific_prices
        }
        
        print("Guardando datos de predicción...")
        filename = f'../data/prediction-{symbol}.json'
        with open(filename, 'w') as json_file:
            json.dump(prediction_data, json_file, indent=4)
        print(f"Datos de predicción guardados exitosamente en {filename}")
        
        print(json.dumps(prediction_data, indent=4))
        
        print(f"Valores para el gráfico: entry={entry}, tp1={tp1}, tp2={tp2}, stop_loss={stop_loss}, trade_direction={trade_direction}")
        
        print("Verificando datos históricos:")
        print(historical_prices)
        print(f"Forma de los datos históricos: {historical_prices.shape}")
        
        chart_path = None
        if not historical_prices.empty and len(historical_prices) >= 2:
            print("Intentando crear el gráfico...")
            chart_path = create_chart(symbol, days=7, entry=entry, tp1=tp1, tp2=tp2, stop_loss=stop_loss, trade_direction=trade_direction, interval=interval)
            if chart_path:
                print(f"Gráfico creado exitosamente en: {chart_path}")
            else:
                print("No se pudo crear el gráfico")
        else:
            print("No se pudo crear el gráfico debido a la falta de datos históricos suficientes.")
            print(f"Datos históricos disponibles: {len(historical_prices)} períodos")

        current_date = datetime.now(colombia_tz).strftime('%Y-%m-%d')
        current_time = datetime.now(colombia_tz).strftime('%H:%M')
        
        message = f"""
*{symbol} Prediction* para {current_date}

Prediction for the next {prediction_days} days:

- Current Price: ${prediction_data['current_price']:.2f}
- Predicted Price: ${prediction_data['predicted_price']:.2f}
- Trade Direction: {prediction_data['trade_direction']}
- MAE (Mean Absolute Error): {prediction_data['mae']:.2f}

Trading Levels:
- Entry Price: ${prediction_data['entry_price']:.2f}
- Stop Loss: ${prediction_data['stop_loss']:.2f}
- Target 1 (TP1): ${prediction_data['target_price_1']:.2f}
- Target 2 (TP2): ${prediction_data['target_price_2']:.2f}

Price Range (last 7 days):
- Highest Price: ${prediction_data['highest_price_4d']:.2f}
- Lowest Price: ${prediction_data['lowest_price_4d']:.2f}

Prices with Highest Volume (last 7 days):
{chr(10).join([f"- {price['date']}: ${price['price']:.2f} (Volume: {price['volume']:,.0f})" for price in prediction_data['highest_volume_prices']])}

Generated on {current_date} at {current_time}
"""

        print("Preparando envío a Telegram...")
        if chart_path and os.path.exists(chart_path):
            print(f"Archivo de gráfico encontrado en {chart_path}")
            send_to_telegram(message, chart_path)
        else:
            print(f"No se pudo encontrar el archivo del gráfico en {chart_path if chart_path else 'ninguna ubicación'}. Enviando solo el mensaje.")
            send_to_telegram(message, None)

        for days in range(7):
            specific_date = now.date() - timedelta(days=days)
            if not historical_prices.empty:
                print(f"\nDatos de hace {days} días ({specific_date}):")
                print(historical_prices.loc[historical_prices.index.date == specific_date])
            else:
                print(f"\nNo hay datos disponibles para hace {days} días ({specific_date})")

        print(f"\nDebug Information:")
        print(f"Current Price: {current_price}")
        print(f"Predicted Price: {predicted_price}")
        print(f"Trade Direction: {trade_direction}")
        print(f"Entry Price: {entry}")
        print(f"Stop Loss: {stop_loss}")
        print(f"Target Price 1: {tp1}")
        print(f"Target Price 2: {tp2}")

    except Exception as e:
        print(f"Se produjo un error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()