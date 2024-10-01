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

# Argparse setup
parser = argparse.ArgumentParser(description='Predict cryptocurrency prices using XGBoost.')
parser.add_argument('symbol', type=str, help='The symbol to predict (e.g., BTC-USD, AVAX-USD)')
args = parser.parse_args()

# Global variables
SYMBOL = args.symbol.upper()
INTERVAL = '1h'
WINDOW = 24
PREDICTION_SCOPE = 24
PERCENTAGE = 0.8

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

def get_historical_prices(symbol, hours=168, interval='1h'):
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(hours=hours)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if data.empty:
        print(f"Advertencia: No se pudieron obtener datos históricos para {symbol}")
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    data.index = pd.to_datetime(data.index)
    return data

def get_highest_volume_prices(historical_prices, hours=42, n=4):
    if historical_prices.empty:
        return []
    
    end_time = historical_prices.index[-1]
    start_time = end_time - timedelta(hours=hours)
    filtered_data = historical_prices[historical_prices.index >= start_time]
    
    # Filtrar períodos con volumen cero
    non_zero_volume = filtered_data[filtered_data['Volume'] > 0]
    
    if non_zero_volume.empty:
        print(f"Advertencia: No hay datos de volumen no cero en las últimas {hours} horas")
        return []
    
    highest_volume_periods = non_zero_volume.sort_values('Volume', ascending=False).head(n)
    
    highest_volume_prices = [
        {
            'date': date.strftime('%Y-%m-%d %H:%M'),
            'price': price,
            'volume': volume
        }
        for date, price, volume in zip(highest_volume_periods.index, highest_volume_periods['Close'], highest_volume_periods['Volume'])
    ]
    
    return highest_volume_prices

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean().iloc[-1]

def find_support_resistance(data, window=10):
    highs = data['High'].rolling(window=window, center=True).max()
    lows = data['Low'].rolling(window=window, center=True).min()
    support = lows.iloc[-1]
    resistance = highs.iloc[-1]
    return support, resistance
def find_entry_and_targets(historical_prices, current_price, predicted_price, trade_direction):
    atr = calculate_atr(historical_prices, period=24)
    recent_data = historical_prices.tail(42)  # Últimas 42 horas
    low = min(recent_data['Low'].min(), current_price)
    high = max(recent_data['High'].max(), current_price)
    support, resistance = find_support_resistance(recent_data, window=12)
    
    if "SHORT" in trade_direction:
        entry = min(high, current_price * 1.01)  # No más del 1% sobre el precio actual
        stop_loss = min(entry * 1.03, entry + (2 * atr))
        risk = stop_loss - entry
        
        # Aseguramos que TP1 sea el precio predicho
        tp1 = predicted_price
        
        # TP2 se calcula basado en el riesgo, pero no puede ser mayor que TP1 en una operación corta
        tp2 = max(entry - (2.5 * risk), low, support)
        tp2 = max(tp2, tp1)  # Aseguramos que TP2 no sea mayor que TP1 en una operación corta
        
    else:  # LONG
        entry = max(low, current_price * 0.99)  # No menos del 1% bajo el precio actual
        stop_loss = max(entry * 0.97, entry - (2 * atr))
        risk = entry - stop_loss
        
        # Aseguramos que TP1 sea el precio predicho
        tp1 = predicted_price
        
        # TP2 se calcula basado en el riesgo, pero no puede ser menor que TP1 en una operación larga
        tp2 = min(entry + (2.5 * risk), high, resistance)
        tp2 = min(tp2, tp1)  # Aseguramos que TP2 no sea menor que TP1 en una operación larga
    
    return entry, tp1, tp2, stop_loss

def determine_trade_direction(current_price, predicted_price):
    return "LONG 📈" if predicted_price > current_price else "SHORT 📉"

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
        X, y = train[i:i+WINDOW], train[i+WINDOW+PREDICTION_SCOPE, -1]
        X_train.append(X)
        y_train.append(y)

    for i in range(len(val)-(WINDOW+PREDICTION_SCOPE)):
        X, y = val[i:i+WINDOW], val[i+WINDOW+PREDICTION_SCOPE, -1]
        X_test.append(X)
        y_test.append(y)

    if not X_train or not y_train:
        raise ValueError("No hay suficientes datos en el conjunto de entrenamiento para crear ventanas")
    if not X_test or not y_test:
        raise ValueError("No hay suficientes datos en el conjunto de validación para crear ventanas")

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def train_test_split(data, WINDOW):
    assert isinstance(data, pd.DataFrame), "data must be a dataframe"
    assert isinstance(WINDOW, int), "Window must be an integer"

    train = data.iloc[:-WINDOW]
    test = data.iloc[-WINDOW:]

    return train, test

def train_validation_split(train, percentage):
    assert isinstance(train, pd.DataFrame), "train must be a dataframe"
    assert isinstance(percentage, float), "percentage must be a float"

    split_index = int(len(train) * percentage)
    if len(train) - split_index < WINDOW + PREDICTION_SCOPE:
        split_index = len(train) - (WINDOW + PREDICTION_SCOPE)

    if split_index <= 0:
        raise ValueError("No hay suficientes datos para crear conjuntos de entrenamiento y validación")

    train_set = train.iloc[:split_index].values
    validation_set = train.iloc[split_index:].values

    return train_set, validation_set

def plotting(y_val, y_test, pred_test, mae, WINDOW, PREDICTION_SCOPE):
    assert type(WINDOW) == int, "Window must be an integer"
    assert type(PREDICTION_SCOPE) == int, "Prediction scope must be an integer"

    ploting_pred = [y_test[-1], pred_test]
    ploting_test = [y_val[-1]]+list(y_test)

    time = (len(y_val)-1)+(len(ploting_test)-1)+(len(ploting_pred)-1)

    x_ticks = list(stock_prices.index[-time:])+[stock_prices.index[-1]+timedelta(hours=PREDICTION_SCOPE*4)]

    _predictprice = round(ploting_pred[-1][0],2)
    _date = x_ticks[-1]
    _hours = PREDICTION_SCOPE * 4

    return _predictprice, _date, _hours

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
        plt.ylabel(f"{SYMBOL} stock price")
        plt.title(f"The MAE for this period is: {round(mae, 3)}")

    return mae, model

def create_chart(symbol, historical_prices, entry, tp1, tp2, stop_loss, trade_direction, interval='1h'):
    if historical_prices.empty or len(historical_prices) < 2:
        print(f"No hay suficientes datos para crear el gráfico de {symbol}")
        return None
    
    data = historical_prices.copy()
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
        annotations.append(mpf.make_addplot(data['SMA20'], color='orange', label='SMA20'))
    
    try:
        fig, axes = mpf.plot(data, type='candle', style=s, volume=True, 
                             addplot=annotations if annotations else None, 
                             title=f'\n{symbol} Price Chart (Last {len(data)} Hours, {interval} interval)',
                             ylabel='Price',
                             ylabel_lower='Volume',
                             figsize=(12, 8),
                             returnfig=True)
        
        axes[0].legend(loc='upper left')
        
        chart_path = os.path.abspath(f'{symbol}_chart.png')
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
    chat_id = '-1002151337518'
    
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

def predictPrice(interval='1h'):
    global stock_prices, SPY, y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE

    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=60)

    print(f"Descargando datos para {SYMBOL} desde {start_date} hasta {end_date}")
    try:
        stock_prices = yf.download(SYMBOL, start=start_date, end=end_date, interval=interval)
        if stock_prices.empty:
            print(f"No se pudieron descargar datos para {SYMBOL}")
            return None, None, None, None
        
        SPY = yf.download("SPY", start=start_date, end=end_date, interval=interval)["Close"]
        if SPY.empty:
            print("No se pudieron descargar datos para SPY")
            return None, None, None, None
    except Exception as e:
        print(f"Error al descargar datos: {e}")
        return None, None, None, None

    print(f"Datos descargados para {SYMBOL}: {len(stock_prices)} filas")
    print(f"Primeras filas de stock_prices:")
    print(stock_prices.head())
    print(f"Últimas filas de stock_prices:")
    print(stock_prices.tail())
    print(f"Datos descargados para SPY: {len(SPY)} filas")
    print(f"Primeras filas de SPY:")
    print(SPY.head())
    print(f"Últimas filas de SPY:")
    print(SPY.tail())

    if len(stock_prices) < WINDOW + PREDICTION_SCOPE:
        print(f"No hay suficientes datos. Se necesitan al menos {WINDOW + PREDICTION_SCOPE} filas, pero solo hay {len(stock_prices)}")
        return None, None, None, None

    stock_prices.index = pd.to_datetime(stock_prices.index)
    SPY.index = pd.to_datetime(SPY.index)

    stock_prices = feature_engineering(stock_prices, SPY)

    train, test = train_test_split(stock_prices, WINDOW)
    train_set, validation_set = train_validation_split(train, PERCENTAGE)

    print(f"Tamaño del conjunto de entrenamiento: {train_set.shape}")
    print(f"Tamaño del conjunto de validación: {validation_set.shape}")

    try:
        X_train, y_train, X_val, y_val = windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)
    except ValueError as e:
        print(f"Error en windowing: {e}")
        return None, None, None, None

    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de y_train: {y_train.shape}")
    print(f"Forma de X_val: {X_val.shape}")
    print(f"Forma de y_val: {y_val.shape}")

    if X_val.size == 0 or y_val.size == 0:
        print("El conjunto de validación está vacío. Ajusta el porcentaje de división o aumenta el tamaño de los datos.")
        return None, None, None, None

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    try:
        mae, xgb_model = train_xgb_model(X_train, y_train, X_val, y_val, plotting=False)
    except Exception as e:
        print(f"Error en el entrenamiento del modelo: {e}")
        return None, None, None, None

    X_test = test.values.reshape(1, -1)
    y_test = test.iloc[:, -1].values

    try:
        pred_test_xgb = xgb_model.predict(X_test)
        predicted_price, prediction_date, prediction_hours = plotting(y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE)
    except Exception as e:
        print(f"Error en la predicción o plotting: {e}")
        return None, None, None, None
    
    return predicted_price, prediction_date, prediction_hours, mae

def main():
    symbol = args.symbol.upper()
    interval = '1h'
    
    try:
        print(f"Iniciando predicción para {symbol} con intervalo {interval}")
        current_price = get_current_price(symbol)
        if current_price is None:
            raise ValueError(f"No se pudo obtener el precio actual para {symbol}")
        print(f"Precio actual obtenido: ${current_price:.4f}")

        predicted_price, prediction_date, prediction_hours, mae = predictPrice(interval)
        if predicted_price is None:
            raise ValueError(f"No se pudo obtener la predicción de precio para {symbol}. Revisa los mensajes de error anteriores.")
        print(f"Precio predicho: ${predicted_price:.4f} para {prediction_date}")
        print(f"MAE (Error Medio Absoluto): {mae:.4f}")

        trade_direction = determine_trade_direction(current_price, predicted_price)
        print(f"Dirección del trade: {trade_direction}")
        
        historical_prices = get_historical_prices(symbol, hours=168, interval=interval)
        recent_prices = historical_prices.tail(42)  # Últimas 42 horas

        highest_volume_prices = get_highest_volume_prices(recent_prices, hours=42, n=4)

        entry, tp1, tp2, stop_loss = find_entry_and_targets(recent_prices, current_price, predicted_price, trade_direction)

        print(f"Niveles calculados: Entry=${entry:.4f}, TP1=${tp1:.4f}, TP2=${tp2:.4f}, SL=${stop_loss:.4f}")

        now = datetime.now(colombia_tz)
        current_date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M')
        
        volume_price_lines = []
        for price in highest_volume_prices:
            formatted_line = f"- {price['date']}: ${price['price']:.4f} (Volumen: {price['volume']:,.0f})"
            volume_price_lines.append(formatted_line)
        volume_price_info = "\n".join(volume_price_lines)

        message = f"""
*{symbol} Prediction* para {current_date}

Predicción para las próximas {prediction_hours} horas:

- Precio Actual: ${current_price:.4f}
- Precio Predicho: ${predicted_price:.4f}
- Dirección del Trade: {trade_direction}
- MAE (Error Medio Absoluto): {mae:.4f}

Niveles de Trading (basados en datos de las últimas 42 horas):
- Precio de Entrada: ${entry:.4f}
- Stop Loss: ${stop_loss:.4f}
- Objetivo 1 (TP1, Precio Predicho): ${tp1:.4f}
- Objetivo 2 (TP2): ${tp2:.4f}

Rango de Precios (últimas 42 horas):
- Precio Más Alto: ${recent_prices['High'].max():.4f}
- Precio Más Bajo: ${recent_prices['Low'].min():.4f}

Precios con Mayor Volumen (últimas 42 horas):
{volume_price_info}

Generado el {current_date} a las {current_time}
"""

        print("Preparando envío a Telegram...")
        chart_path = create_chart(symbol, recent_prices, entry, tp1, tp2, stop_loss, trade_direction, interval)
        if chart_path and os.path.exists(chart_path):
            print(f"Archivo de gráfico encontrado en {chart_path}")
            send_to_telegram(message, chart_path)
        else:
            print(f"No se pudo encontrar el archivo del gráfico en {chart_path if chart_path else 'ninguna ubicación'}. Enviando solo el mensaje.")
            send_to_telegram(message, None)

        print(f"\nDebug Information:")
        print(f"Current Price: {current_price:.4f}")
        print(f"Predicted Price: {predicted_price:.4f}")
        print(f"Trade Direction: {trade_direction}")
        print(f"Entry Price: {entry:.4f}")
        print(f"Stop Loss: {stop_loss:.4f}")
        print(f"Target Price 1: {tp1:.4f}")
        print(f"Target Price 2: {tp2:.4f}")
        print(f"\nRecent price range:")
        print(f"Highest price (42h): {recent_prices['High'].max():.4f}")
        print(f"Lowest price (42h): {recent_prices['Low'].min():.4f}")

    except Exception as e:
        print(f"Se produjo un error: {e}")
        import traceback
        print(traceback.format_exc())
        print(f"Información adicional de depuración:")
        print(f"Símbolo: {symbol}")
        print(f"Intervalo: {interval}")
        print(f"Precio actual: {current_price if 'current_price' in locals() else 'No disponible'}")

if __name__ == "__main__":
    main()