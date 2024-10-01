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
INTERVAL = '1h'  # Changed to 1h

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

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

def get_historical_prices(symbol, hours=42, interval='1h'):
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(hours=hours)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if data.empty:
        print(f"Advertencia: No se pudieron obtener datos históricos para {symbol}")
        return pd.DataFrame(columns=['Close', 'High', 'Low', 'Volume'])
    data.index = pd.to_datetime(data.index)
    return data

def get_highest_volume_prices(historical_prices, n=4):
    if historical_prices.empty:
        return []
    
    highest_volume_periods = historical_prices.sort_values('Volume', ascending=False).head(n)
    
    highest_volume_prices = [
        {
            'date': date.strftime('%Y-%m-%d %H:%M'),
            'price': price,
            'volume': volume
        }
        for date, price, volume in zip(highest_volume_periods.index, highest_volume_periods['Close'], highest_volume_periods['Volume'])
    ]
    
    return highest_volume_prices

def determine_trade_direction(current_price, predicted_price):
    return "LONG 📈" if predicted_price > current_price else "SHORT 📉"

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean().iloc[-1]

def fibonacci_levels(low, high):
    diff = high - low
    return {
        0: low,
        0.236: low + 0.236 * diff,
        0.382: low + 0.382 * diff,
        0.5: low + 0.5 * diff,
        0.618: low + 0.618 * diff,
        1: high,
        1.618: high + 0.618 * diff,
        2.618: high + 1.618 * diff
    }

def find_support_resistance(data, window=10):
    highs = data['High'].rolling(window=window, center=True).max()
    lows = data['Low'].rolling(window=window, center=True).min()
    support = lows.iloc[-1]
    resistance = highs.iloc[-1]
    return support, resistance

def find_entry_and_targets(historical_prices, current_price, predicted_price, trade_direction):
    atr = calculate_atr(historical_prices, period=24)  # 24 períodos de 1 hora = 1 día
    low = min(historical_prices['Low'].min(), current_price)
    high = max(historical_prices['High'].max(), current_price)
    fib_levels = fibonacci_levels(low, high)
    support, resistance = find_support_resistance(historical_prices, window=12)  # 12 períodos de 1 hora = 0.5 días
    
    if "SHORT" in trade_direction:
        entry = high
        stop_loss = min(entry * 1.03, entry + (2 * atr))
        risk = stop_loss - entry
        tp1 = predicted_price  # Cambiado a precio predicho
        tp2 = max(entry - (2.5 * risk), fib_levels[0.618])
    else:  # LONG
        entry = low
        stop_loss = max(entry * 0.97, entry - (2 * atr))
        risk = entry - stop_loss
        tp1 = predicted_price  # Cambiado a precio predicho
        tp2 = min(entry + (2.5 * risk), fib_levels[1.618])
    
    return entry, tp1, tp2, stop_loss

def create_chart(symbol, hours=42, entry=None, tp1=None, tp2=None, stop_loss=None, trade_direction=None, interval='1h'):
    print(f"Iniciando creación del gráfico para {symbol} con intervalo {interval}")
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(hours=hours)
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
                             title=f'\n{symbol} Price Chart (Last {hours} Hours, {interval} interval)',
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
    chat_id = '-1002215332273'
    
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

def predictPrice(interval='1h'):  # Cambiado de '1h' a '1h'
    global stock_prices, SPY, y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE

    PERCENTAGE = 0.8
    WINDOW = 24
    PREDICTION_SCOPE = 24

    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=60)  # Cambiado de 365 a 60 días

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
        print(f"Precio actual obtenido: ${current_price:.2f}")

        predicted_price, prediction_date, prediction_hours, mae = predictPrice(interval)
        if predicted_price is None:
            raise ValueError(f"No se pudo obtener la predicción de precio para {symbol}. Revisa los mensajes de error anteriores.")
        print(f"Precio predicho: ${predicted_price:.2f} para {prediction_date}")
        print(f"MAE (Error Medio Absoluto): {mae:.2f}")

        trade_direction = determine_trade_direction(current_price, predicted_price)
        print(f"Dirección del trade: {trade_direction}")
        
        historical_prices = get_historical_prices(symbol, hours=168, interval=interval)  # 168 horas = 7 días
        print("Datos históricos obtenidos:")
        print(historical_prices)

        highest_volume_prices = get_highest_volume_prices(historical_prices, n=4)

        if historical_prices.empty:
            print("Advertencia: No se pudieron obtener datos históricos recientes. Usando valores alternativos.")
            highest_price_42h = max(current_price, predicted_price)
            lowest_price_42h = min(current_price, predicted_price)
            entry, tp1, tp2, stop_loss = current_price, predicted_price, predicted_price, current_price * (1.03 if "SHORT" in trade_direction else 0.97)
        else:
            highest_price_42h = historical_prices['High'].max()
            lowest_price_42h = historical_prices['Low'].min()
            entry, tp1, tp2, stop_loss = find_entry_and_targets(historical_prices, current_price, predicted_price, trade_direction)

        print(f"Niveles calculados: Entry=${entry:.2f}, TP1=${tp1:.2f}, TP2=${tp2:.2f}, SL=${stop_loss:.2f}")

        now = datetime.now(colombia_tz)
        specific_prices = {}
        for hours in range(4, 43, 4):
            price_key = f"price_{hours}h_ago"
            price_date = now - timedelta(hours=hours)
            if historical_prices.empty:
                specific_prices[price_key] = None
            else:
                period_prices = historical_prices[historical_prices.index <= price_date]
                if not period_prices.empty:
                    if "SHORT" in trade_direction:
                        specific_prices[price_key] = float(period_prices['High'].max())
                    else:
                        specific_prices[price_key] = float(period_prices['Low'].min())
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
            "highest_price_42h": float(highest_price_42h),
            "lowest_price_42h": float(lowest_price_42h),
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
            chart_path = create_chart(symbol, hours=42, entry=entry, tp1=tp1, tp2=tp2, stop_loss=stop_loss, trade_direction=trade_direction, interval=interval)
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

Predicción para las próximas {prediction_hours} horas:

- Precio Actual: ${prediction_data['current_price']:.2f}
- Precio Predicho: ${prediction_data['predicted_price']:.2f}
- Dirección del Trade: {prediction_data['trade_direction']}
- MAE (Error Medio Absoluto): {prediction_data['mae']:.2f}

Niveles de Trading (basados en datos de las últimas 42 horas):
- Precio de Entrada: ${prediction_data['entry_price']:.2f}
- Stop Loss: ${prediction_data['stop_loss']:.2f}
- Objetivo 1 (TP1): ${prediction_data['target_price_1']:.2f}
- Objetivo 2 (TP2): ${prediction_data['target_price_2']:.2f}

Rango de Precios (últimas 42 horas):
- Precio Más Alto: ${prediction_data['highest_price_42h']:.2f}
- Precio Más Bajo: ${prediction_data['lowest_price_42h']:.2f}

Precios con Mayor Volumen (últimas 42 horas):
{chr(10).join([f"- {{price['date']}}: ${{price['price']:.2f}} (Volumen: {{price['volume']:,.0f}})" for price in prediction_data['highest_volume_prices']])}

Generado el {current_date} a las {current_time}
"""

        print("Preparando envío a Telegram...")
        if chart_path and os.path.exists(chart_path):
            print(f"Archivo de gráfico encontrado en {chart_path}")
            send_to_telegram(message, chart_path)
        else:
            print(f"No se pudo encontrar el archivo del gráfico en {chart_path if chart_path else 'ninguna ubicación'}. Enviando solo el mensaje.")
            send_to_telegram(message, None)

        for hours in range(4, 43, 4):
            specific_date = now - timedelta(hours=hours)
            if not historical_prices.empty:
                print(f"\nDatos de hace {hours} horas ({specific_date}):")
                print(historical_prices.loc[historical_prices.index <= specific_date].tail(1))
            else:
                print(f"\nNo hay datos disponibles para hace {hours} horas ({specific_date})")

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
        # Agregar más información de depuración aquí
        print(f"Información adicional de depuración:")
        print(f"Símbolo: {symbol}")
        print(f"Intervalo: {interval}")
        print(f"Precio actual: {current_price if 'current_price' in locals() else 'No disponible'}")
        # Puedes agregar más variables relevantes aquí si es necesario

if __name__ == "__main__":
    main()