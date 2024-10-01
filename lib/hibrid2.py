import sys
sys.path.insert(0, './lib')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xgboost as xgbs
from xgboost import plot_importance, plot_tree
import yfinance as yf
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import warnings
import json
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import telegram

#resolver comunicación con telegram 


warnings.filterwarnings("ignore")

def feature_engineering(data, SPY, predictions=np.array([None]))->pd.core.frame.DataFrame:
    assert type(data) == pd.core.frame.DataFrame, "data must be a dataframe"
    assert type(SPY) == pd.core.series.Series, "SPY must be a dataframe"
    assert type(predictions) == np.ndarray, "predictions must be an array"

    if predictions.any() == True:
        data = yf.download("AVAX-USD", start="2001-11-30")
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
        plt.ylabel("AVAX-USD stock price")
        plt.title(f"The MAE for this period is: {round(mae, 3)}")

    return mae, model

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

def get_historical_prices(symbol, days=4):
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date, end=end_date)
    return data[['Close', 'High', 'Low']]

def determine_trade_direction(current_price, predicted_price):
    return "LONG" if predicted_price > current_price else "SHORT"

def find_suggested_entry_price(historical_prices, trade_direction):
    if trade_direction == "SHORT":
        return historical_prices['High'].max()
    else:
        return historical_prices['Low'].min()

def predictPrice():
    global stock_prices, SPY, y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE

    PERCENTAGE = 0.995
    WINDOW = 2
    PREDICTION_SCOPE = 0

    stock_prices = yf.download("AVAX-USD")
    SPY = yf.download("SPY")["Close"]

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
    return plotting(y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE)

def main():
    symbol = "AVAX-USD"
    
    current_price = get_current_price(symbol)
    predicted_price, prediction_date, prediction_days = predictPrice()
    trade_direction = determine_trade_direction(current_price, predicted_price)
    
    historical_prices = get_historical_prices(symbol, days=4)
    suggested_entry_price = find_suggested_entry_price(historical_prices, trade_direction)
    
    now = datetime.now(pytz.UTC)
    specific_prices = {}
    for days in range(1, 5):
        price_key = f"price_{days}d_ago"
        price_date = now.date() - timedelta(days=days)
        day_prices = historical_prices[historical_prices.index.date == price_date]
        if trade_direction == "SHORT":
            specific_prices[price_key] = day_prices['High'].max() if not day_prices.empty else None
        else:
            specific_prices[price_key] = day_prices['Low'].min() if not day_prices.empty else None
    
    prediction_data = {
        "precio actual": float(current_price),
        "precio prediccion": float(predicted_price),
        "prediction_date": datetime.now().strftime('%Y-%m-%d'),
        "prediction_days": int(prediction_days),
        "direcion del trade": trade_direction,
        "entrada sugerida": float(suggested_entry_price) if suggested_entry_price is not None else None,
        "highest_price_4d": float(historical_prices['High'].max()),
        "lowest_price_4d": float(historical_prices['Low'].min()),
        "token": symbol,
        **{k: float(v) if v is not None else None for k, v in specific_prices.items()}
    }
    
    with open('../data/prediction-avax.json', 'w') as json_file:
        json.dump(prediction_data, json_file, indent=4)
    
    print(json.dumps(prediction_data, indent=4))
    print("\nHistorical prices:")
    print(historical_prices)
    
    # Imprimir datos específicos de cada día
    for days in range(4):
        specific_date = now.date() - timedelta(days=days)
        print(f"\nDatos de hace {days} días ({specific_date}):")
        print(historical_prices.loc[historical_prices.index.date == specific_date])

if __name__ == "__main__":
    main()