import sys
sys.path.insert(0, './lib')

import datetime
from connect import ConnectMt5
from hibrid import predictPrice
import yfinance as yf
import pandas as pd

def previous_close():
    apple = yf.Ticker("AAPL")
    historical_data = apple.history(period="2d")  # Pedimos 2 días para asegurarnos de obtener el cierre del día anterior
    # Obtener el cierre del día anterior
    previous_close = historical_data.iloc[-2]['Close']
    return previous_close

def CalculateDirection(predictprice):
    if predictprice > previous_close():
        return 'long'
    elif predictprice < previous_close():
        return 'short'

_predictprice, datePredicted, days  = predictPrice()
direction = CalculateDirection(_predictprice)
print("Cierre del día anterior de Apple:", previous_close())
print("Predicted Price:", _predictprice)
print('Direction is: ', CalculateDirection(_predictprice))
# sys.exit()

# from hibridstrategy import NyHibridStrategy

class RiskManagment:

    def __init__(self, balance):
        self.balance = balance
        pass 

    def set_risk(self, risk):
        self.risk = self.balance * (float(risk)/100) 
        return 
    
    def get_risk(self):
        return self.risk
    
    def calculateSize(self, sl):
        risk = self.get_risk() #self.balance * risk_percentage 
        valpip = risk * 1 / sl
        print('valpip',valpip, 'risk', risk)
        contractSize = 100.000
    
        lotSize = valpip / 0.10
   
        return round(lotSize / contractSize, 2)

# 4 => Viernes
launching = [
    [1, "01:47"], # Prelondon,
    [4, "02:56"], # London  
    [4, "08:02"] # Ny
] 
    
ny_open_time = datetime.datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
print(ny_open_time)

# Load NyStrategy
# strategy = NyHibridStrategy()
# print(strategy.setUp(df_sorted))

# for i in range(len(df_sorted)):
#     print(strategy.checkLongSignal(i))

class Broker:
    
    def __init__(self, connect):
        self.connect = connect 

    def Execute(self, type = '', size=0 ): 
        self.connect.run(Type=type, size=size)

connect = ConnectMt5()
riskMgt = RiskManagment(connect.get_balance())
_risk = riskMgt.set_risk(1) # 1%
size =  riskMgt.calculateSize(sl=10)


print('Balance:', connect.get_balance(), 
      'calculated risk >>', riskMgt.get_risk(),
      'Size:', size)


# Detener el proceso
# sys.exit()


broker = Broker(connect)


while True:
    current_time = [datetime.datetime.now().weekday(), datetime.datetime.now().strftime("%H:%M")] #%S"
    print("current_time", current_time, 'balance', connect.get_balance())
    if current_time in launching:
        if direction == 'short':
            broker.Execute('sell',  size=size)
        elif direction == 'long':
            broker.Execute('buy', size)
        istime = True
        print("its time!", datetime.datetime.now().strftime("%H:%M:%S"))
    else: 
        istime = False
        print("is not time yet")