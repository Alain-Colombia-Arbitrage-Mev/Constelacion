import MetaTrader5 as mt5

class ConnectMt5:
    balance = 0

    def __init__(self):
        self.symbol = "EURUSD"

        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        else:
            account_info = mt5.account_info()
            # Comprobar si se obtuvo la información de la cuenta correctamente
            if account_info:
                self.balance = account_info.balance
        
    def get_balance(self):
        return self.balance 
    
    def run(self, Type='', size=0.01):
        # Extract filling_mode
        # filling_type = mt5.symbol_info(self.symbol).filling_mode
        # print('filling_type', filling_type)

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(self.symbol, "not found, can not call order_check()")
            mt5.shutdown()
            quit()
        # Establish connection to the MetaTrader 5 terminal
        
        # prepare the buy request structure
        # if the symbol is unavailable in MarketWatch, add it
        if not symbol_info.visible:
            print(self.symbol, "is not visible, trying to switch on")
            if not mt5.symbol_select(self.symbol,True):
                print("symbol_select({}}) failed, exit",self.symbol)
                mt5.shutdown()
                quit()
        
    
        lot = 0.01
        point = mt5.symbol_info(self.symbol).point
        price = mt5.symbol_info_tick(self.symbol).ask
        deviation = 20

        if Type == "buy":
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                
                "deviation": deviation,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        elif Type == "sell":
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                
                "volume": size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                
                "deviation": deviation,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

        
        # send a trading request
        result = mt5.order_send(request)
        # check the execution result
        print("1. order_send(): by {} {} lots at {} with deviation={} points".format(self.symbol,lot,price,deviation));
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("2. order_send failed, retcode={}".format(result.retcode))
            # request the result as a dictionary and display it element by element
            result_dict=result._asdict()
            for field in result_dict.keys():
                print("   {}={}".format(field,result_dict[field]))
                # if this is a trading request structure, display it element by element as well
                if field=="request":
                    traderequest_dict=result_dict[field]._asdict()
                    for tradereq_filed in traderequest_dict:
                        print("       traderequest: {}={}".format(tradereq_filed,traderequest_dict[tradereq_filed]))
            print("shutdown() and quit")
            mt5.shutdown()
            quit()
        
        # shut down connection to the MetaTrader 5 terminal
        mt5.shutdown()