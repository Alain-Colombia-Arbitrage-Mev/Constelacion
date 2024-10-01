from threading import Thread
from DEX.cli import exec_time
import itertools
from prettytable import PrettyTable
from DEX.Converter import Converter
from networkx import DiGraph, simple_cycles


class AdvancedScanner:
    def __init__(self, *exchanges, quote_asset: str, quote_amount):
        """
        @param exchanges: Exchange objects for scanning
        @param quote_asset: token symbol name, used for measuring volume or depth to scan
        @param quote_amount: amount of quote asset token
        """
        self.exchanges = {exchange.name: exchange for exchange in exchanges}
        self.converter = Converter(quote_asset, quote_amount)
        self.arbitrage_spreads = None

    def update_quote_asset_prices(self):
        """
        Set quote asset prices for all exchange objects
        """
        self.exchanges[list(self.exchanges)[0]].quote_asset_prices = self.converter.convert()

    # @exec_time
    def update_prices(self):
        """
        Create and run threads for update_price_book method
        for all passed in init exchanges
        """
        threads = []
        for exchange in self.exchanges.values():
            thread = Thread(target=exchange.update_price_book())
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    @exec_time
    def get_edges(self):
        """
        Finds possible connections between vertices
        Vertex has form  like "{exchange.name}_{pair_name}_{buy/sell}"
        return: list of edges
        """
        edges = []
        for exchange1, exchange2 in itertools.permutations(self.exchanges.values(), 2):

            for pricebook1 in exchange1.price_book:
                base_vertex1 = f'{exchange1.name}_{pricebook1}_buy'
                sec_vertex1 = None
                base_vertex2 = f'{exchange1.name}_{pricebook1}_sell'
                sec_vertex2 = None
                for pricebook2 in exchange2.price_book:
                    if pricebook1.split('-')[0] == pricebook2.split('-')[0]:
                        sec_vertex1 = f'{exchange2.name}_{pricebook2}_sell'
                    elif pricebook1.split('-')[0] == pricebook2.split('-')[1]:
                        sec_vertex1 = f'{exchange2.name}_{pricebook2}_buy'

                    if pricebook1.split('-')[1] == pricebook2.split('-')[1]:
                        sec_vertex2 = f'{exchange2.name}_{pricebook2}_buy'

                    elif pricebook1.split('-')[1] == pricebook2.split('-')[0]:
                        sec_vertex2 = f'{exchange2.name}_{pricebook2}_sell'

                    if sec_vertex1:
                        edges.append((base_vertex1, sec_vertex1))
                    if sec_vertex2:
                        edges.append((base_vertex2, sec_vertex2))
        return edges

    def scan(self, spread_threshold=-0.2, max_path_length=4):
        """
        @param max_path_length: Max amount of steps in arbitrage
        @param spread_threshold: Minimum potential income
        to show arbitrage opportunity in %
        """
        # Updates quote_asset_prices for all exchanges
        self.update_quote_asset_prices()
        # Updates pricebook of every exchange
        self.update_prices()

        self.arbitrage_spreads = []
        # Creates graph with edges from get_edges method
        my_graph = DiGraph(self.get_edges())

        # Finds possible arbitrage routes and calculate potential profit
        cycles = sorted(filter(lambda x: len(x) > 1, simple_cycles(my_graph, max_path_length)))
        for cycle in cycles:
            spread = self.calculate_path_income(cycle)
            if spread[1] >= spread_threshold:
                self.arbitrage_spreads.append(spread)

        # prints arbitrage table
        self.print_arbitrage_table()

    @exec_time
    def print_arbitrage_table(self):
        """
        Print on a display arbitrage table with 2 columns
        1) PATH - arbitrage steps
        2) Possible profit in %
        """
        arbitrage_table = PrettyTable()
        arbitrage_table.field_names = ['PATH', 'Profit %']
        arbitrage_table.sortby = 'Profit %'
        arbitrage_table.max_table_width = 800
        arbitrage_table.reversesort = True

        for spread in self.arbitrage_spreads:
            arbitrage_table.add_row([' -> '.join(spread[0].keys()), spread[1]], divider=True)

        print(arbitrage_table)

    def calculate_path_income(self, path):
        """
        Calculates profit of a certain arbitrage path
        :param path: List of arbitrage steps
        :return: path_preview like "SushiSwapV3/500_WMATIC-WETH_sell -> UniswapV3/500_WMATIC-WETH_buy",
        and profit in percents
        """
        initial_amount = None
        amount_in = None
        path_preview = {}
        amount_out = 0
        for index, step in enumerate(path):

            exchange_name, pair, action = step.split('_')
            exchange = self.exchanges[exchange_name]
            step_price = exchange.price_book[pair][f'{action}_price']

            if action == 'buy':
                if index == 0:
                    initial_amount = exchange.price_book[pair][f'{action}_amount'] * step_price
                    amount_in = initial_amount
                amount_out = amount_in / step_price
            else:
                if index == 0:
                    amount_in = exchange.price_book[pair][f'{action}_amount']
                    initial_amount = amount_in
                amount_out = amount_in * step_price
            path_preview[step] = [step_price, amount_in]
            amount_in = amount_out

        profit = (amount_out - initial_amount) / initial_amount * 100

        return path_preview, profit


if __name__ == "__main__":
    from DEX.UniswapV3 import UniswapV3
    from DEX.SushiSwapV2 import SushiSwapV2
    from DEX.SushiSwapV3 import SushiSwapV3
    import os
    from dotenv import load_dotenv
    import time

    load_dotenv()
    net = "Polygon"
    subnet = "MAINNET"
    web3_provider = os.environ['INFURA_POLYGON']

    uniswap_v3_pools_3000 = ['WMATIC-WETH', 'WETH-USDC', 'WBTC-WETH', 'WMATIC-USDC', 'LINK-WETH', 'WETH-USDT']
    uniswap_v3_pools_500 = ['WMATIC-WETH', 'WETH-USDC', 'WBTC-WETH', 'WMATIC-USDC', 'LINK-WETH', 'WETH-USDT']
    sushi3_pools_3000 = ['WMATIC-WETH', 'WETH-USDC', 'WBTC-WETH', 'WMATIC-USDC', 'LINK-WETH', 'WETH-USDT']
    sushi3_pools_500 = ['WMATIC-WETH', 'WETH-USDC', 'WBTC-WETH', 'WMATIC-USDC', 'LINK-WETH', 'WETH-USDT']
    sushi2_pairs = ['WMATIC-WETH', 'WETH-USDC', 'WBTC-WETH', 'WMATIC-USDC', 'LINK-WETH', 'WETH-USDT']

    uniswapV3_3000 = UniswapV3(net, subnet, web3_provider, 3000, uniswap_v3_pools_3000)
    uniswapV3_500 = UniswapV3(net, subnet, web3_provider, 500, uniswap_v3_pools_500)
    sushi3_3000 = SushiSwapV3(net, subnet, web3_provider, 3000, sushi3_pools_3000)
    sushi3_500 = SushiSwapV3(net, subnet, web3_provider, 500, sushi3_pools_500)
    sushi2 = SushiSwapV2(net, subnet, web3_provider, sushi2_pairs)

    scanner = AdvancedScanner(uniswapV3_500, uniswapV3_3000, sushi2, sushi3_500, sushi3_3000,
                              quote_asset='USDC', quote_amount=100)
    while True:
        scanner.scan(spread_threshold=-0.4, max_path_length=4)
        time.sleep(10)