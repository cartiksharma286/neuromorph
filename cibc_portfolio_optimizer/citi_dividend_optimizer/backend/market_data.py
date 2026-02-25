"""
Market Data Generator for Citi Dividend App.
Simulates high-quality dividend stock data.
"""

import numpy as np
import datetime

class MarketData:
    def __init__(self):
        self.stocks = [
            {'symbol': 'JPM', 'name': 'JPMorgan Chase', 'sector': 'Financials', 'base_price': 170, 'div_yield': 2.8},
            {'symbol': 'C', 'name': 'Citigroup Inc.', 'sector': 'Financials', 'base_price': 55, 'div_yield': 3.9},
            {'symbol': 'XOM', 'name': 'Exxon Mobil', 'sector': 'Energy', 'base_price': 105, 'div_yield': 3.5},
            {'symbol': 'CVX', 'name': 'Chevron', 'sector': 'Energy', 'base_price': 150, 'div_yield': 4.0},
            {'symbol': 'PG', 'name': 'Procter & Gamble', 'sector': 'Consumer', 'base_price': 155, 'div_yield': 2.4},
            {'symbol': 'KO', 'name': 'Coca-Cola', 'sector': 'Consumer', 'base_price': 60, 'div_yield': 3.1},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'base_price': 160, 'div_yield': 2.9},
            {'symbol': 'PFE', 'name': 'Pfizer', 'sector': 'Healthcare', 'base_price': 28, 'div_yield': 5.8},
            {'symbol': 'VZ', 'name': 'Verizon', 'sector': 'Comms', 'base_price': 40, 'div_yield': 6.5},
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology', 'base_price': 185, 'div_yield': 0.5},
            {'symbol': 'MSFT', 'name': 'Microsoft', 'sector': 'Technology', 'base_price': 400, 'div_yield': 0.7},
            {'symbol': 'NEE', 'name': 'NextEra Energy', 'sector': 'Utilities', 'base_price': 60, 'div_yield': 3.2}
        ]
        
    def get_universe(self):
        return self.stocks
        
    def generate_history(self, days=252):
        """Simulate price history for statistical learning."""
        history = {}
        for stock in self.stocks:
            prices = [stock['base_price']]
            for _ in range(days):
                change = np.random.normal(0.0005, 0.015) # Slight upward drift
                prices.append(prices[-1] * (1 + change))
            history[stock['symbol']] = np.array(prices)
        return history

    def get_market_context(self):
        return {
            "sp500_trend": "Bullish",
            "vix": 14.5,
            "10y_treasury": 4.1
        }
