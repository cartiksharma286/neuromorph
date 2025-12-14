"""
Optimal Spread Calculator for Forex & Equity Trading
----------------------------------------------------
Calculates optimal bid-ask spreads using market microstructure models
(Avellaneda-Stoikov approximation) and volatility analysis.
"""

import numpy as np
from typing import Dict, List

class SpreadOptimizer:
    def __init__(self):
        # Base liquidity parameters (simulated inverse liquidity)
        self.liquidity_factors = {
            'Financials': 0.0001,
            'Energy': 0.0002,
            'Tech': 0.0003, # Higher spread for tech possibly due to vol
            'Forex_Major': 0.00005,
            'Forex_Minor': 0.00015
        }
        
        self.forex_pairs = [
            {'symbol': 'EUR/USD', 'price': 1.10, 'volatility': 0.005, 'type': 'Forex_Major'},
            {'symbol': 'USD/CAD', 'price': 1.35, 'volatility': 0.006, 'type': 'Forex_Major'},
            {'symbol': 'GBP/USD', 'price': 1.25, 'volatility': 0.007, 'type': 'Forex_Major'},
            {'symbol': 'USD/JPY', 'price': 145.0, 'volatility': 0.008, 'type': 'Forex_Major'},
            {'symbol': 'AUD/CAD', 'price': 0.90, 'volatility': 0.009, 'type': 'Forex_Minor'}
        ]

    def calculate_optimal_spread(self, price: float, volatility: float, liquidity_param: float, risk_aversion: float = 0.1) -> Dict:
        """
        Calculate optimal spread using a simplified market maker model.
        Spread = (2/gamma) * ln(1 + gamma/k)  <-- classic theoretical form
        Here we use a practical approximation: 
        Spread ~ alpha * Volatility + beta * (1/Liquidity)
        """
        # Model: Spread = Risk_Premium + Liquidity_Cost
        # Risk_Premium = Risk_Aversion * Volatility^2 * Price
        # Liquidity_Cost = Base_Cost * Impact
        
        half_spread = (risk_aversion * (volatility**2) * price) + (liquidity_param * price)
        
        # Ensure minimum spread (1 pip for forex, 1 cent for stocks approx)
        min_spread = 0.0001 if price < 10 else 0.01
        total_spread = max(2 * half_spread, min_spread)
        
        bid = price - (total_spread / 2)
        ask = price + (total_spread / 2)
        
        return {
            'bid': round(bid, 4),
            'ask': round(ask, 4),
            'spread': round(total_spread, 4),
            'spread_bps': round((total_spread / price) * 10000, 2)
        }

    def generate_spreads(self, stocks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Generate optimal spreads for both Equity (stocks input) and internal Forex list.
        """
        results = {
            'Equities': [],
            'Forex': []
        }
        
        # Process Equities
        for stock in stocks:
            sector = stock.get('sector', 'Other')
            k_liq = self.liquidity_factors.get(sector, 0.0002)
            
            # Estimate daily vol from annual or generic
            vol = 0.02 # default daily vol estimate (2%)
            
            spread_data = self.calculate_optimal_spread(stock['price'], vol, k_liq)
            results['Equities'].append({
                'symbol': stock['symbol'],
                'price': stock['price'],
                **spread_data
            })
            
        # Process Forex
        for fx in self.forex_pairs:
            k_liq = self.liquidity_factors.get(fx['type'], 0.0001)
            # Forex vol is usually lower per day
            vol = fx['volatility']
            
            spread_data = self.calculate_optimal_spread(fx['price'], vol, k_liq)
            results['Forex'].append({
                'symbol': fx['symbol'],
                'price': fx['price'],
                **spread_data
            })
            
        return results
