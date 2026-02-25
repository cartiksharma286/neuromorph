"""
CIBC Market Data Generator
Simulates Canadian dividend stock universe with realistic data
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import random


class MarketDataGenerator:
    """Generate realistic market data for Canadian dividend stocks"""
    
    # TSX Blue-chip dividend stocks by sector
    STOCK_UNIVERSE = {
        'Financials': [
            {'symbol': 'RY.TO', 'name': 'Royal Bank of Canada', 'base_yield': 4.2},
            {'symbol': 'TD.TO', 'name': 'TD Bank', 'base_yield': 4.8},
            {'symbol': 'BNS.TO', 'name': 'Bank of Nova Scotia', 'base_yield': 5.5},
            {'symbol': 'BMO.TO', 'name': 'Bank of Montreal', 'base_yield': 4.9},
            {'symbol': 'CM.TO', 'name': 'CIBC', 'base_yield': 5.2},
            {'symbol': 'MFC.TO', 'name': 'Manulife Financial', 'base_yield': 5.0},
            {'symbol': 'SLF.TO', 'name': 'Sun Life Financial', 'base_yield': 4.3}
        ],
        'Utilities': [
            {'symbol': 'FTS.TO', 'name': 'Fortis Inc', 'base_yield': 4.0},
            {'symbol': 'EMA.TO', 'name': 'Emera Inc', 'base_yield': 5.8},
            {'symbol': 'AQN.TO', 'name': 'Algonquin Power', 'base_yield': 6.5},
            {'symbol': 'CU.TO', 'name': 'Canadian Utilities', 'base_yield': 5.2}
        ],
        'Energy': [
            {'symbol': 'ENB.TO', 'name': 'Enbridge Inc', 'base_yield': 7.2},
            {'symbol': 'TRP.TO', 'name': 'TC Energy', 'base_yield': 7.5},
            {'symbol': 'CNQ.TO', 'name': 'Canadian Natural Resources', 'base_yield': 4.8},
            {'symbol': 'SU.TO', 'name': 'Suncor Energy', 'base_yield': 4.5}
        ],
        'Telecom': [
            {'symbol': 'BCE.TO', 'name': 'BCE Inc', 'base_yield': 8.5},
            {'symbol': 'T.TO', 'name': 'TELUS Corp', 'base_yield': 6.8},
            {'symbol': 'RCI.B.TO', 'name': 'Rogers Communications', 'base_yield': 3.5}
        ],
        'REITs': [
            {'symbol': 'REI.UN.TO', 'name': 'RioCan REIT', 'base_yield': 6.2},
            {'symbol': 'HR.UN.TO', 'name': 'H&R REIT', 'base_yield': 7.0}
        ],
        'Flash Gemini Tech': [
            {'symbol': 'NVDA', 'name': 'NVIDIA Corp', 'base_yield': 0.03},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc', 'base_yield': 0.0},
            {'symbol': 'AAPL', 'name': 'Apple Inc', 'base_yield': 0.5}
        ],
        'NASDAQ 100': [
            {'symbol': 'MSFT', 'name': 'Microsoft Corp', 'base_yield': 0.8},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc', 'base_yield': 0.0},
            {'symbol': 'META', 'name': 'Meta Platforms', 'base_yield': 0.0},
            {'symbol': 'TSLA', 'name': 'Tesla Inc', 'base_yield': 0.0},
            {'symbol': 'NFLX', 'name': 'Netflix Inc', 'base_yield': 0.0},
            {'symbol': 'AVGO', 'name': 'Broadcom Inc', 'base_yield': 1.2},
            {'symbol': 'CSCO', 'name': 'Cisco Systems', 'base_yield': 3.1},
            {'symbol': 'PEP', 'name': 'PepsiCo Inc', 'base_yield': 2.7},
            {'symbol': 'COST', 'name': 'Costco Wholesale', 'base_yield': 0.6},
            {'symbol': 'TMUS', 'name': 'T-Mobile US', 'base_yield': 0.0}
        ]
    }
    
    def __init__(self, seed: int = 42):
        """Initialize market data generator with random seed"""
        np.random.seed(seed)
        random.seed(seed)
        self.stocks = self._initialize_stocks()
        
    def _initialize_stocks(self) -> List[Dict]:
        """Initialize stock data with realistic parameters"""
        stocks = []
        
        for sector, sector_stocks in self.STOCK_UNIVERSE.items():
            for stock_info in sector_stocks:
                # Generate realistic stock data
                base_price = np.random.uniform(20, 150)
                base_yield = 35.0  # Bumped up to 35%
                annual_dividend = base_price * (base_yield / 100)
                
                # Historical dividend growth
                dividend_history = self._generate_dividend_history(annual_dividend)
                
                # Fundamental metrics
                eps = annual_dividend / np.random.uniform(0.4, 0.7)  # Payout ratio 40-70%
                pe_ratio = np.random.uniform(10, 20)
                pb_ratio = np.random.uniform(1.2, 3.0)
                roe = np.random.uniform(10, 18)
                debt_to_equity = np.random.uniform(0.5, 2.0)
                
                # Dividend growth metrics
                consecutive_increases = np.random.randint(3, 25)
                
                stock = {
                    'symbol': stock_info['symbol'],
                    'name': stock_info['name'],
                    'sector': sector,
                    'price': round(base_price, 2),
                    'annual_dividend': round(annual_dividend, 2),
                    'quarterly_dividend': round(annual_dividend / 4, 2),
                    'dividend_yield': round(base_yield, 2),
                    'dividend_history': dividend_history,
                    'eps': round(eps, 2),
                    'pe_ratio': round(pe_ratio, 2),
                    'pb_ratio': round(pb_ratio, 2),
                    'roe': round(roe, 2),
                    'debt_to_equity': round(debt_to_equity, 2),
                    'consecutive_increases': consecutive_increases,
                    'payment_frequency': 'quarterly',
                    'next_ex_date': self._generate_next_ex_date(),
                    'is_eligible_dividend': True,  # Most Canadian corps pay eligible dividends
                    'fcf_coverage': round(np.random.uniform(1.2, 2.5), 2),
                    'market_cap': round(base_price * np.random.uniform(10, 100) * 1e9, 0),
                    'profit_margin': round(np.random.uniform(0.05, 0.25), 2),
                    'ma_200': round(base_price * np.random.uniform(0.9, 1.1), 2),
                    'rsi': int(np.random.uniform(30, 80))
                }
                
                stocks.append(stock)
        
        return stocks
    
    def _generate_dividend_history(self, current_dividend: float, years: int = 10) -> List[float]:
        """Generate historical dividend data with realistic growth"""
        history = []
        dividend = current_dividend
        
        # Work backwards from current dividend
        for year in range(years):
            # Random growth rate between 3% and 8% per year
            growth_rate = np.random.uniform(0.03, 0.08)
            dividend = dividend / (1 + growth_rate)
            history.insert(0, round(dividend, 2))
        
        history.append(current_dividend)
        return history
    
    def _generate_next_ex_date(self) -> str:
        """Generate next ex-dividend date"""
        today = datetime.now()
        # Random date in next 90 days
        days_ahead = np.random.randint(1, 90)
        next_date = today + timedelta(days=days_ahead)
        return next_date.strftime('%Y-%m-%d')
    
    def get_all_stocks(self) -> List[Dict]:
        """Get all stocks in universe"""
        return self.stocks
    
    def get_stocks_by_sector(self, sector: str) -> List[Dict]:
        """Get stocks filtered by sector"""
        return [s for s in self.stocks if s['sector'] == sector]
    
    def get_stock_by_symbol(self, symbol: str) -> Dict:
        """Get specific stock by symbol"""
        for stock in self.stocks:
            if stock['symbol'] == symbol:
                return stock
        return {}
    
    def generate_covariance_matrix(self) -> np.ndarray:
        """Generate realistic covariance matrix for stocks"""
        n = len(self.stocks)
        
        # Base volatilities by sector
        sector_volatility = {
            'Financials': 0.20,
            'Utilities': 0.15,
            'Energy': 0.30,
            'Telecom': 0.18,
            'Telecom': 0.18,
            'REITs': 0.22,
            'Flash Gemini Tech': 0.35,
            'NASDAQ 100': 0.25
        }
        
        # Create correlation matrix
        correlation = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Higher correlation within same sector
                if self.stocks[i]['sector'] == self.stocks[j]['sector']:
                    corr = np.random.uniform(0.5, 0.8)
                else:
                    corr = np.random.uniform(0.1, 0.4)
                
                correlation[i, j] = corr
                correlation[j, i] = corr
        
        # Create volatility vector
        volatilities = np.array([
            sector_volatility[stock['sector']] for stock in self.stocks
        ])
        
        # Covariance = correlation * outer(volatilities)
        covariance = correlation * np.outer(volatilities, volatilities)
        
        # Ensure positive semi-definite
        min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
        if min_eig < 0:
            covariance -= 10 * min_eig * np.eye(*covariance.shape)
            
        return covariance
    
    def generate_expected_returns(self) -> np.ndarray:
        """Generate expected returns based on dividend yield + growth"""
        returns = []
        
        for stock in self.stocks:
            # Expected return = dividend yield + dividend growth + price appreciation
            dividend_yield = stock['dividend_yield'] / 100
            
            # Estimate growth from history
            if len(stock['dividend_history']) >= 5 and stock['dividend_history'][-5] > 0:
                recent_growth = (stock['dividend_history'][-1] / stock['dividend_history'][-5]) ** 0.2 - 1
            else:
                # For zero-dividend stocks (Flash Gemini Tech), assume high growth
                recent_growth = 0.15 if stock['sector'] == 'Flash Gemini Tech' else 0.05
            
            # Price appreciation (typically lower for high-yield stocks)
            price_growth = np.random.uniform(0.02, 0.06)
            
            total_return = dividend_yield + recent_growth + price_growth
            returns.append(total_return)
        
        return np.array(returns)
    
    def get_dividend_yields(self) -> np.ndarray:
        """Get array of dividend yields"""
        return np.array([stock['dividend_yield'] for stock in self.stocks])
    
    def get_sector_mapping(self) -> Dict[str, List[int]]:
        """Get mapping of sectors to stock indices"""
        sector_map = {}
        
        for idx, stock in enumerate(self.stocks):
            sector = stock['sector']
            if sector not in sector_map:
                sector_map[sector] = []
            sector_map[sector].append(idx)
        
        return sector_map
    
    def update_prices(self, volatility_factor: float = 1.0):
        """Simulate price updates (for real-time demo)"""
        for stock in self.stocks:
            # Random price movement
            change_pct = np.random.normal(0, 0.01 * volatility_factor)
            stock['price'] *= (1 + change_pct)
            stock['price'] = round(stock['price'], 2)
            
            # Update yield
            stock['dividend_yield'] = round(
                (stock['annual_dividend'] / stock['price']) * 100, 2
            )
    
    def get_market_summary(self) -> Dict:
        """Get overall market summary statistics"""
        yields = [s['dividend_yield'] for s in self.stocks]
        pe_ratios = [s['pe_ratio'] for s in self.stocks]
        
        return {
            'total_stocks': len(self.stocks),
            'sectors': list(self.STOCK_UNIVERSE.keys()),
            'avg_dividend_yield': round(np.mean(yields), 2),
            'median_dividend_yield': round(np.median(yields), 2),
            'avg_pe_ratio': round(np.mean(pe_ratios), 2),
            'highest_yield_stock': max(self.stocks, key=lambda x: x['dividend_yield'])['symbol'],
            'highest_yield': round(max(yields), 2)
        }
    
    def generate_historical_returns(self, days: int = 252) -> np.ndarray:
        """
        Generate simulated historical returns for ML covariance estimation
        
        Args:
            days: Number of trading days to simulate (default 252 = 1 year)
        
        Returns:
            Array of shape (days, num_stocks) with daily returns
        """
        n_stocks = len(self.stocks)
        cov_matrix = self.generate_covariance_matrix()
        expected_returns = self.generate_expected_returns()
        
        # Convert annual to daily
        daily_returns = expected_returns / 252
        daily_cov = cov_matrix / 252
        
        # Generate multivariate normal returns
        returns = np.random.multivariate_normal(
            daily_returns, 
            daily_cov, 
            size=days
        )
        
        return returns

