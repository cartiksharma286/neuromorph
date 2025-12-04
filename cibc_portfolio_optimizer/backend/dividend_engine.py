"""
CIBC Dividend Engine
Handles dividend calculations, forecasting, and Canadian tax credit analysis
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json


class DividendEngine:
    """Comprehensive dividend analysis and forecasting engine"""
    
    # Canadian dividend tax credit rates (2024)
    ELIGIBLE_DIVIDEND_GROSS_UP = 1.38
    ELIGIBLE_DIVIDEND_TAX_CREDIT = 0.150198
    NON_ELIGIBLE_GROSS_UP = 1.15
    NON_ELIGIBLE_TAX_CREDIT = 0.090301
    
    def __init__(self):
        self.dividend_history = {}
        
    def calculate_dividend_yield(self, 
                                 annual_dividend: float, 
                                 current_price: float) -> float:
        """Calculate current dividend yield"""
        if current_price <= 0:
            return 0.0
        return (annual_dividend / current_price) * 100
    
    def calculate_dividend_growth_rate(self, 
                                       dividend_history: List[float], 
                                       years: int = 5) -> float:
        """
        Calculate compound annual growth rate (CAGR) of dividends
        
        Args:
            dividend_history: List of annual dividends (oldest to newest)
            years: Number of years to calculate CAGR over
        
        Returns:
            CAGR as percentage
        """
        if len(dividend_history) < 2 or years < 1:
            return 0.0
        
        # Use last 'years' of data
        data = dividend_history[-min(years + 1, len(dividend_history)):]
        
        if len(data) < 2 or data[0] <= 0:
            return 0.0
        
        starting_dividend = data[0]
        ending_dividend = data[-1]
        num_years = len(data) - 1
        
        # CAGR formula: (Ending/Beginning)^(1/years) - 1
        cagr = (ending_dividend / starting_dividend) ** (1 / num_years) - 1
        return cagr * 100
    
    def calculate_payout_ratio(self, 
                               annual_dividend: float, 
                               earnings_per_share: float) -> float:
        """
        Calculate dividend payout ratio
        
        Args:
            annual_dividend: Annual dividend per share
            earnings_per_share: Annual earnings per share
        
        Returns:
            Payout ratio as percentage
        """
        if earnings_per_share <= 0:
            return 100.0  # Unsustainable if no earnings
        
        payout_ratio = (annual_dividend / earnings_per_share) * 100
        return min(payout_ratio, 100.0)
    
    def calculate_sustainability_score(self, 
                                       payout_ratio: float,
                                       dividend_growth_5yr: float,
                                       free_cash_flow_coverage: float = 1.5) -> Dict:
        """
        Calculate dividend sustainability score
        
        Args:
            payout_ratio: Current payout ratio (%)
            dividend_growth_5yr: 5-year dividend growth rate (%)
            free_cash_flow_coverage: FCF / Dividends ratio
        
        Returns:
            Dictionary with score and rating
        """
        score = 100.0
        
        # Payout ratio scoring (lower is better, up to a point)
        if payout_ratio < 30:
            score -= 10  # Too low might indicate underutilization
        elif payout_ratio > 80:
            score -= 30  # High risk of dividend cut
        elif payout_ratio > 60:
            score -= 15
        
        # Growth scoring
        if dividend_growth_5yr < 0:
            score -= 40  # Dividend cuts are very negative
        elif dividend_growth_5yr < 3:
            score -= 20  # Below inflation
        elif dividend_growth_5yr > 10:
            score += 10  # Strong growth
        
        # Cash flow coverage
        if free_cash_flow_coverage < 1.0:
            score -= 30  # Not covered by FCF
        elif free_cash_flow_coverage > 2.0:
            score += 10  # Strong coverage
        
        # Determine rating
        if score >= 80:
            rating = "Excellent"
        elif score >= 65:
            rating = "Good"
        elif score >= 50:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            'score': max(0, min(100, score)),
            'rating': rating,
            'payout_ratio': payout_ratio,
            'growth_rate': dividend_growth_5yr,
            'fcf_coverage': free_cash_flow_coverage
        }
    
    def is_dividend_aristocrat(self, 
                               consecutive_increases: int,
                               market: str = 'TSX') -> bool:
        """
        Check if stock qualifies as dividend aristocrat
        
        TSX: 5+ years of consecutive increases
        S&P 500: 25+ years of consecutive increases
        """
        if market == 'TSX':
            return consecutive_increases >= 5
        elif market == 'SP500':
            return consecutive_increases >= 25
        return False
    
    def calculate_canadian_tax_credit(self,
                                     dividend_amount: float,
                                     is_eligible: bool = True,
                                     marginal_tax_rate: float = 0.50) -> Dict:
        """
        Calculate Canadian dividend tax credit
        
        Args:
            dividend_amount: Dividend received
            is_eligible: Whether dividend is eligible for enhanced credit
            marginal_tax_rate: Investor's marginal tax rate
        
        Returns:
            Dictionary with gross-up, taxable amount, and after-tax dividend
        """
        if is_eligible:
            gross_up = dividend_amount * self.ELIGIBLE_DIVIDEND_GROSS_UP
            taxable_amount = gross_up
            federal_credit = gross_up * self.ELIGIBLE_DIVIDEND_TAX_CREDIT
        else:
            gross_up = dividend_amount * self.NON_ELIGIBLE_GROSS_UP
            taxable_amount = gross_up
            federal_credit = gross_up * self.NON_ELIGIBLE_TAX_CREDIT
        
        # Calculate tax
        tax_before_credit = taxable_amount * marginal_tax_rate
        tax_after_credit = max(0, tax_before_credit - federal_credit)
        after_tax_dividend = dividend_amount - tax_after_credit
        
        # Effective tax rate
        effective_rate = (tax_after_credit / dividend_amount * 100) if dividend_amount > 0 else 0
        
        return {
            'dividend_received': dividend_amount,
            'gross_up_amount': gross_up - dividend_amount,
            'taxable_amount': taxable_amount,
            'tax_credit': federal_credit,
            'tax_payable': tax_after_credit,
            'after_tax_dividend': after_tax_dividend,
            'effective_tax_rate': effective_rate,
            'dividend_type': 'Eligible' if is_eligible else 'Non-Eligible'
        }
    
    def generate_dividend_calendar(self,
                                   holdings: List[Dict],
                                   months_ahead: int = 12) -> List[Dict]:
        """
        Generate dividend payment calendar
        
        Args:
            holdings: List of stock holdings with dividend info
            months_ahead: Number of months to project
        
        Returns:
            List of upcoming dividend payments
        """
        calendar = []
        today = datetime.now()
        
        for holding in holdings:
            symbol = holding.get('symbol', '')
            shares = holding.get('shares', 0)
            quarterly_dividend = holding.get('quarterly_dividend', 0)
            next_ex_date = holding.get('next_ex_date')
            payment_frequency = holding.get('payment_frequency', 'quarterly')
            
            if not next_ex_date:
                continue
            
            # Parse ex-dividend date
            if isinstance(next_ex_date, str):
                next_ex_date = datetime.strptime(next_ex_date, '%Y-%m-%d')
            
            # Generate future payments
            current_date = next_ex_date
            end_date = today + timedelta(days=months_ahead * 30)
            
            # Determine payment interval
            if payment_frequency == 'monthly':
                interval_days = 30
            elif payment_frequency == 'quarterly':
                interval_days = 91
            elif payment_frequency == 'semi-annual':
                interval_days = 182
            elif payment_frequency == 'annual':
                interval_days = 365
            else:
                interval_days = 91  # Default to quarterly
            
            while current_date <= end_date:
                payment_date = current_date + timedelta(days=14)  # Typically 2 weeks after ex-date
                
                calendar.append({
                    'symbol': symbol,
                    'ex_date': current_date.strftime('%Y-%m-%d'),
                    'payment_date': payment_date.strftime('%Y-%m-%d'),
                    'dividend_per_share': quarterly_dividend,
                    'shares': shares,
                    'total_payment': quarterly_dividend * shares,
                    'frequency': payment_frequency
                })
                
                current_date += timedelta(days=interval_days)
        
        # Sort by payment date
        calendar.sort(key=lambda x: x['payment_date'])
        
        return calendar
    
    def forecast_dividend_income(self,
                                portfolio_value: float,
                                weights: np.ndarray,
                                dividend_yields: np.ndarray,
                                growth_rates: np.ndarray,
                                years: int = 10) -> Dict:
        """
        Forecast future dividend income
        
        Args:
            portfolio_value: Total portfolio value
            weights: Asset allocation weights
            dividend_yields: Current dividend yields (%)
            growth_rates: Expected dividend growth rates (%)
            years: Number of years to forecast
        
        Returns:
            Dictionary with annual income projections
        """
        # Calculate initial annual income
        portfolio_yield = np.dot(weights, dividend_yields) / 100
        initial_income = portfolio_value * portfolio_yield
        
        # Calculate weighted average growth rate
        avg_growth_rate = np.dot(weights, growth_rates) / 100
        
        # Project future income
        projections = []
        cumulative_income = 0
        
        for year in range(1, years + 1):
            annual_income = initial_income * ((1 + avg_growth_rate) ** year)
            cumulative_income += annual_income
            
            projections.append({
                'year': year,
                'annual_income': round(annual_income, 2),
                'cumulative_income': round(cumulative_income, 2),
                'yield_on_cost': round((annual_income / portfolio_value) * 100, 2)
            })
        
        return {
            'initial_annual_income': round(initial_income, 2),
            'initial_yield': round(portfolio_yield * 100, 2),
            'average_growth_rate': round(avg_growth_rate * 100, 2),
            'projections': projections,
            'total_income_10yr': round(cumulative_income, 2)
        }
    
    def analyze_dividend_stock(self, stock_data: Dict) -> Dict:
        """
        Comprehensive dividend stock analysis
        
        Args:
            stock_data: Dictionary with stock information
        
        Returns:
            Complete dividend analysis
        """
        # Extract data
        current_price = stock_data.get('price', 0)
        annual_dividend = stock_data.get('annual_dividend', 0)
        dividend_history = stock_data.get('dividend_history', [])
        eps = stock_data.get('eps', 0)
        consecutive_increases = stock_data.get('consecutive_increases', 0)
        
        # Calculate metrics
        current_yield = self.calculate_dividend_yield(annual_dividend, current_price)
        growth_3yr = self.calculate_dividend_growth_rate(dividend_history, 3)
        growth_5yr = self.calculate_dividend_growth_rate(dividend_history, 5)
        growth_10yr = self.calculate_dividend_growth_rate(dividend_history, 10)
        payout_ratio = self.calculate_payout_ratio(annual_dividend, eps)
        
        # Sustainability
        sustainability = self.calculate_sustainability_score(
            payout_ratio,
            growth_5yr,
            stock_data.get('fcf_coverage', 1.5)
        )
        
        # Aristocrat status
        is_aristocrat = self.is_dividend_aristocrat(consecutive_increases, 'TSX')
        
        return {
            'symbol': stock_data.get('symbol', ''),
            'current_yield': round(current_yield, 2),
            'annual_dividend': annual_dividend,
            'dividend_growth': {
                '3_year': round(growth_3yr, 2),
                '5_year': round(growth_5yr, 2),
                '10_year': round(growth_10yr, 2)
            },
            'payout_ratio': round(payout_ratio, 2),
            'sustainability': sustainability,
            'consecutive_increases': consecutive_increases,
            'is_dividend_aristocrat': is_aristocrat,
            'recommendation': self._generate_recommendation(
                current_yield, growth_5yr, sustainability['score'], is_aristocrat
            )
        }
    
    def _generate_recommendation(self,
                                 current_yield: float,
                                 growth_5yr: float,
                                 sustainability_score: float,
                                 is_aristocrat: bool) -> str:
        """Generate investment recommendation"""
        if sustainability_score >= 80 and current_yield >= 4.0:
            return "Strong Buy - High yield with excellent sustainability"
        elif sustainability_score >= 65 and growth_5yr >= 5.0:
            return "Buy - Good growth with solid sustainability"
        elif sustainability_score >= 50:
            return "Hold - Moderate dividend quality"
        else:
            return "Caution - Dividend sustainability concerns"
