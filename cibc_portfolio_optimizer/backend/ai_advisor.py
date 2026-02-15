"""
CIBC AI Advisor
Generative AI-powered portfolio recommendations and natural language insights
"""

import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime


class AIAdvisor:
    """AI-powered portfolio advisor with natural language capabilities"""
    
    def __init__(self):
        self.conversation_history = []
        
    def analyze_portfolio(self,
                         portfolio_data: Dict,
                         market_conditions: Dict,
                         user_profile: Dict) -> str:
        """
        Generate comprehensive portfolio analysis in natural language
        
        Args:
            portfolio_data: Current portfolio metrics and holdings
            market_conditions: Current market environment
            user_profile: User risk tolerance and goals
        
        Returns:
            Natural language analysis
        """
        analysis_parts = []
        
        # Portfolio overview
        total_value = portfolio_data.get('total_value', 0)
        dividend_yield = portfolio_data.get('dividend_yield', 0)
        annual_income = total_value * (dividend_yield / 100)
        
        analysis_parts.append(
            f"Your dividend portfolio is currently valued at ${total_value:,.2f}, "
            f"generating an annual dividend income of ${annual_income:,.2f} "
            f"with a portfolio yield of {dividend_yield:.2f}%."
        )
        
        # Risk assessment
        volatility = portfolio_data.get('volatility', 0)
        sharpe_ratio = portfolio_data.get('sharpe_ratio', 0)
        
        risk_assessment = self._assess_risk_level(volatility, sharpe_ratio)
        analysis_parts.append(risk_assessment)
        
        # Sector diversification
        sector_allocation = portfolio_data.get('sector_allocation', {})
        diversification_comment = self._analyze_diversification(sector_allocation)
        analysis_parts.append(diversification_comment)
        
        # Dividend sustainability
        sustainability_score = portfolio_data.get('avg_sustainability_score', 75)
        sustainability_comment = self._analyze_sustainability(sustainability_score)
        analysis_parts.append(sustainability_comment)
        
        # Market conditions context
        market_comment = self._analyze_market_conditions(market_conditions)
        analysis_parts.append(market_comment)
        
        return " ".join(analysis_parts)
    
    def generate_recommendations(self,
                                portfolio_data: Dict,
                                stocks: List[Dict],
                                user_goals: Dict) -> List[Dict]:
        """
        Generate AI-powered portfolio recommendations
        
        Args:
            portfolio_data: Current portfolio data
            stocks: Available stock universe
            user_goals: User investment goals
        
        Returns:
            List of recommendations with rationale
        """
        recommendations = []
        
        # Analyze current holdings
        current_weights = portfolio_data.get('weights', [])
        sector_allocation = portfolio_data.get('sector_allocation', {})
        
        # Recommendation 1: Sector rebalancing
        if self._needs_sector_rebalancing(sector_allocation):
            recommendations.append({
                'type': 'Rebalancing',
                'priority': 'High',
                'title': 'Sector Diversification Opportunity',
                'description': self._generate_rebalancing_recommendation(sector_allocation),
                'action_items': self._suggest_rebalancing_trades(sector_allocation, stocks)
            })
        
        # Recommendation 2: Dividend growth opportunities
        high_growth_stocks = self._find_high_growth_dividend_stocks(stocks)
        if high_growth_stocks:
            recommendations.append({
                'type': 'Growth',
                'priority': 'Medium',
                'title': 'Dividend Growth Opportunities',
                'description': f"Consider adding dividend growth stocks to enhance long-term income. "
                              f"These stocks have demonstrated strong dividend growth rates above 7% annually.",
                'action_items': [
                    {
                        'symbol': stock['symbol'],
                        'name': stock['name'],
                        'current_yield': stock['dividend_yield'],
                        'growth_rate': self._calculate_growth_rate(stock),
                        'rationale': f"Strong {stock['sector']} position with {stock['consecutive_increases']} years of increases"
                    }
                    for stock in high_growth_stocks[:3]
                ]
            })
        
        # Recommendation 3: Tax optimization
        tax_efficiency = portfolio_data.get('tax_efficiency_score', 70)
        if tax_efficiency < 75:
            recommendations.append({
                'type': 'Tax Optimization',
                'priority': 'Medium',
                'title': 'Enhance Tax Efficiency',
                'description': self._generate_tax_optimization_advice(portfolio_data),
                'action_items': self._suggest_tax_optimization_trades(stocks)
            })
        
        # Recommendation 4: Yield enhancement
        current_yield = portfolio_data.get('dividend_yield', 0)
        target_yield = user_goals.get('target_yield', 5.0)
        
        if current_yield < target_yield:
            recommendations.append({
                'type': 'Yield Enhancement',
                'priority': 'High',
                'title': f'Increase Portfolio Yield to {target_yield}%',
                'description': f"Your current yield of {current_yield:.2f}% is below your target of {target_yield}%. "
                              f"Consider adding higher-yielding stocks while maintaining quality.",
                'action_items': self._suggest_yield_enhancement_trades(stocks, target_yield)
            })
        
        return recommendations
    
    def answer_question(self, question: str, context: Dict) -> str:
        """
        Answer user questions about dividend investing using AI
        
        Args:
            question: User's question
            context: Portfolio and market context
        
        Returns:
            Natural language answer
        """
        question_lower = question.lower()
        
        # Pattern matching for common questions
        if 'best' in question_lower and 'dividend' in question_lower:
            return self._answer_best_dividend_stocks(context)
        
        elif 'risk' in question_lower:
            return self._answer_risk_question(context)
        
        elif 'tax' in question_lower:
            return self._answer_tax_question(context)
        
        elif 'rebalance' in question_lower or 'adjust' in question_lower:
            return self._answer_rebalancing_question(context)
        
        elif 'income' in question_lower or 'how much' in question_lower:
            return self._answer_income_question(context)
        
        elif 'sector' in question_lower:
            return self._answer_sector_question(context)
        
        else:
            return self._generate_general_answer(question, context)
    
    def _assess_risk_level(self, volatility: float, sharpe_ratio: float) -> str:
        """Assess portfolio risk level"""
        if volatility < 12:
            risk_level = "low"
            comment = "Your portfolio exhibits low volatility, suitable for conservative investors."
        elif volatility < 18:
            risk_level = "moderate"
            comment = "Your portfolio has moderate volatility, balanced for steady growth."
        else:
            risk_level = "elevated"
            comment = "Your portfolio shows elevated volatility. Consider adding more stable dividend payers."
        
        if sharpe_ratio > 1.0:
            comment += f" The Sharpe ratio of {sharpe_ratio:.2f} indicates excellent risk-adjusted returns."
        elif sharpe_ratio > 0.5:
            comment += f" The Sharpe ratio of {sharpe_ratio:.2f} shows reasonable risk-adjusted performance."
        else:
            comment += f" The Sharpe ratio of {sharpe_ratio:.2f} suggests room for improvement in risk-adjusted returns."
        
        return comment
    
    def _analyze_diversification(self, sector_allocation: Dict) -> str:
        """Analyze sector diversification"""
        if not sector_allocation:
            return "Sector allocation data unavailable."
        
        max_sector = max(sector_allocation.items(), key=lambda x: x[1])
        max_concentration = max_sector[1] * 100
        
        if max_concentration > 40:
            return (f"Your portfolio is heavily concentrated in {max_sector[0]} ({max_concentration:.1f}%). "
                   f"Consider diversifying across other sectors to reduce concentration risk.")
        elif max_concentration > 30:
            return (f"Your portfolio has moderate concentration in {max_sector[0]} ({max_concentration:.1f}%). "
                   f"Diversification is reasonable but could be improved.")
        else:
            return f"Your portfolio shows good sector diversification with balanced exposure across multiple sectors."
    
    def _analyze_sustainability(self, sustainability_score: float) -> str:
        """Analyze dividend sustainability"""
        if sustainability_score >= 80:
            return "Your dividend holdings demonstrate excellent sustainability with strong payout ratios and cash flow coverage."
        elif sustainability_score >= 65:
            return "Your dividends appear sustainable with good fundamentals, though monitoring is recommended."
        elif sustainability_score >= 50:
            return "Some dividend holdings show moderate sustainability concerns. Review payout ratios and earnings trends."
        else:
            return "Warning: Several holdings show dividend sustainability risks. Consider upgrading to higher-quality dividend payers."
    
    def _analyze_market_conditions(self, market_conditions: Dict) -> str:
        """Analyze current market conditions"""
        # Simplified market analysis
        return ("Current market conditions favor quality dividend stocks with strong balance sheets. "
               "Interest rate environment remains supportive of dividend-focused strategies.")
    
    def _needs_sector_rebalancing(self, sector_allocation: Dict) -> bool:
        """Determine if sector rebalancing is needed"""
        if not sector_allocation:
            return False
        
        max_weight = max(sector_allocation.values())
        return max_weight > 0.35  # More than 35% in one sector
    
    def _generate_rebalancing_recommendation(self, sector_allocation: Dict) -> str:
        """Generate sector rebalancing recommendation"""
        sorted_sectors = sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True)
        overweight = sorted_sectors[0]
        underweight = sorted_sectors[-1]
        
        return (f"Your portfolio is overweight {overweight[0]} at {overweight[1]*100:.1f}%. "
               f"Consider reducing exposure and increasing allocation to {underweight[0]} "
               f"(currently {underweight[1]*100:.1f}%) for better diversification.")
    
    def _suggest_rebalancing_trades(self, sector_allocation: Dict, stocks: List[Dict]) -> List[Dict]:
        """Suggest specific rebalancing trades"""
        # Find overweight sector
        overweight_sector = max(sector_allocation.items(), key=lambda x: x[1])[0]
        underweight_sector = min(sector_allocation.items(), key=lambda x: x[1])[0]
        
        # Find stocks to reduce/add
        reduce_stocks = [s for s in stocks if s['sector'] == overweight_sector][:2]
        add_stocks = [s for s in stocks if s['sector'] == underweight_sector][:2]
        
        trades = []
        for stock in reduce_stocks:
            trades.append({
                'action': 'Reduce',
                'symbol': stock['symbol'],
                'name': stock['name'],
                'sector': stock['sector']
            })
        
        for stock in add_stocks:
            trades.append({
                'action': 'Add',
                'symbol': stock['symbol'],
                'name': stock['name'],
                'sector': stock['sector'],
                'yield': stock['dividend_yield']
            })
        
        return trades
    
    def _find_high_growth_dividend_stocks(self, stocks: List[Dict]) -> List[Dict]:
        """Find stocks with high dividend growth"""
        growth_stocks = []
        
        for stock in stocks:
            if len(stock.get('dividend_history', [])) >= 5:
                recent = stock['dividend_history'][-1]
                old = stock['dividend_history'][-5]
                growth_rate = ((recent / old) ** 0.2 - 1) * 100 if old > 0 else 0
                
                if growth_rate > 7:
                    growth_stocks.append(stock)
        
        return sorted(growth_stocks, key=lambda x: x.get('dividend_yield', 0), reverse=True)
    
    def _calculate_growth_rate(self, stock: Dict) -> float:
        """Calculate dividend growth rate"""
        history = stock.get('dividend_history', [])
        if len(history) >= 5:
            return ((history[-1] / history[-5]) ** 0.2 - 1) * 100
        return 0.0
    
    def _generate_tax_optimization_advice(self, portfolio_data: Dict) -> str:
        """Generate tax optimization advice"""
        return ("Maximize your after-tax returns by ensuring all dividends are eligible for the Canadian dividend tax credit. "
               "Consider holding dividend stocks in non-registered accounts to benefit from preferential tax treatment, "
               "while placing fixed income in registered accounts (RRSP/TFSA).")
    
    def _suggest_tax_optimization_trades(self, stocks: List[Dict]) -> List[Dict]:
        """Suggest tax-optimized trades"""
        eligible_stocks = [s for s in stocks if s.get('is_eligible_dividend', True)][:3]
        
        return [
            {
                'action': 'Consider',
                'symbol': stock['symbol'],
                'name': stock['name'],
                'benefit': 'Eligible dividend - lower effective tax rate',
                'yield': stock['dividend_yield']
            }
            for stock in eligible_stocks
        ]
    
    def _suggest_yield_enhancement_trades(self, stocks: List[Dict], target_yield: float) -> List[Dict]:
        """Suggest trades to enhance yield"""
        high_yield_stocks = sorted(
            [s for s in stocks if s['dividend_yield'] >= target_yield],
            key=lambda x: x['dividend_yield'],
            reverse=True
        )[:5]
        
        return [
            {
                'symbol': stock['symbol'],
                'name': stock['name'],
                'yield': stock['dividend_yield'],
                'sector': stock['sector'],
                'sustainability': 'Good' if stock.get('consecutive_increases', 0) > 5 else 'Monitor',
                'rationale': f"High yield with {stock['consecutive_increases']} years of dividend growth"
            }
            for stock in high_yield_stocks
        ]
    
    def _answer_best_dividend_stocks(self, context: Dict) -> str:
        """Answer question about best dividend stocks"""
        stocks = context.get('stocks', [])
        top_stocks = sorted(stocks, key=lambda x: x['dividend_yield'], reverse=True)[:5]
        
        response = "Based on current market conditions, here are the top dividend stocks to consider:\n\n"
        for i, stock in enumerate(top_stocks, 1):
            response += (f"{i}. {stock['name']} ({stock['symbol']}) - "
                        f"Yield: {stock['dividend_yield']:.2f}%, "
                        f"Sector: {stock['sector']}, "
                        f"{stock['consecutive_increases']} years of increases\n")
        
        response += "\nThese stocks offer attractive yields with solid dividend growth histories."
        return response
    
    def _answer_risk_question(self, context: Dict) -> str:
        """Answer risk-related questions"""
        portfolio = context.get('portfolio_data', {})
        volatility = portfolio.get('volatility', 15)
        
        return (f"Your portfolio's current volatility is {volatility:.1f}%. "
               f"For a dividend-focused portfolio, this is {'low' if volatility < 12 else 'moderate' if volatility < 18 else 'elevated'}. "
               f"To reduce risk, consider increasing allocation to stable sectors like Utilities and Financials, "
               f"while maintaining diversification across at least 15-20 holdings.")
    
    def _answer_tax_question(self, context: Dict) -> str:
        """Answer tax-related questions"""
        return ("Canadian dividend investors benefit from the dividend tax credit, which significantly reduces "
               "the effective tax rate on eligible dividends. Eligible dividends from Canadian corporations "
               "are grossed up by 38% but receive a federal tax credit of ~15%, resulting in an effective "
               "tax rate of approximately 30% (vs 50% on interest income). This makes dividend stocks "
               "highly tax-efficient for non-registered accounts.")
    
    def _answer_rebalancing_question(self, context: Dict) -> str:
        """Answer rebalancing questions"""
        return ("Portfolio rebalancing should be done quarterly or semi-annually to maintain your target "
               "asset allocation. For dividend portfolios, consider rebalancing when: (1) a sector exceeds "
               "35% of portfolio, (2) a single position exceeds 10%, or (3) your dividend yield drifts "
               "more than 0.5% from target. Use new contributions to rebalance when possible to minimize "
               "tax implications.")
    
    def _answer_income_question(self, context: Dict) -> str:
        """Answer income projection questions"""
        portfolio = context.get('portfolio_data', {})
        total_value = portfolio.get('total_value', 100000)
        dividend_yield = portfolio.get('dividend_yield', 5.0)
        annual_income = total_value * (dividend_yield / 100)
        monthly_income = annual_income / 12
        
        return (f"Your current portfolio generates approximately ${annual_income:,.2f} in annual dividend income, "
               f"or ${monthly_income:,.2f} per month. With an average dividend growth rate of 5% annually, "
               f"your income could grow to ${annual_income * 1.05**5:,.2f} in 5 years and "
               f"${annual_income * 1.05**10:,.2f} in 10 years, assuming you reinvest and maintain current allocations.")
    
    def _answer_sector_question(self, context: Dict) -> str:
        """Answer sector allocation questions"""
        return ("For a balanced dividend portfolio, consider: Financials (25-30%) for stable yields and growth, "
               "Utilities (15-20%) for defensive income, Energy/Pipelines (15-20%) for high yields, "
               "Telecom (10-15%) for reliable dividends, and REITs (10-15%) for diversification. "
               "Adjust based on your risk tolerance and income needs.")
    
    def _generate_general_answer(self, question: str, context: Dict) -> str:
        """Generate general answer for unmatched questions"""
        return ("I can help you with dividend portfolio questions including: best dividend stocks, "
               "risk assessment, tax optimization, rebalancing strategies, income projections, "
               "and sector allocation. Please ask a specific question about your dividend portfolio.")

    def generate_python_code(self, query: str, context: Dict) -> str:
        """
        Generate Python code for portfolio analysis based on user query
        Simulates a code-generation LLM
        """
        query_lower = query.lower()
        
        if 'optimization' in query_lower or 'optimize' in query_lower:
            return self._generate_optimization_code()
        elif 'efficient frontier' in query_lower:
            return self._generate_frontier_code()
        elif 'risk' in query_lower or 'var' in query_lower:
            return self._generate_risk_code()
        else:
            return self._generate_analysis_code()

    def _generate_optimization_code(self) -> str:
        return """
import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(returns, cov_matrix, risk_aversion=1.0):
    n_assets = len(returns)
    
    def objective(weights):
        portfolio_return = np.dot(weights, returns)
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        return -(portfolio_return - risk_aversion * portfolio_var)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, np.ones(n_assets)/n_assets, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
"""

    def _generate_frontier_code(self) -> str:
        return """
import numpy as np

def generate_efficient_frontier(returns, cov_matrix, num_portfolios=1000):
    results = []
    n_assets = len(returns)
    
    for _ in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        p_ret = np.dot(weights, returns)
        p_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        results.append((p_std, p_ret))
        
    return results
"""

    def _generate_risk_code(self) -> str:
        return """
import numpy as np
from scipy.stats import norm

def calculate_var_cvar(returns, confidence=0.95):
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    z_score = norm.ppf(1 - confidence)
    var = mu + z_score * sigma
    cvar = mu - sigma * norm.pdf(z_score) / (1 - confidence)
    
    return var, cvar
"""

    def _generate_analysis_code(self) -> str:
        return """
import pandas as pd

def analyze_portfolio_performance(holdings_df):
    # Calculate basic metrics
    total_value = (holdings_df['Shares'] * holdings_df['Price']).sum()
    dividend_income = (holdings_df['Shares'] * holdings_df['Annual_Dividend']).sum()
    yield_pct = (dividend_income / total_value) * 100
    
    print(f"Portfolio Value: ${total_value:,.2f}")
    print(f"Annual Income: ${dividend_income:,.2f}")
    print(f"Yield: {yield_pct:.2f}%")
    
    return {
        'total_value': total_value,
        'yield': yield_pct
    }
"""
