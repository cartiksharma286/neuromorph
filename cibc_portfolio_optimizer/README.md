# CIBC Optimal Dividend Portfolio with Generative AI

A comprehensive capital markets portfolio management system that leverages **quantum computing** and **generative AI** to optimize dividend-focused portfolios for Canadian investors.

## 🚀 Features

### Quantum-Enhanced Optimization
- **VQE (Variational Quantum Eigensolver)** for portfolio optimization
- Quantum circuit design for dividend yield maximization
- Risk-adjusted return optimization with quantum annealing
- Real-time quantum metrics visualization

### Generative AI Advisor
- Natural language portfolio analysis and recommendations
- Intelligent rebalancing suggestions
- Risk assessment and commentary generation
- Interactive Q&A for dividend investing strategies

### Comprehensive Dividend Analytics
- **Canadian dividend tax credit calculations** (eligible vs non-eligible)
- Dividend growth rate analysis (3-year, 5-year, 10-year CAGR)
- Payout ratio sustainability scoring
- Dividend aristocrat/achiever identification
- Ex-dividend date tracking and calendar generation
- 10-year income forecasting

### Advanced Portfolio Analytics
- Risk metrics: Sharpe ratio, Sortino ratio, VaR, CVaR
- Efficient frontier generation
- Performance attribution (sector, security)
- Tax efficiency scoring for Canadian investors
- Correlation matrix analysis

### Premium CIBC-Branded Interface
- Dark theme with glassmorphism effects
- Real-time interactive charts (Chart.js)
- Responsive design
- Dividend calendar visualization
- Income projection charts

## 📊 Canadian Dividend Stock Universe

The system includes **24 TSX blue-chip dividend stocks** across 5 sectors:

- **Financials**: RY, TD, BNS, BMO, CM, MFC, SLF
- **Utilities**: FTS, EMA, AQN, CU
- **Energy**: ENB, TRP, CNQ, SU
- **Telecom**: BCE, T, RCI.B
- **REITs**: REI.UN, CAR.UN, HR.UN

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- CUDA-Q (for quantum optimization)
- Modern web browser

### Setup

1. **Install Python dependencies:**
```powershell
cd C:\Users\User\.gemini\antigravity\scratch\cibc-dividend-portfolio
pip install -r requirements.txt
```

2. **Start the backend server:**
```powershell
cd backend
python server.py
```

Server will start on `http://localhost:5000`

3. **Open the web interface:**
```powershell
cd ..\web
start index.html
```

Or navigate to `C:\Users\User\.gemini\antigravity\scratch\cibc-dividend-portfolio\web\index.html` in your browser.

## 📖 Usage Guide

### 1. Portfolio Optimization

1. Set your **Portfolio Value** (default: $100,000)
2. Choose **Risk Tolerance**: Conservative, Moderate, or Aggressive
3. Set **Target Dividend Yield** (e.g., 5.0%)
4. Click **"Optimize with Quantum AI"**

The quantum optimizer will:
- Find optimal asset allocation
- Maximize dividend yield while managing risk
- Apply sector diversification constraints
- Display quantum circuit metrics

### 2. AI Advisor

Ask questions like:
- "What are the best dividend stocks for income?"
- "How can I reduce portfolio risk?"
- "What's the tax impact of my dividends?"
- "Should I rebalance my portfolio?"

The AI will provide personalized recommendations based on your portfolio.

### 3. Dividend Calendar

View upcoming dividend payments by month, helping you plan cash flow.

### 4. Income Forecast

See projected dividend income growth over 10 years based on historical dividend growth rates.

### 5. Efficient Frontier

Click **"Generate"** to visualize the risk-return tradeoff across different portfolio allocations.

## 🔧 API Endpoints

### Portfolio Optimization
```
POST /api/optimize
Body: {
  "portfolio_value": 100000,
  "risk_tolerance": "moderate",
  "target_dividend_yield": 5.0
}
```

### AI Analysis
```
POST /api/ai/analyze
Body: {
  "portfolio_data": {...},
  "market_conditions": {...},
  "user_profile": {...}
}
```

### Dividend Calendar
```
POST /api/dividend/calendar
Body: {
  "holdings": [...],
  "months_ahead": 12
}
```

### Income Forecast
```
POST /api/dividend/forecast
Body: {
  "portfolio_value": 100000,
  "weights": [...],
  "years": 10
}
```

## 🎨 CIBC Branding Guidelines

The interface uses official CIBC brand colors:
- **Primary Red**: #ED1C24
- **Dark Red**: #C41E3A
- **Light Red**: #FF4D4D

Typography: **Inter** font family

## 🧪 Testing

Run portfolio optimization with different risk profiles to verify:
- Conservative → stable dividend payers (Utilities, Financials)
- Moderate → balanced allocation
- Aggressive → higher growth stocks (Energy, REITs)

## 📊 Key Metrics Explained

### Sharpe Ratio
Risk-adjusted return metric. Higher is better (>1.0 is excellent).

### Sortino Ratio
Similar to Sharpe but focuses on downside risk only.

### VaR (Value at Risk)
Maximum expected loss at 95% confidence level.

### CVaR (Conditional VaR)
Expected loss beyond the VaR threshold.

### Dividend Sustainability Score
Composite score (0-100) based on:
- Payout ratio
- Dividend growth history
- Free cash flow coverage

## 🇨🇦 Canadian Tax Considerations

The system accounts for:
- **Eligible dividend tax credit** (38% gross-up, ~15% federal credit)
- **Effective tax rate** (~30% vs 50% on interest)
- **Tax efficiency scoring** for non-registered accounts

## 🚀 Future Enhancements

- Real-time market data integration
- Multi-currency support
- ESG scoring integration
- Mobile app version
- Portfolio backtesting

## 📝 License

For demonstration purposes only. Not financial advice.

## 🤝 Support

For questions or issues, contact CIBC Capital Markets.

---

**Built with ❤️ using Quantum Computing & Generative AI**
