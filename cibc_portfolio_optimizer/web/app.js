/**
 * CIBC Dividend Portfolio Optimizer - Main Application
 */

const API_BASE = 'http://localhost:5001/api';

// State
let currentPortfolio = null;
let allStocks = [];
let charts = {};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    setupFlashGemini();
    setupSignalAnalyzer();
});

async function initializeApp() {
    try {
        // Load market data
        await loadMarketData();

        // Initialize charts
        initializeCharts();

        console.log('CIBC Dividend Portfolio Optimizer initialized');
    } catch (error) {
        console.error('Initialization error:', error);
        showNotification('Failed to initialize application', 'error');
    }
}

function setupEventListeners() {
    // Optimization
    document.getElementById('optimizeBtn').addEventListener('click', optimizePortfolio);

    // AI Advisor
    document.getElementById('askAiBtn').addEventListener('click', askAI);
    document.getElementById('aiQuestion').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') askAI();
    });

    // Efficient Frontier
    document.getElementById('generateFrontierBtn').addEventListener('click', generateEfficientFrontier);

    // Export
    // Export
    document.getElementById('exportBtn').addEventListener('click', exportPortfolio);

    // AI Code Generation
    const generateCodeBtn = document.getElementById('generateCodeBtn');
    if (generateCodeBtn) {
        generateCodeBtn.addEventListener('click', generateCode);
    }

    // Trade Execution
    const tradeBtn = document.getElementById('executeTradeBtn');
    if (tradeBtn) {
        tradeBtn.addEventListener('click', executeTrade);
    }

    // Signal Analyzer
    const signalSelect = document.getElementById('signalStockSelect');
    if (signalSelect) {
        signalSelect.addEventListener('change', updateSignalAnalysis);
    }
}

async function setupSignalAnalyzer() {
    // Populate dropdown once stocks are loaded
    // We can wait for loadMarketData to complete, which is called in initializeApp
    // But we need to ensure allStocks is populated.
    // simpler: check periodically or just call it after loadMarketData returns
}

async function setupFlashGemini() {
    // 1. Fetch Optimal Spreads
    await loadOptimalSpreads();

    // 2. Initialize Flash Gemini Chart (with dummy projection initially)
    initializeFlashGeminiChart();
}

async function loadOptimalSpreads() {
    try {
        const response = await fetch(`${API_BASE}/spreads`);
        const data = await response.json();

        const tbody = document.getElementById('spreadsBody');
        tbody.innerHTML = '';

        // Combine Equities and Forex
        const allAssets = [...data.Equities.slice(0, 5), ...data.Forex];

        allAssets.forEach(asset => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${asset.symbol}</strong></td>
                <td>${asset.type || 'Equity'}</td>
                <td>${asset.bid.toFixed(4)}</td>
                <td>${asset.ask.toFixed(4)}</td>
                <td><span class="badge badge-success">${asset.spread_bps} bps</span></td>
            `;
            tbody.appendChild(row);
        });
    } catch (e) {
        console.error("Error loading spreads:", e);
    }
}

async function executeTrade() {
    if (!currentPortfolio || !currentPortfolio.holdings) {
        showNotification("No portfolio generated to execute.", "warning");
        return;
    }

    const btn = document.getElementById('executeTradeBtn');
    btn.disabled = true;
    btn.textContent = "Executing...";

    try {
        const response = await fetch(`${API_BASE}/trade/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ holdings: currentPortfolio.holdings })
        });

        const result = await response.json();
        const count = result.orders ? result.orders.length : 0;
        showNotification(`Executed ${count} orders via IBKR`, 'success');

    } catch (e) {
        showNotification("Trade execution failed: " + e, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = "Execute Orders (IBKR)";
    }
}

function initializeFlashGeminiChart() {
    const ctx = document.getElementById('flashGeminiChart').getContext('2d');

    // Simulated "Variational Paths" for Flash Gemini Tech stocks
    const years = [2024, 2025, 2026, 2027, 2028];

    charts.flashGemini = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [
                {
                    label: 'NVDA Forecast (AI Boom)',
                    data: [100, 145, 210, 285, 390],
                    borderColor: '#10B981', // Green
                    tension: 0.4
                },
                {
                    label: 'Standard Market Drift',
                    data: [100, 107, 114, 122, 131],
                    borderColor: '#6366F1', // Blue
                    borderDash: [5, 5],
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Variational State Projections', color: '#B8C1EC' },
                legend: { labels: { color: '#B8C1EC' } }
            },
            scales: {
                y: { ticks: { color: '#B8C1EC' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { ticks: { color: '#B8C1EC' }, grid: { display: false } }
            }
        }
    });
}

async function loadMarketData() {
    try {
        const response = await fetch(`${API_BASE}/stocks`);
        const data = await response.json();
        allStocks = data.stocks;
        console.log(`Loaded ${allStocks.length} dividend stocks`);

        // Populate Signal Analyzer Dropdown
        const select = document.getElementById('signalStockSelect');
        if (select) {
            allStocks.forEach(stock => {
                const option = document.createElement('option');
                option.value = stock.symbol;
                option.textContent = `${stock.symbol} - ${stock.name}`;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading market data:', error);
    }
}

async function optimizePortfolio() {
    const btn = document.getElementById('optimizeBtn');
    const progress = document.getElementById('optimizationProgress');

    try {
        // Disable button and show progress
        btn.disabled = true;
        progress.style.display = 'block';

        // Animate progress
        animateProgress();

        // Get parameters
        const portfolioValue = parseFloat(document.getElementById('portfolioValue').value);
        const riskTolerance = document.getElementById('riskTolerance').value;
        const targetYield = parseFloat(document.getElementById('targetYield').value);

        // Call optimization API
        const response = await fetch(`${API_BASE}/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                portfolio_value: portfolioValue,
                risk_tolerance: riskTolerance,
                target_dividend_yield: targetYield,
                target_return: 0.30 // Requested 30% Return
            })
        });

        const data = await response.json();

        if (data.success) {
            currentPortfolio = data;
            displayOptimizationResults(data);
        } else {
            showNotification('Optimization failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Optimization error:', error);
        showNotification('Optimization failed: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        progress.style.display = 'none';
    }
}

function animateProgress() {
    const fill = document.getElementById('progressFill');
    let width = 0;
    const interval = setInterval(() => {
        if (width >= 90) {
            clearInterval(interval);
        } else {
            width += Math.random() * 10;
            fill.style.width = Math.min(width, 90) + '%';
        }
    }, 200);
}

function displayOptimizationResults(data) {
    // Update header stats
    const portfolioValue = data.total_value;
    const annualIncome = portfolioValue * (data.portfolio_metrics.dividend_yield / 100);

    document.getElementById('headerPortfolioValue').textContent = formatCurrency(portfolioValue);
    document.getElementById('headerAnnualIncome').textContent = formatCurrency(annualIncome);
    document.getElementById('headerYield').textContent = data.portfolio_metrics.dividend_yield.toFixed(2) + '%';

    // Update metrics
    document.getElementById('metricReturn').textContent = data.portfolio_metrics.expected_return.toFixed(2) + '%';
    document.getElementById('metricYield').textContent = data.portfolio_metrics.dividend_yield.toFixed(2) + '%';
    document.getElementById('metricSharpe').textContent = data.portfolio_metrics.sharpe_ratio.toFixed(2);
    document.getElementById('metricVolatility').textContent = data.portfolio_metrics.volatility.toFixed(2) + '%';

    // Risk metrics
    document.getElementById('metricVaR').textContent = data.risk_metrics.var_95.toFixed(2) + '%';
    document.getElementById('metricCVaR').textContent = data.risk_metrics.cvar_95.toFixed(2) + '%';
    document.getElementById('metricSortino').textContent = data.portfolio_metrics.sortino_ratio.toFixed(2);

    // Quantum metrics
    displayQuantumMetrics(data.quantum_metrics);

    // Update holdings table
    updateHoldingsTable(data.holdings);

    // Update charts
    updateAllocationChart(data.holdings);
    updateSectorChart(data.sector_allocation);

    // Generate dividend calendar and forecast
    generateDividendCalendar(data.holdings);
    generateIncomeForecast(data);
}

function displayQuantumMetrics(metrics) {
    const container = document.getElementById('quantumMetrics');
    container.style.display = 'block';

    document.getElementById('quantumDepth').textContent = metrics.circuit_depth || '--';
    document.getElementById('quantumQubits').textContent = metrics.num_qubits || '--';
    document.getElementById('quantumIterations').textContent = metrics.convergence_iterations || '--';
    document.getElementById('quantumEnergy').textContent = (metrics.final_energy || 0).toFixed(4);
}

function updateHoldingsTable(holdings) {
    const tbody = document.getElementById('holdingsBody');
    tbody.innerHTML = '';

    holdings.forEach(holding => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${holding.symbol}</strong></td>
            <td>${holding.name}</td>
            <td><span class="badge badge-info">${holding.sector}</span></td>
            <td>${(holding.weight * 100).toFixed(2)}%</td>
            <td>${formatCurrency(holding.value)}</td>
            <td>${holding.shares.toLocaleString()}</td>
            <td>${formatCurrency(holding.price)}</td>
            <td><span class="badge badge-success">${holding.dividend_yield.toFixed(2)}%</span></td>
            <td>${formatCurrency(holding.annual_dividend)}</td>
        `;
        tbody.appendChild(row);
    });
}

function initializeCharts() {
    const chartConfig = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#B8C1EC' }
            }
        }
    };

    // Allocation Chart
    charts.allocation = new Chart(
        document.getElementById('allocationChart'),
        {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
            options: chartConfig
        }
    );

    // Sector Chart
    charts.sector = new Chart(
        document.getElementById('sectorChart'),
        {
            type: 'bar',
            data: { labels: [], datasets: [{ data: [], backgroundColor: '#ED1C24' }] },
            options: {
                ...chartConfig,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#B8C1EC' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#B8C1EC' },
                        grid: { display: false }
                    }
                }
            }
        }
    );

    // Dividend Calendar Chart
    charts.dividendCalendar = new Chart(
        document.getElementById('dividendCalendarChart'),
        {
            type: 'bar',
            data: { labels: [], datasets: [{ data: [], backgroundColor: '#10B981' }] },
            options: {
                ...chartConfig,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#B8C1EC' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#B8C1EC' },
                        grid: { display: false }
                    }
                }
            }
        }
    );

    // Income Forecast Chart
    charts.incomeForecast = new Chart(
        document.getElementById('incomeForecastChart'),
        {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#ED1C24', backgroundColor: 'rgba(237, 28, 36, 0.1)', fill: true }] },
            options: {
                ...chartConfig,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#B8C1EC' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#B8C1EC' },
                        grid: { display: false }
                    }
                }
            }
        }
    );

    // Efficient Frontier Chart
    charts.efficientFrontier = new Chart(
        document.getElementById('efficientFrontierChart'),
        {
            type: 'scatter',
            data: { datasets: [] },
            options: {
                ...chartConfig,
                scales: {
                    y: {
                        title: { display: true, text: 'Expected Return (%)', color: '#B8C1EC' },
                        ticks: { color: '#B8C1EC' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        title: { display: true, text: 'Volatility (%)', color: '#B8C1EC' },
                        ticks: { color: '#B8C1EC' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        }
    );
}

function updateAllocationChart(holdings) {
    const top10 = holdings.slice(0, 10);
    const labels = top10.map(h => h.symbol);
    const data = top10.map(h => h.weight * 100);
    const colors = generateColors(top10.length);

    charts.allocation.data.labels = labels;
    charts.allocation.data.datasets[0].data = data;
    charts.allocation.data.datasets[0].backgroundColor = colors;
    charts.allocation.update();
}

function updateSectorChart(sectorAllocation) {
    const labels = Object.keys(sectorAllocation);
    const data = Object.values(sectorAllocation).map(v => v * 100);

    charts.sector.data.labels = labels;
    charts.sector.data.datasets[0].data = data;
    charts.sector.update();
}

async function generateDividendCalendar(holdings) {
    try {
        const response = await fetch(`${API_BASE}/dividend/calendar`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ holdings, months_ahead: 12 })
        });

        const data = await response.json();

        // Aggregate by month
        const monthlyTotals = data.monthly_totals;
        const labels = Object.keys(monthlyTotals).sort();
        const values = labels.map(month => monthlyTotals[month]);

        charts.dividendCalendar.data.labels = labels.map(m => {
            const date = new Date(m + '-01');
            return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
        });
        charts.dividendCalendar.data.datasets[0].data = values;
        charts.dividendCalendar.data.datasets[0].label = 'Monthly Dividend Income';
        charts.dividendCalendar.update();

    } catch (error) {
        console.error('Error generating dividend calendar:', error);
    }
}

async function generateIncomeForecast(portfolioData) {
    try {
        const response = await fetch(`${API_BASE}/dividend/forecast`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                portfolio_value: portfolioData.total_value,
                weights: portfolioData.optimization.weights,
                years: 10
            })
        });

        const data = await response.json();

        const labels = data.projections.map(p => `Year ${p.year}`);
        const values = data.projections.map(p => p.annual_income);

        charts.incomeForecast.data.labels = labels;
        charts.incomeForecast.data.datasets[0].data = values;
        charts.incomeForecast.data.datasets[0].label = 'Projected Annual Income';
        charts.incomeForecast.update();

    } catch (error) {
        console.error('Error generating income forecast:', error);
    }
}

async function generateEfficientFrontier() {
    const btn = document.getElementById('generateFrontierBtn');
    btn.disabled = true;
    btn.textContent = 'Generating...';

    try {
        const response = await fetch(`${API_BASE}/analytics/efficient-frontier`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_portfolios: 200 })
        });

        const data = await response.json();

        // Plot all portfolios
        const portfolios = data.portfolios.returns.map((ret, i) => ({
            x: data.portfolios.volatilities[i],
            y: ret
        }));

        // Highlight optimal portfolios
        const maxSharpe = {
            x: data.max_sharpe_portfolio.volatility,
            y: data.max_sharpe_portfolio.return
        };

        const minVol = {
            x: data.min_volatility_portfolio.volatility,
            y: data.min_volatility_portfolio.return
        };

        charts.efficientFrontier.data.datasets = [
            {
                label: 'Portfolios',
                data: portfolios,
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                pointRadius: 4
            },
            {
                label: 'Max Sharpe',
                data: [maxSharpe],
                backgroundColor: '#10B981',
                pointRadius: 10,
                pointStyle: 'star'
            },
            {
                label: 'Min Volatility',
                data: [minVol],
                backgroundColor: '#ED1C24',
                pointRadius: 10,
                pointStyle: 'triangle'
            }
        ];

        charts.efficientFrontier.update();
        showNotification('Efficient frontier generated', 'success');

    } catch (error) {
        console.error('Error generating efficient frontier:', error);
        showNotification('Failed to generate efficient frontier', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate';
    }
}

async function askAI() {
    const input = document.getElementById('aiQuestion');
    const question = input.value.trim();

    if (!question) return;

    // Add user message
    addChatMessage(question, 'user');
    input.value = '';

    try {
        const response = await fetch(`${API_BASE}/ai/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                context: {
                    portfolio_data: currentPortfolio ? currentPortfolio.portfolio_metrics : {}
                }
            })
        });

        const data = await response.json();
        addChatMessage(data.answer, 'ai');

    } catch (error) {
        console.error('AI error:', error);
        addChatMessage('Sorry, I encountered an error. Please try again.', 'ai');
    }
}

async function getAIAnalysis() {
    if (!currentPortfolio) return;

    try {
        const response = await fetch(`${API_BASE}/ai/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                portfolio_data: {
                    total_value: currentPortfolio.total_value,
                    dividend_yield: currentPortfolio.portfolio_metrics.dividend_yield,
                    volatility: currentPortfolio.portfolio_metrics.volatility,
                    sharpe_ratio: currentPortfolio.portfolio_metrics.sharpe_ratio,
                    sector_allocation: currentPortfolio.sector_allocation
                },
                market_conditions: {},
                user_profile: {}
            })
        });

        const data = await response.json();
        addChatMessage(data.analysis, 'ai');

    } catch (error) {
        console.error('AI analysis error:', error);
    }
}

async function getAIRecommendations() {
    if (!currentPortfolio) return;

    try {
        const response = await fetch(`${API_BASE}/ai/recommendations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                portfolio_data: {
                    weights: currentPortfolio.optimization.weights,
                    sector_allocation: currentPortfolio.sector_allocation,
                    dividend_yield: currentPortfolio.portfolio_metrics.dividend_yield,
                    tax_efficiency_score: 75
                },
                user_goals: {
                    target_yield: parseFloat(document.getElementById('targetYield').value)
                }
            })
        });

        const data = await response.json();
        displayRecommendations(data.recommendations);

    } catch (error) {
        console.error('AI recommendations error:', error);
    }
}

async function getAdvancedRiskAnalysis(portfolioData) {
    try {
        // Prepare data for risk analysis
        // In a real app, we would use actual historical returns
        // Here we simulate some returns based on portfolio metrics
        const returns = Array.from({ length: 252 }, () => {
            const vol = portfolioData.portfolio_metrics.volatility / 100 / Math.sqrt(252);
            const mean = portfolioData.portfolio_metrics.expected_return / 100 / 252;
            return mean + vol * (Math.random() - 0.5) * 2; // Simplified random walk
        });

        // 1. Risk Classification
        const classResponse = await fetch(`${API_BASE}/risk/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                returns: returns,
                volatility: portfolioData.portfolio_metrics.volatility / 100,
                beta: 1.0, // Assuming market beta of 1 for simplicity
                var_95: portfolioData.risk_metrics.var_95 / 100
            })
        });
        const classData = await classResponse.json();

        // Update UI
        const riskClassEl = document.getElementById('riskClass');
        riskClassEl.textContent = classData.risk_class;
        riskClassEl.style.color = classData.color;
        document.getElementById('riskScore').textContent = `Score: ${classData.risk_score.toFixed(1)}`;

        // 2. Parametric Metrics
        const paramResponse = await fetch(`${API_BASE}/risk/parametric`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                returns: returns,
                portfolio_value: portfolioData.total_value
            })
        });
        const paramData = await paramResponse.json();

        // Update UI
        document.getElementById('marketRegime').textContent = paramData.regime_analysis.regime_stats.find(r => r.regime === paramData.regime_analysis.current_regime).label;
        document.getElementById('regimeProb').textContent = `Confidence: ${(paramData.regime_analysis.regime_probability * 100).toFixed(1)}%`;
        document.getElementById('paramVaR').textContent = (paramData.var_cvar.parametric.var * 100).toFixed(2) + '%';
        document.getElementById('maxDrawdown').textContent = (paramData.drawdown.max_drawdown * 100).toFixed(2) + '%';
        document.getElementById('distFit').textContent = paramData.distribution_fit.best_distribution;
        document.getElementById('tailRisk').textContent = paramData.tail_risk.left_tail_index.toFixed(2);

    } catch (error) {
        console.error('Error getting risk analysis:', error);
    }
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('aiRecommendations');
    const list = document.getElementById('recommendationsList');

    if (recommendations.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    list.innerHTML = '';

    recommendations.forEach(rec => {
        const div = document.createElement('div');
        div.className = 'glass-card';
        div.style.marginBottom = '0.5rem';
        div.style.padding = '0.75rem';

        const priorityBadge = rec.priority === 'High' ? 'badge-danger' :
            rec.priority === 'Medium' ? 'badge-warning' : 'badge-info';

        div.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                <strong>${rec.title}</strong>
                <span class="badge ${priorityBadge}">${rec.priority}</span>
            </div>
            <p style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                ${rec.description}
            </p>
        `;

        list.appendChild(div);
    });
}

function addChatMessage(message, sender) {
    const chat = document.getElementById('aiChat');
    const div = document.createElement('div');
    div.className = `chat-message ${sender}`;
    div.innerHTML = `<strong>${sender === 'user' ? 'You' : 'AI Advisor'}:</strong> ${message}`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function exportPortfolio() {
    if (!currentPortfolio) {
        showNotification('No portfolio to export', 'warning');
        return;
    }

    const csv = generateCSV(currentPortfolio.holdings);
    downloadCSV(csv, 'cibc_dividend_portfolio.csv');
    showNotification('Portfolio exported successfully', 'success');
}

function generateCSV(holdings) {
    const headers = ['Symbol', 'Name', 'Sector', 'Weight', 'Value', 'Shares', 'Price', 'Yield', 'Annual Dividend'];
    const rows = holdings.map(h => [
        h.symbol,
        h.name,
        h.sector,
        (h.weight * 100).toFixed(2) + '%',
        h.value.toFixed(2),
        h.shares,
        h.price.toFixed(2),
        h.dividend_yield.toFixed(2) + '%',
        h.annual_dividend.toFixed(2)
    ]);

    return [headers, ...rows].map(row => row.join(',')).join('\n');
}

function downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-CA', {
        style: 'currency',
        currency: 'CAD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

function generateColors(count) {
    const colors = [
        '#ED1C24', '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6',
        '#EC4899', '#14B8A6', '#F97316', '#6366F1', '#EF4444'
    ];


    return Array.from({ length: count }, (_, i) => colors[i % colors.length]);
}

async function updateSignalAnalysis() {
    const symbol = document.getElementById('signalStockSelect').value;
    const container = document.getElementById('signalsContainer');

    if (!symbol) {
        container.style.display = 'none';
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/stocks/${symbol}`);
        const data = await response.json();
        const signals = data.analysis.signals;

        container.style.display = 'grid'; // grid-4 class handles layout

        // 1. Yield Safety
        updateSignalCard('signalYield', signals['Yield Safety']);

        // 2. Growth Trajectory
        updateSignalCard('signalGrowth', signals['Growth Trajectory']);

        // 3. Fundamental Health
        updateSignalCard('signalHealth', signals['Fundamental Health']);

        // 4. Momentum
        updateSignalCard('signalMomentum', signals['Momentum']);

    } catch (e) {
        console.error("Error fetching signals:", e);
    }
}

function updateSignalCard(id, signalData) {
    const card = document.getElementById(id);
    const valueEl = card.querySelector('.metric-value');
    const changeEl = card.querySelector('.metric-change');
    const fillEl = card.querySelector('.progress-fill');

    valueEl.textContent = signalData.status;

    // Format metrics
    const metrics = Object.entries(signalData.metrics)
        .map(([k, v]) => `${k}: ${v}`)
        .join(' | ');
    changeEl.textContent = metrics;

    fillEl.style.width = `${signalData.score}%`;

    // Color coding based on score
    if (signalData.score > 70) fillEl.style.backgroundColor = 'var(--success)';
    else if (signalData.score < 40) fillEl.style.backgroundColor = 'var(--danger)';
    else fillEl.style.backgroundColor = 'var(--warning)';
}

function showNotification(message, type = 'info') {
    // Simple console notification for now
    console.log(`[${type.toUpperCase()}] ${message}`);

    // Could implement toast notifications here
}

async function generateCode() {
    const input = document.getElementById('aiQuestion');
    const query = input.value.trim() || "Generate portfolio optimization code";

    try {
        const response = await fetch(`${API_BASE}/ai/generate-code`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                context: currentPortfolio ? {
                    portfolio_metrics: currentPortfolio.portfolio_metrics,
                    holdings: currentPortfolio.holdings
                } : {}
            })
        });

        const data = await response.json();

        document.getElementById('generatedCode').textContent = data.code;
        document.getElementById('codeModal').style.display = 'block';

    } catch (error) {
        console.error('Error generating code:', error);
        showNotification('Failed to generate code', 'error');
    }
}
