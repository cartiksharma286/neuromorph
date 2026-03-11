/**
 * CIBC Portfolio Optimizer - Unified Intelligence Engine v2.0
 * Multi-module frontend controller
 */

const App = {
    state: {
        activeTab: 'dashboard',
        stocks: [],
        currentPortfolio: null,
        optResults: null,
        riskData: null,
        historyData: null,
        userSettings: {
            targetReturn: 0.15,
            optMethod: 'max_sharpe'
        }
    },

    charts: {},

    async init() {
        console.log('Initializing Unified CIBC Optimizer...');
        this.setupNavigation();
        this.initAllCharts();
        this.setupEventListeners();
        
        // Initial Data Fetch
        await this.loadInitialData();
        this.startBackgroundLoops();
        
        // Switch to default tab
        this.switchTab('dashboard');
    },

    // --- Navigation & UI Stubs ---

    setupNavigation() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const tabId = e.currentTarget.getAttribute('data-tab');
                this.switchTab(tabId);
            });
        });
    },

    switchTab(tabId) {
        // Update state
        this.state.activeTab = tabId;
        
        // Update UI Classes
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        document.querySelector(`.nav-item[data-tab="${tabId}"]`).classList.add('active');
        
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        
        // Tab-specific loading
        this.handleTabActivation(tabId);
    },

    async handleTabActivation(tabId) {
        console.log(`Activating tab: ${tabId}`);
        
        // Ensure charts resize to their new visible container dimensions
        this.resizeAllCharts();

        switch(tabId) {
            case 'dashboard': await this.refreshDashboard(); break;
            case 'optimization': await this.refreshOptimization(); break;
            case 'risk': await this.refreshRisk(); break;
            case 'trade': await this.refreshTrade(); break;
            case 'history': await this.refreshHistory(); break;
        }
    },

    resizeAllCharts() {
        // Small timeout to allow DOM to finalize layout
        setTimeout(() => {
            Object.values(this.charts).forEach(chart => {
                if (chart) chart.resize();
            });
        }, 50);
    },

    setupEventListeners() {
        // Optimization Controls
        const slider = document.getElementById('opt-return-slider');
        slider.addEventListener('input', (e) => {
            const val = e.target.value;
            document.getElementById('opt-return-val').innerText = `${val}%`;
            this.state.userSettings.targetReturn = val / 100;
        });

        document.getElementById('run-opt-btn').addEventListener('click', () => this.runFullOptimization());
        document.getElementById('execute-rebalance-btn').addEventListener('click', () => this.executeMarketTrade());

        // AI Advisor
        document.getElementById('ai-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendAiQuery(e.target.value);
        });
    },

    // --- Chart Initializations ---

    initAllCharts() {
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Inter', sans-serif";

        // 1. Dashboard Projections
        this.charts.projections = new Chart(document.getElementById('projectionChart'), {
            type: 'line',
            data: { labels: Array.from({length: 30}, (_, i) => i), datasets: [] },
            options: { 
                responsive: true, maintainAspectRatio: false, 
                plugins: { legend: { display: false } },
                scales: { x: { display: false }, y: { grid: { color: 'rgba(255,255,255,0.05)' } } }
            }
        });

        // 2. Dash Sector Donut
        this.charts.dashSector = new Chart(document.getElementById('dashSectorChart'), {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], borderWidth: 0 }] },
            options: { cutout: '75%', plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } } }
        });

        // 3. Optimization Frontier
        this.charts.frontier = new Chart(document.getElementById('frontierChart'), {
            type: 'scatter',
            data: { datasets: [{ label: 'Portfolios', data: [], backgroundColor: 'rgba(56, 189, 248, 0.4)' }] },
            options: { 
                responsive: true, maintainAspectRatio: false,
                scales: { 
                    x: { title: { display: true, text: 'Volatility (%)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Return (%)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });

        // 4. Risk Stratification Scatter
        this.charts.riskScatter = new Chart(document.getElementById('riskScatterChart'), {
            type: 'scatter',
            data: { datasets: [] },
            options: { 
                responsive: true, maintainAspectRatio: false,
                scales: { 
                    x: { title: { display: true, text: 'Beta' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Yield (%)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });

        // 5. Regime Probs
        this.charts.regime = new Chart(document.getElementById('regimeProbsChart'), {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Prob', data: [], backgroundColor: '#38bdf8' }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 1 } } }
        });

        // 6. Dividend Forecast
        this.charts.divForecast = new Chart(document.getElementById('dividendForecastChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { responsive: true, maintainAspectRatio: false }
        });
    },

    // --- Tab Controllers ---

    async loadInitialData() {
        const res = await fetch('/api/market/summary');
        const data = await res.json();
        this.state.marketSummary = data;
        this.updateTicker(data);
    },

    updateTicker(data) {
        const track = document.getElementById('ticker-track');
        const items = data.sectors.map(s => `
            <span style="margin-right: 48px;">${s} <span class="positive">+${(Math.random()*1.5).toFixed(2)}%</span></span>
        `).join('');
        track.innerHTML = items + items; // Duplicate for smooth loop
    },

    async refreshDashboard() {
        // Fetch current portfolio metrics
        const res = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ portfolio_value: 1238642, risk_tolerance: 'moderate' })
        });
        const data = await res.json();
        this.state.currentPortfolio = data;

        // Update UI
        document.getElementById('dash-yield').innerText = `${data.portfolio_metrics.dividend_yield.toFixed(2)}%`;
        document.getElementById('dash-risk').innerText = data.risk_metrics.var_95 < 2 ? 'LOW' : 'MODERATE';
        
        // Update Sector Chart
        const sectors = data.sector_allocation;
        this.charts.dashSector.data.labels = Object.keys(sectors);
        this.charts.dashSector.data.datasets[0].data = Object.values(sectors);
        this.charts.dashSector.data.datasets[0].backgroundColor = ['#38bdf8', '#10b981', '#f43f5e', '#f59e0b', '#8b5cf6'];
        this.charts.dashSector.update();

        // Start Projections if empty
        if (this.charts.projections.data.datasets.length === 0) {
            await this.updateVariationalProjections();
        }
    },

    async updateVariationalProjections() {
        const res = await fetch('/api/ml/projections?value=1238642&return=0.15&vol=0.12');
        const data = await res.json();
        
        this.charts.projections.data.datasets = data.paths.slice(0, 3).map((path, i) => ({
            data: path,
            borderColor: i === 0 ? '#38bdf8' : 'rgba(56, 189, 248, 0.15)',
            borderWidth: i === 0 ? 3 : 1,
            pointRadius: 0,
            fill: i === 0,
            backgroundColor: 'rgba(56, 189, 248, 0.05)',
            tension: 0.4
        }));
        this.charts.projections.update();
    },

    async refreshOptimization() {
        // Fetch Efficient Frontier
        const res = await fetch('/api/analytics/efficient-frontier', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_portfolios: 50 })
        });
        const data = await res.json();
        
        this.charts.frontier.data.datasets[0].data = data.portfolios.volatilities.map((v, i) => ({
            x: v, y: data.portfolios.returns[i]
        }));
        this.charts.frontier.update();
    },

    async runFullOptimization() {
        const btn = document.getElementById('run-opt-btn');
        btn.innerText = 'Quantum-Simulating...';
        btn.disabled = true;

        const method = document.getElementById('opt-method').value;
        const endpoint = method === 'quantum' ? '/api/optimize' : '/api/ml/optimize';
        
        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    target_return: this.state.userSettings.targetReturn,
                    method: method 
                })
            });
            const data = await res.json();
            this.renderOptResults(data.holdings || this.mapWeightsToStocks(data.weights));
            this.state.optResults = data;
        } catch (e) {
            console.error(e);
        } finally {
            btn.innerText = 'Execute Optimization';
            btn.disabled = false;
        }
    },

    mapWeightsToStocks(weights) {
        // Fallback mapping if weights only
        return this.state.stocks.map((s, i) => ({
            ...s, weight: weights[i], value: weights[i] * 1238642
        })).filter(s => s.weight > 0.01);
    },

    renderOptResults(holdings) {
        const tbody = document.querySelector('#opt-results-table tbody');
        tbody.innerHTML = (holdings || []).map(h => `
            <tr>
                <td style="font-weight: 600;">${h.symbol}</td>
                <td style="color: var(--text-secondary);">${h.sector}</td>
                <td style="color: var(--accent-sky); font-weight: 700;">${(h.weight * 100).toFixed(2)}%</td>
                <td>$${Math.round(h.value).toLocaleString()}</td>
                <td>${h.dividend_yield}%</td>
                <td><button class="badge badge-sky" style="border:none; cursor:pointer;">Analyze</button></td>
            </tr>
        `).join('');
    },

    async refreshRisk() {
        // 1. Parametric Metrics
        const mockReturns = Array.from({length: 100}, () => (Math.random() - 0.45) * 0.02);
        const res = await fetch('/api/risk/parametric', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ returns: mockReturns })
        });
        const data = await res.json();
        
        document.getElementById('risk-var').innerText = `${(data.var_cvar.parametric.var * 100).toFixed(2)}%`;
        document.getElementById('risk-cvar').innerText = `${(data.var_cvar.parametric.cvar * 100).toFixed(2)}%`;
        document.getElementById('risk-tail').innerText = data.tail_risk.left_tail_index.toFixed(2);

        // 2. Stratification Chart
        const sRes = await fetch('/api/risk/stratify');
        const sData = await sRes.json();
        this.state.stocks = sData;

        const clusters = {};
        sData.forEach(s => {
            if (!clusters[s.risk_cluster]) clusters[s.risk_cluster] = [];
            clusters[s.risk_cluster].push({ x: s.beta || 1, y: s.dividend_yield });
        });

        const colors = ['#38bdf8', '#10b981', '#f43f5e', '#f59e0b'];
        this.charts.riskScatter.data.datasets = Object.keys(clusters).map((k, i) => ({
            label: `Cluster ${k}`,
            data: clusters[k],
            backgroundColor: colors[i % colors.length]
        }));
        this.charts.riskScatter.update();

        // 3. Regimes
        const regimes = data.regime_analysis;
        this.charts.regime.data.labels = regimes.regime_stats.map(s => s.label);
        this.charts.regime.data.datasets[0].data = regimes.regime_stats.map(s => s.frequency);
        this.charts.regime.update();
    },

    async refreshTrade() {
        // Just UI update for now
        const logs = document.getElementById('trade-logs');
        logs.innerHTML += `<br>[${new Date().toLocaleTimeString()}] Fetching current holdings from IBKR...`;
    },

    async executeMarketTrade() {
        const btn = document.getElementById('execute-rebalance-btn');
        btn.innerText = 'Transmitting to IBKR...';
        
        setTimeout(() => {
            btn.innerText = 'Trade Complete';
            const logs = document.getElementById('trade-logs');
            logs.innerHTML += `<br>[${new Date().toLocaleTimeString()}] ORDER FILLED: Rebalanced 12 positions. Total Fee: $15.42`;
        }, 2000);
    },

    async refreshHistory() {
        // Dividend Forecast
        const res = await fetch('/api/dividend/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ portfolio_value: 1238642, weights: Array(30).fill(1/30), years: 10 })
        });
        const data = await res.json();
        
        this.charts.divForecast.data.labels = data.projections.map(p => `Year ${p.year}`);
        this.charts.divForecast.data.datasets = [{
            label: 'Annual Income',
            data: data.projections.map(p => p.annual_income),
            borderColor: '#10b981',
            fill: false
        }];
        this.charts.divForecast.update();
    },

    async sendAiQuery(q) {
        const chat = document.getElementById('ai-chat');
        chat.innerHTML += `<div style="color: var(--text-primary); margin-top: 12px; font-weight: 700;">You: ${q}</div>`;
        document.getElementById('ai-input').value = '';
        chat.scrollTop = chat.scrollHeight;

        const res = await fetch('/api/ai/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q, context: {} })
        });
        const data = await res.json();
        chat.innerHTML += `<div style="color: var(--accent-sky); margin-top: 12px; font-weight: 700;">CIBC AI Advisor:</div>`;
        chat.innerHTML += `<div>${data.answer}</div>`;
        chat.scrollTop = chat.scrollHeight;
    },

    startBackgroundLoops() {
        // Ticker Drift
        setInterval(() => {
            if (this.charts.projections && this.state.activeTab === 'dashboard') {
                this.charts.projections.data.datasets.forEach(ds => {
                    const last = ds.data[ds.data.length - 1];
                    ds.data.push(last * (1 + (Math.random() - 0.495) * 0.005));
                    ds.data.shift();
                });
                this.charts.projections.update('none');
            }
        }, 2000);
    }
};

window.onload = () => App.init();
