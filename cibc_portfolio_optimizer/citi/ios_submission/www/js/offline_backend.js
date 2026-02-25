/**
 * Offline Backend Logic for TraderBot_Can PWA
 * Replicates app.py functionality in pure JavaScript for static hosting.
 */

const APP_DATA = {
    ASSETS: [
        { symbol: "AAPL", name: "Apple Inc.", type: "Stock" },
        { symbol: "MSFT", name: "Microsoft Corp.", type: "Stock" },
        { symbol: "NVDA", name: "NVIDIA Corp.", type: "Stock" },
        { symbol: "TSLA", name: "Tesla Inc.", type: "Stock" },
        { symbol: "AMZN", name: "Amazon.com", type: "Stock" },
        { symbol: "EUR/USD", name: "Euro / US Dollar", type: "Forex" },
        { symbol: "GBP/USD", name: "British Pound / USD", type: "Forex" },
        { symbol: "USD/JPY", name: "US Dollar / Japanese Yen", type: "Forex" },
        { symbol: "USD/CAD", name: "US Dollar / Canadian Dollar", type: "Forex" },
        { symbol: "LITH", name: "Lithium Carbonate", type: "Commodity" },
        { symbol: "COBT", name: "Cobalt Futures", type: "Commodity" },
        { symbol: "URA", name: "Uranium Spot", type: "Commodity" },
        { symbol: "VALE", name: "Vale S.A.", type: "Stock" },
        { symbol: "RIO", name: "Rio Tinto", type: "Stock" },
        { symbol: "BHP", name: "BHP Group", type: "Stock" },
        { symbol: "CCJ", name: "Cameco Corp.", type: "Stock" },
        { symbol: "MP", name: "MP Materials", type: "Stock" }
    ],
    MINERAL_ASSETS: [
        { ticker: "LITH_SPOT", name: "Lithium Carbonate", type: "Commodity", price: 14500.0, vol: 0.45, yield: 0.0 },
        { ticker: "URA_SPOT", name: "Uranium (U3O8)", type: "Commodity", price: 82.50, vol: 0.38, yield: 0.0 },
        { ticker: "RIO", name: "Rio Tinto", type: "Stock", price: 68.50, vol: 0.25, yield: 6.5 },
        { ticker: "BHP", name: "BHP Group", type: "Stock", price: 59.20, vol: 0.22, yield: 5.8 },
        { ticker: "VALE", name: "Vale S.A.", type: "Stock", price: 14.10, vol: 0.35, yield: 7.2 },
        { ticker: "CCJ", name: "Cameco Corp", type: "Stock", price: 48.00, vol: 0.40, yield: 0.5 },
        { ticker: "FCX", name: "Freeport-McMoRan", type: "Stock", price: 42.00, vol: 0.32, yield: 2.1 },
        { ticker: "ALB", name: "Albemarle", type: "Stock", price: 120.00, vol: 0.45, yield: 1.5 }
    ]
};

// --- Math Helpers ---

function gaussianRandom(mean = 0, stdev = 1) {
    const u = 1 - Math.random();
    const v = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdev + mean;
}

// Error Function Polyfill (Abramowitz and Stegun)
function erf(x) {
    const sign = (x >= 0) ? 1 : -1;
    x = Math.abs(x);
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
}

function cdf(x) {
    return (1.0 + erf(x / Math.sqrt(2.0))) / 2.0;
}

// Black-Scholes
function bsPrice(S, K, T, r, sigma, type = 'call') {
    const d1 = (Math.log(S / K) + (r + 0.5 * Math.pow(sigma, 2)) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);

    if (type === 'call') {
        return S * cdf(d1) - K * Math.exp(-r * T) * cdf(d2);
    } else {
        return K * Math.exp(-r * T) * cdf(-d2) - S * cdf(-d1);
    }
}

// --- API Logic Simulation ---

const MockBackend = {
    async getPortfolio() {
        const portfolio = APP_DATA.ASSETS.map(asset => {
            let basePrice = 100.0;
            if (asset.type === 'Forex') basePrice = asset.symbol.includes('JPY') ? 145.0 : 1.05;
            if (asset.type === 'Commodity') basePrice = 25000.0;
            if (asset.type === 'Stock') basePrice = Math.random() * 700 + 50;

            const changePct = gaussianRandom(0, 1.5);
            const currentPrice = basePrice * (1 + changePct / 100);

            // Logic
            const ma50 = basePrice * (1 + gaussianRandom(0, 0.02));
            const score = 50 + changePct * 5 + (currentPrice > ma50 ? 10 : -10);
            const aiConf = Math.min(Math.max(score + gaussianRandom(0, 5), 0), 99);

            let rec = "HOLD";
            if (aiConf > 80) rec = "STRONG BUY";
            else if (aiConf > 60) rec = "BUY";
            else if (aiConf < 30) rec = "SELL";

            return {
                ticker: asset.symbol,
                name: asset.name,
                type: asset.type,
                price: parseFloat(currentPrice.toFixed(asset.type === 'Forex' ? 4 : 2)),
                change: parseFloat(changePct.toFixed(2)),
                stats: {
                    beta: (Math.random() * 1.5).toFixed(2),
                    volatility: asset.type === 'Commodity' ? '8%' : '5%'
                },
                ai_score: aiConf.toFixed(1),
                recommendation: rec,
                suggested_allocation: (aiConf / 2.5).toFixed(1) + '%'
            };
        });

        return portfolio.sort((a, b) => b.ai_score - a.ai_score);
    },

    async getMarketData() {
        // Generate S&P, Lithium, VIX curves
        const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        let sp = 5200;
        let lith = 14500;
        let vix = 14.5;

        const spData = [], lithData = [], vixData = [];

        months.forEach(m => {
            sp += gaussianRandom(35, 40);
            lith += gaussianRandom(200, 350);
            vix = vix * 0.9 + 1.4 + gaussianRandom(0, 0.8);

            spData.push({ month: m, price: Math.round(sp) });
            lithData.push({ month: m, price: Math.round(lith) });
            vixData.push({ month: m, price: vix.toFixed(2) });
        });

        return {
            sp500: spData,
            lithium: lithData,
            vix: vixData,
            meta: {
                sp_target: Math.round(sp),
                lithium_target: Math.round(lith),
                vix_target: vix.toFixed(2)
            }
        };
    },

    async generateMineralStrategy() {
        const rRate = 0.047;
        const T = 0.25; // 3 months

        const portfolio = [];
        const derivatives = [];
        let totalYield = 0;

        APP_DATA.MINERAL_ASSETS.forEach(asset => {
            const spot = asset.price * (1 + gaussianRandom(0, 0.005));
            const callStrike = spot * 1.05;
            const putStrike = spot * 0.95;

            const callPrem = bsPrice(spot, callStrike, T, rRate, asset.vol, 'call');
            const putPrem = bsPrice(spot, putStrike, T, rRate, asset.vol, 'put');

            const optYield = (callPrem / spot) * 4 * 0.8 * 100;
            const effYield = asset.yield + optYield;

            const score = effYield / (asset.vol * 100);
            const rsi = Math.random() * 100;

            derivatives.push({
                ticker: asset.ticker,
                spot_price: '$' + spot.toFixed(2),
                trend: rsi > 50 ? "BULLISH" : "BEARISH",
                call_opt: { strike: '$' + callStrike.toFixed(2), price: '$' + callPrem.toFixed(2) },
                put_opt: { strike: '$' + putStrike.toFixed(2), price: '$' + putPrem.toFixed(2) },
                signals: { entry: '$' + (spot * 0.98).toFixed(2) }
            });

            if (effYield > 12.0) {
                portfolio.push({
                    ticker: asset.ticker,
                    dividend_yield: `${asset.yield}% + ${optYield.toFixed(1)}% = ${effYield.toFixed(1)}%`,
                    _raw_score: score,
                    _eff_yield: effYield
                });
            }
        });

        // allocations
        const totalScore = portfolio.reduce((sum, i) => sum + i._raw_score, 0);
        portfolio.forEach(p => {
            const w = (p._raw_score / totalScore);
            p.allocation = (w * 100).toFixed(1) + '%';
            p.projected_income = '$' + (100000 * w * (p._eff_yield / 100)).toFixed(2) + ' /yr';

            totalYield += w * p._eff_yield;
        });

        return {
            target_yield: totalYield.toFixed(1) + '%',
            strategy_name: "Statistical Yield Maximizer (Offline Mode)",
            optimized_portfolio: portfolio,
            derivatives_chain: derivatives
        };
    },

    async optimizeTrade(ticker) {
        // Feynman Logic Port
        const spot = 450.0;
        const vol = 0.35;
        const bsPrice = spot * Math.exp(0.05);

        // Simulation
        const nPaths = 200;
        const nSteps = 50;
        const dt = 1 / nSteps;
        const finals = [];

        for (let i = 0; i < nPaths; i++) {
            let pathSpot = spot;
            for (let j = 0; j < nSteps; j++) {
                const noise = gaussianRandom(0, Math.sqrt(dt));
                pathSpot *= Math.exp((0.05 - 0.5 * vol * vol) * dt + vol * noise);
            }
            finals.push(pathSpot);
        }

        finals.sort((a, b) => a - b);
        const med = finals[Math.floor(finals.length / 2)];
        const qAlpha = ((med - bsPrice) / bsPrice * 100).toFixed(2);

        // Plot Data
        const plot = [];
        let cv = spot, qv = spot;
        for (let i = 1; i <= 12; i++) {
            cv += (bsPrice - spot) / 12;
            qv += (med - spot) / 12 + gaussianRandom(0, 1.0);
            plot.push({ step: i, bs_val: cv, feynman_val: qv });
        }

        return {
            bs_signature: bsPrice.toFixed(2),
            feynman_integral: med.toFixed(2),
            quantum_alpha: qAlpha + '%',
            optimal_entry: (spot * 0.98).toFixed(2),
            comparative_plot: plot
        };
    },

    async placeOrder(order) {
        await new Promise(r => setTimeout(r, 600));
        return {
            status: "FILLED",
            filled_price: order.price,
            ticker: order.ticker,
            action: order.action
        };
    }
};
