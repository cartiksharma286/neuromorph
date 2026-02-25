const API_URL = 'http://localhost:5005/api';

// Format Currency
const formatUSD = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });

// Init
document.addEventListener('DOMContentLoaded', async () => {
    loadMarketContext();
    loadNVQData();

    // Refresh NVQ every 30s
    setInterval(loadNVQData, 30000);
});

async function loadMarketContext() {
    try {
        const res = await fetch(`${API_URL}/init`);
        const data = await res.json();

        document.getElementById('spxTrend').textContent = data.market_context.sp500_trend;
        document.getElementById('vixVal').textContent = data.market_context.vix;

        // Color code trend
        const trendEl = document.getElementById('spxTrend');
        trendEl.style.color = data.market_context.sp500_trend === 'Bullish' ? '#10b981' : '#ef4444';
    } catch (e) { console.error(e); }
}

async function loadNVQData() {
    try {
        const res = await fetch(`${API_URL}/nvq/minerals`);
        const data = await res.json();

        const tbody = document.querySelector('#nvqTable tbody');
        if (!data.length) return;

        tbody.innerHTML = '';
        data.forEach(ore => {
            const tr = document.createElement('tr');

            // Generate Signal Badge
            let sigColor = '#94a3b8';
            if (ore.nvq_signal === 'BUY') sigColor = '#10b981';
            if (ore.nvq_signal === 'SELL') sigColor = '#ef4444';

            tr.innerHTML = `
                <td style="font-weight:600">${ore.symbol}</td>
                <td>${formatUSD.format(ore.prices.Spot)}</td>
                <td style="color:${sigColor}; font-weight:bold">${ore.nvq_signal}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) { console.error(e); }
}

async function runOptimization() {
    const btn = document.querySelector('.btn-primary');
    const risk = document.getElementById('riskProfile').value;

    btn.textContent = "Processing (Quantum/Ridge)...";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_URL}/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ risk })
        });
        const data = await res.json();

        const tbody = document.getElementById('holdingsBody');
        tbody.innerHTML = '';

        data.allocations.forEach(alloc => {
            if (alloc.weight < 0.1) return; // Filter dust

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>
                    <div style="font-weight:600; color:#fff">${alloc.symbol}</div>
                    <div style="font-size:0.8rem; color:#64748b">${alloc.name}</div>
                </td>
                <td><span class="badge" style="background:#0f172a; border-color:#334155">${alloc.sector}</span></td>
                <td style="color:${alloc.ml_forecast > 0 ? '#10b981' : '#94a3b8'}">
                    ${alloc.ml_forecast > 0 ? '+' : ''}${alloc.ml_forecast}%
                </td>
                <td style="font-weight:bold; color:var(--accent-cyan)">${alloc.weight}%</td>
                <td>${formatUSD.format(alloc.value)}</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (e) {
        alert("Optimization Failed");
        console.error(e);
    } finally {
        btn.textContent = "Run Statistical Optimization";
        btn.disabled = false;
    }
}
