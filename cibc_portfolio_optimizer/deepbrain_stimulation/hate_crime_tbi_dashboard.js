/**
 * hate_crime_tbi_dashboard.js
 * Frontend dashboard for TBI Repair (Hate Crime) DBS Simulation
 * Pattern follows asd_dashboard.js / sad_dashboard.js
 */

(function () {
    'use strict';

    const API_BASE = '';  // same origin

    // ── Helpers ─────────────────────────────────────────────────
    function post(url, body) {
        return fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        }).then(r => r.json());
    }
    function get(url) {
        return fetch(url).then(r => r.json());
    }

    function setEl(id, html) {
        const el = document.getElementById(id);
        if (el) el.innerHTML = html;
    }
    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }
    function setStyle(id, prop, val) {
        const el = document.getElementById(id);
        if (el) el.style[prop] = val;
    }

    function showLoading(btnId, msg = 'Running…') {
        const btn = document.getElementById(btnId);
        if (btn) { btn._origText = btn.innerHTML; btn.innerHTML = `⏳ ${msg}`; btn.disabled = true; }
    }
    function stopLoading(btnId) {
        const btn = document.getElementById(btnId);
        if (btn && btn._origText) { btn.innerHTML = btn._origText; btn.disabled = false; }
    }

    function triColor(tri) {
        if (tri >= 0.75) return '#34c759';   // green
        if (tri >= 0.50) return '#ffd60a';   // yellow
        if (tri >= 0.30) return '#ff9f0a';   // orange
        return '#ff453a';                     // red
    }

    function goseColor(g) {
        if (g >= 7) return '#34c759';
        if (g >= 5) return '#ffd60a';
        if (g >= 3) return '#ff9f0a';
        return '#ff453a';
    }

    // ─────────────────────────────────────────────────────────────
    // Recovery Trajectory Canvas
    // ─────────────────────────────────────────────────────────────
    function drawRecoveryChart(canvasId, weeklyResults) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        if (!weeklyResults || weeklyResults.length === 0) return;

        const pad = { left: 55, right: 20, top: 20, bottom: 45 };
        const chartW = W - pad.left - pad.right;
        const chartH = H - pad.top - pad.bottom;

        const triValues = weeklyResults.map(w => w.tri);
        const overlayVals = weeklyResults.map(w => w.trauma_overlay);
        const inflamVals = weeklyResults.map(w => w.neuroinflammation);
        const weeks = weeklyResults.map(w => w.week);
        const maxWeek = weeks[weeks.length - 1];

        function xPos(week) { return pad.left + ((week - 1) / (maxWeek - 1)) * chartW; }
        function yPos(val, min = 0, max = 1) { return pad.top + chartH - ((val - min) / (max - min)) * chartH; }

        // Background grid
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = pad.top + (i / 5) * chartH;
            ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + chartW, y); ctx.stroke();
            ctx.fillStyle = '#888'; ctx.font = '10px Inter';
            ctx.textAlign = 'right';
            ctx.fillText((1 - i / 5).toFixed(1), pad.left - 6, y + 4);
        }

        // X-axis labels
        ctx.fillStyle = '#888'; ctx.font = '10px Inter'; ctx.textAlign = 'center';
        weeks.forEach((w, i) => {
            if (i % Math.ceil(weeks.length / 6) === 0 || i === weeks.length - 1) {
                ctx.fillText(`Wk ${w}`, xPos(w), H - 8);
            }
        });

        // Draw neuroinflammation (dashed, orange)
        ctx.strokeStyle = 'rgba(255,159,10,0.55)';
        ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
        ctx.beginPath();
        inflamVals.forEach((v, i) => {
            const x = xPos(weeks[i]), y = yPos(v);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke(); ctx.setLineDash([]);

        // Draw trauma overlay (dashed, magenta)
        ctx.strokeStyle = 'rgba(255,55,120,0.55)';
        ctx.lineWidth = 1.5; ctx.setLineDash([3, 5]);
        ctx.beginPath();
        overlayVals.forEach((v, i) => {
            const x = xPos(weeks[i]), y = yPos(v);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke(); ctx.setLineDash([]);

        // Draw TRI (solid gradient)
        const gradient = ctx.createLinearGradient(pad.left, 0, pad.left + chartW, 0);
        gradient.addColorStop(0, '#ff453a');
        gradient.addColorStop(0.5, '#ffd60a');
        gradient.addColorStop(1, '#34c759');
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 3;
        ctx.beginPath();
        triValues.forEach((v, i) => {
            const x = xPos(weeks[i]), y = yPos(v);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Draw dots for TRI
        triValues.forEach((v, i) => {
            if (i % 2 === 0 || i === triValues.length - 1) {
                ctx.beginPath();
                ctx.arc(xPos(weeks[i]), yPos(v), 4, 0, 2 * Math.PI);
                ctx.fillStyle = triColor(v);
                ctx.fill();
            }
        });

        // Legend
        const legendItems = [
            { color: '#34c759', label: 'TRI', dash: false },
            { color: 'rgba(255,55,120,0.8)', label: 'Trauma Overlay', dash: true },
            { color: 'rgba(255,159,10,0.8)', label: 'Neuroinflammation', dash: true },
        ];
        legendItems.forEach((item, i) => {
            const lx = pad.left + 10 + i * 130;
            const ly = pad.top + 8;
            ctx.setLineDash(item.dash ? [4, 4] : []);
            ctx.strokeStyle = item.color;
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 22, ly); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#ccc'; ctx.font = '10px Inter';
            ctx.textAlign = 'left';
            ctx.fillText(item.label, lx + 26, ly + 4);
        });
    }

    // ─────────────────────────────────────────────────────────────
    // Clinical Trial Population Chart
    // ─────────────────────────────────────────────────────────────
    function drawTrialChart(canvasId, preTris, postTris) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        if (!preTris || preTris.length === 0) return;

        const bins = 12;
        const minV = 0, maxV = 1;
        const binW = (maxV - minV) / bins;

        function histogram(vals) {
            const counts = new Array(bins).fill(0);
            vals.forEach(v => {
                const b = Math.min(bins - 1, Math.floor((v - minV) / binW));
                counts[b]++;
            });
            return counts;
        }

        const preH = histogram(preTris);
        const postH = histogram(postTris);
        const maxCount = Math.max(...preH, ...postH, 1);

        const pad = { left: 40, right: 20, top: 20, bottom: 35 };
        const chartW = W - pad.left - pad.right;
        const chartH = H - pad.top - pad.bottom;
        const bw = chartW / bins / 2;

        // Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = pad.top + (i / 5) * chartH;
            ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + chartW, y); ctx.stroke();
            ctx.fillStyle = '#888'; ctx.font = '9px Inter'; ctx.textAlign = 'right';
            ctx.fillText(Math.round((1 - i / 5) * maxCount), pad.left - 4, y + 3);
        }

        // x-axis
        ctx.fillStyle = '#888'; ctx.font = '9px Inter'; ctx.textAlign = 'center';
        for (let b = 0; b < bins; b += 2) {
            const x = pad.left + b * (chartW / bins) + chartW / (bins * 2);
            ctx.fillText((minV + b * binW).toFixed(1), x, H - 6);
        }

        // Bars
        preH.forEach((count, b) => {
            const x = pad.left + b * (chartW / bins);
            const barH = (count / maxCount) * chartH;
            ctx.fillStyle = 'rgba(255,69,58,0.65)';
            ctx.fillRect(x + 1, pad.top + chartH - barH, bw - 1, barH);
        });
        postH.forEach((count, b) => {
            const x = pad.left + b * (chartW / bins) + bw;
            const barH = (count / maxCount) * chartH;
            ctx.fillStyle = 'rgba(52,199,89,0.65)';
            ctx.fillRect(x + 1, pad.top + chartH - barH, bw - 1, barH);
        });

        // Legend
        ctx.fillStyle = 'rgba(255,69,58,0.7)'; ctx.fillRect(pad.left, pad.top, 12, 10);
        ctx.fillStyle = '#ccc'; ctx.font = '10px Inter'; ctx.textAlign = 'left';
        ctx.fillText('Pre-DBS TRI', pad.left + 16, pad.top + 9);
        ctx.fillStyle = 'rgba(52,199,89,0.7)'; ctx.fillRect(pad.left + 110, pad.top, 12, 10);
        ctx.fillStyle = '#ccc';
        ctx.fillText('Post-DBS TRI', pad.left + 126, pad.top + 9);
    }

    // ─────────────────────────────────────────────────────────────
    // Biomarkers panel
    // ─────────────────────────────────────────────────────────────
    function renderBiomarkers(biomarkers) {
        const items = [
            { key: 'heart_rate_variability_ms', label: 'HRV', unit: 'ms', good: v => v > 60 },
            { key: 'cortisol_ug_dl', label: 'Cortisol', unit: 'μg/dL', good: v => v < 20 },
            { key: 'bdnf_ng_ml', label: 'BDNF', unit: 'ng/mL', good: v => v > 25 },
            { key: 'serotonin_ng_ml', label: 'Serotonin', unit: 'ng/mL', good: v => v > 100 },
            { key: 'axonal_integrity_index', label: 'Axonal Integrity', unit: '', good: v => v > 0.6 },
            { key: 'neuroinflammation_index', label: 'Neuroinflammation', unit: '', good: v => v < 0.4 },
            { key: 'icp_proxy_mmhg', label: 'ICP Proxy', unit: 'mmHg', good: v => v < 15 },
            { key: 'gcs', label: 'GCS Score', unit: '/15', good: v => v >= 13 },
            { key: 'trauma_overlay', label: 'Trauma Severity', unit: '', good: v => v < 0.4 },
        ];
        const html = items.map(item => {
            const val = biomarkers[item.key];
            if (val === undefined) return '';
            const good = item.good(val);
            const color = good ? '#34c759' : '#ff9f0a';
            return `
            <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;border-left:3px solid ${color};">
                <span style="color:#aaa;font-size:0.85em;">${item.label}</span>
                <span style="color:${color};font-size:1.05em;font-weight:600;font-family:monospace;">${typeof val === 'number' ? val.toFixed(val < 2 ? 3 : 1) : val}${item.unit}</span>
            </div>`;
        }).join('');
        setEl('tbioBiomarkersGrid', html);
    }

    // ─────────────────────────────────────────────────────────────
    // Activity bar
    // ─────────────────────────────────────────────────────────────
    function renderActivityBars(activity) {
        const regionLabels = {
            thalamus: 'CM-Pf Thalamus', dlPFC: 'dlPFC', hippocampus: 'Hippocampus',
            amygdala: 'Amygdala (Trauma)', acc: 'ACC', brainstem: 'Brainstem RAS', raphe: 'Raphe (5-HT)',
        };
        const html = Object.entries(activity).map(([key, val]) => {
            const label = regionLabels[key] || key;
            const pct = Math.min(100, val * 100).toFixed(0);
            const healthy = key === 'amygdala' ? val <= 1.0 : val >= 0.70;
            const color = healthy ? '#34c759' : val > 1.0 ? '#ff453a' : '#ff9f0a';
            return `
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;font-size:0.82em;color:#aaa;">
                    <span>${label}</span>
                    <span style="color:${color};">${(val * 100).toFixed(1)}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:7px;overflow:hidden;">
                    <div style="width:${pct}%;height:100%;background:${color};border-radius:4px;transition:width 0.5s;"></div>
                </div>
            </div>`;
        }).join('');
        setEl('tbiActivityBars', html);
    }

    // ─────────────────────────────────────────────────────────────
    // Main init
    // ─────────────────────────────────────────────────────────────
    function initTBIDashboard() {

        // ── Simulate button ──────────────────────────────────────
        const simBtn = document.getElementById('tbiSimulateBtn');
        if (simBtn) {
            simBtn.addEventListener('click', async () => {
                showLoading('tbiSimulateBtn', 'Running DBS…');
                try {
                    const payload = {
                        severity: document.getElementById('tbiSeverity')?.value || 'moderate',
                        injury_type: document.getElementById('tbiInjuryType')?.value || 'blunt',
                        target_region: document.getElementById('tbiTarget')?.value || 'thalamus',
                        frequency_hz: parseFloat(document.getElementById('tbiFrequency')?.value || 130),
                        amplitude_ma: parseFloat(document.getElementById('tbiAmplitude')?.value || 3.0),
                        pulse_width_us: parseFloat(document.getElementById('tbiPulseWidth')?.value || 90),
                        duration_s: 1.0,
                    };
                    const data = await post('/api/tbi/simulate', payload);
                    if (data.success) {
                        const r = data.result;
                        setText('tbiTRI', r.tri.toFixed(3));
                        setStyle('tbiTRI', 'color', triColor(r.tri));
                        setText('tbiGOSE', `${r.gose} — ${r.gose_label}`);
                        setStyle('tbiGOSE', 'color', goseColor(r.gose));
                        setText('tbiStimEffect', (r.stimulation_effect * 100).toFixed(1) + '%');
                        renderActivityBars(r.activity);
                        renderBiomarkers(r.metrics);
                        // Update ICP and GCS inline
                        setText('tbiICP', r.metrics.icp_proxy_mmhg + ' mmHg');
                        setText('tbiGCS', r.metrics.gcs + '/15');
                    } else {
                        alert('Simulation error: ' + data.error);
                    }
                } catch (e) { alert('Network error: ' + e.message); }
                finally { stopLoading('tbiSimulateBtn'); }
            });
        }

        // ── Predict button ───────────────────────────────────────
        const predBtn = document.getElementById('tbiPredictBtn');
        if (predBtn) {
            predBtn.addEventListener('click', async () => {
                showLoading('tbiPredictBtn', 'Predicting…');
                try {
                    const payload = {
                        severity: document.getElementById('tbiSeverity')?.value || 'moderate',
                        injury_type: document.getElementById('tbiInjuryType')?.value || 'blunt',
                        target_region: document.getElementById('tbiTarget')?.value || 'thalamus',
                        frequency_hz: parseFloat(document.getElementById('tbiFrequency')?.value || 130),
                        amplitude_ma: parseFloat(document.getElementById('tbiAmplitude')?.value || 3.0),
                        pulse_width_us: parseFloat(document.getElementById('tbiPulseWidth')?.value || 90),
                        treatment_weeks: parseInt(document.getElementById('tbiWeeks')?.value || 12),
                    };
                    const data = await post('/api/tbi/predict', payload);
                    if (data.success) {
                        const p = data.prediction;
                        drawRecoveryChart('tbiRecoveryCanvas', p.weekly_results);
                        setText('tbiPredInitTRI', p.initial_tri.toFixed(3));
                        setText('tbiPredFinalTRI', p.final_tri.toFixed(3));
                        setStyle('tbiPredFinalTRI', 'color', triColor(p.final_tri));
                        setText('tbiPredResp', (p.response_rate * 100).toFixed(1) + '%');
                        setText('tbiPredFinalGOSE', p.final_gose + ' — ' + p.final_gose_label);
                        setStyle('tbiPredFinalGOSE', 'color', goseColor(p.final_gose));

                        const respBadge = document.getElementById('tbiResponderBadge');
                        if (respBadge) {
                            respBadge.textContent = p.responder ? '✔ RESPONDER' : '✘ NON-RESPONDER';
                            respBadge.style.color = p.responder ? '#34c759' : '#ff453a';
                        }
                    } else {
                        alert('Prediction error: ' + data.error);
                    }
                } catch (e) { alert('Network error: ' + e.message); }
                finally { stopLoading('tbiPredictBtn'); }
            });
        }

        // ── Clinical Trial button ────────────────────────────────
        const trialBtn = document.getElementById('tbiTrialBtn');
        if (trialBtn) {
            trialBtn.addEventListener('click', async () => {
                showLoading('tbiTrialBtn', 'Running trial…');
                try {
                    const payload = {
                        n_subjects: parseInt(document.getElementById('tbiTrialN')?.value || 30),
                        target_region: document.getElementById('tbiTarget')?.value || 'thalamus',
                        frequency_hz: parseFloat(document.getElementById('tbiFrequency')?.value || 130),
                        amplitude_ma: parseFloat(document.getElementById('tbiAmplitude')?.value || 3.0),
                        treatment_weeks: parseInt(document.getElementById('tbiWeeks')?.value || 8),
                    };
                    const data = await post('/api/tbi/trial', payload);
                    if (data.success) {
                        const t = data.trial;
                        setText('tbiTrialN', t.n_subjects);
                        setText('tbiTrialPreTRI', t.pre_tri_mean.toFixed(3));
                        setText('tbiTrialPostTRI', t.post_tri_mean.toFixed(3));
                        setStyle('tbiTrialPostTRI', 'color', triColor(t.post_tri_mean));
                        setText('tbiTrialResponderRate', (t.responder_rate * 100).toFixed(1) + '%');
                        setText('tbiTrialPValue', t.p_value.toFixed(6));
                        const sigEl = document.getElementById('tbiTrialSig');
                        if (sigEl) {
                            sigEl.textContent = t.significant ? '✔ Statistically Significant (p < 0.05)' : '✘ Not Significant (p ≥ 0.05)';
                            sigEl.style.color = t.significant ? '#34c759' : '#ff453a';
                        }
                        setText('tbiTrialPreGOSE', t.pre_gose_mean.toFixed(1));
                        setText('tbiTrialPostGOSE', t.post_gose_mean.toFixed(1));
                        drawTrialChart('tbiTrialCanvas', t.pre_tris, t.post_tris);
                        document.getElementById('tbiTrialResults').style.display = 'block';
                    } else {
                        alert('Trial error: ' + data.error);
                    }
                } catch (e) { alert('Network error: ' + e.message); }
                finally { stopLoading('tbiTrialBtn'); }
            });
        }

        // ── Optimize button ──────────────────────────────────────
        const optBtn = document.getElementById('tbiOptimizeBtn');
        if (optBtn) {
            optBtn.addEventListener('click', async () => {
                showLoading('tbiOptimizeBtn', 'Optimizing…');
                try {
                    const payload = {
                        severity: document.getElementById('tbiSeverity')?.value || 'moderate',
                        injury_type: document.getElementById('tbiInjuryType')?.value || 'blunt',
                        target_region: document.getElementById('tbiTarget')?.value || 'thalamus',
                    };
                    const data = await post('/api/tbi/optimize', payload);
                    if (data.success) {
                        const opt = data.optimization;
                        const bp = opt.best_parameters;
                        setText('tbiOptFreq', bp.frequency_hz + ' Hz');
                        setText('tbiOptAmp', bp.amplitude_ma + ' mA');
                        setText('tbiOptPW', bp.pulse_width_us + ' μs');
                        setText('tbiOptBestTRI', opt.best_tri.toFixed(3));
                        setStyle('tbiOptBestTRI', 'color', triColor(opt.best_tri));

                        const topHtml = opt.top_results.slice(0, 5).map((r, i) =>
                            `<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.85em;color:#ccc;">
                                <span>#${i + 1}</span>
                                <span>${r.frequency_hz} Hz / ${r.amplitude_ma} mA / ${r.pulse_width_us} μs</span>
                                <span style="color:${triColor(r.tri)};">TRI ${r.tri.toFixed(3)}</span>
                            </div>`
                        ).join('');
                        setEl('tbiOptTopResults', topHtml);
                        document.getElementById('tbiOptResults').style.display = 'block';
                    } else {
                        alert('Optimization error: ' + data.error);
                    }
                } catch (e) { alert('Network error: ' + e.message); }
                finally { stopLoading('tbiOptimizeBtn'); }
            });
        }
    }

    // Attach on DOMContentLoaded or immediately if already loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTBIDashboard);
    } else {
        initTBIDashboard();
    }

})();
