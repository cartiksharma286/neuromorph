let latestModel;

window.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => switchTab(button.dataset.tab)));
  const slider = document.getElementById('interventionLevel');
  slider.addEventListener('input', () => loadModel(Number(slider.value) / 100));
  loadModel(Number(slider.value) / 100);
});

function switchTab(tabName) {
  document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.tab === tabName));
  document.querySelectorAll('.panel').forEach(panel => panel.classList.toggle('active', panel.id === tabName));
}

async function loadModel(interventionLevel) {
  try {
    const response = await fetch('/api/model', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({intervention_level: interventionLevel})
    });
    latestModel = await response.json();
    renderModel(latestModel);
  } catch (error) {
    showToast('Unable to load prevention model.');
    console.error(error);
  }
}

function renderModel(data) {
  const scenario = data.scenario;
  document.getElementById('recordCount').textContent = `${data.data_quality.hate_incident_records.toLocaleString()} hate-crime records`;
  document.getElementById('levelLabel').textContent = `${Math.round(scenario.intervention_level * 100)}%`;
  document.getElementById('riskIndex').textContent = `${scenario.community_risk_index_pct.toFixed(1)}%`;
  document.getElementById('targetStatus').textContent = scenario.target_status;
  document.getElementById('targetStatus').style.color = scenario.community_risk_index_pct < 2 ? '#0c9b82' : '#b9444d';
  document.getElementById('scenarioNote').textContent = scenario.planning_note;
  document.getElementById('limitations').textContent = data.data_quality.limitations;

  document.getElementById('genderBiasCount').textContent = data.women_and_survivor_support.gender_bias_incidents.toLocaleString();
  document.getElementById('womenCount').textContent = data.women_and_survivor_support.women_targeted_incidents.toLocaleString();
  renderProtocols('protocolList', data.protocols);
  renderSimpleList('supportList', data.women_and_survivor_support.protocols);
  renderAllocation(data.division_allocation);
  drawYearChart(data.yearly);
  drawBiasChart(data.bias_categories);
}

function renderProtocols(id, protocols) {
  const root = document.getElementById(id);
  root.innerHTML = protocols.map(protocol => `<div class="protocol"><strong>${escapeHtml(protocol.name)}</strong><span>${escapeHtml(protocol.detail)}</span></div>`).join('');
}

function renderSimpleList(id, entries) {
  const root = document.getElementById(id);
  root.innerHTML = entries.map(entry => `<div class="protocol"><span>${escapeHtml(entry)}</span></div>`).join('');
}

function renderAllocation(rows) {
  const root = document.getElementById('allocationBody');
  root.innerHTML = rows.map(row => `<tr>
    <td><strong>${escapeHtml(row.division)}</strong></td><td>${row.hate_incidents}</td><td>${row.planning_score}</td>
    <td>p = ${row.prime_cycle}</td><td>${row.qml_kernel_feature}</td><td>${row.recommended_prevention_hours} h</td>
    <td>${row.expected_reduction_pct}%</td></tr>`).join('');
}

function drawYearChart(rows) {
  const canvas = document.getElementById('yearChart'); const ctx = canvas.getContext('2d');
  chartBase(ctx, canvas); const {width:w,height:h} = canvas; const max = Math.max(...rows.map(r => r.incidents));
  const left=42, bottom=h-30, chartH=h-55, barW=Math.max(10,(w-65)/rows.length-8);
  rows.forEach((row, index) => {
    const x=left+index*((w-65)/rows.length)+4; const bh=(row.incidents/max)*chartH;
    ctx.fillStyle='#0c9b82'; ctx.fillRect(x,bottom-bh,barW,bh);
    ctx.fillStyle='#63726c'; ctx.font='10px DM Mono'; ctx.fillText(String(row.year).slice(2),x,bottom+15);
  });
  ctx.fillStyle='#63726c'; ctx.font='10px DM Mono'; ctx.fillText('reported incidents',left,14);
}

function drawBiasChart(rows) {
  const canvas = document.getElementById('biasChart'); const ctx = canvas.getContext('2d');
  chartBase(ctx, canvas); const {width:w,height:h} = canvas; const max=Math.max(...rows.map(r=>r.incidents),1);
  const left=155, top=24, rowH=(h-38)/rows.length;
  rows.forEach((row,index)=>{
    const y=top+index*rowH; const bw=(row.incidents/max)*(w-left-35);
    ctx.fillStyle='#63726c';ctx.font='10px Manrope';ctx.textAlign='right';ctx.fillText(row.category,left-10,y+12);
    ctx.fillStyle=index===0?'#f2c14e':'#165a4c';ctx.fillRect(left,y,bw,13);
    ctx.fillStyle='#17231f';ctx.textAlign='left';ctx.fillText(String(row.incidents),left+bw+6,y+11);
  }); ctx.textAlign='left';
}

function chartBase(ctx, canvas) {
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#fbfcfa';ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#dce3dc';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(35,canvas.height-30);ctx.lineTo(canvas.width-15,canvas.height-30);ctx.stroke();
}

function escapeHtml(value) { const node=document.createElement('span'); node.textContent=String(value); return node.innerHTML; }
function showToast(message) { const el=document.getElementById('toast'); el.textContent=message; el.style.display='block'; setTimeout(()=>el.style.display='none',3000); }
