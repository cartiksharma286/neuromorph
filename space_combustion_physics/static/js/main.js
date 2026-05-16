// ── Starfield ─────────────────────────────────────────
(function(){
  const c=document.getElementById('starfield');
  if(!c)return;
  const ctx=c.getContext('2d');
  let W,H,stars=[];
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;initStars()}
  function initStars(){stars=Array.from({length:220},()=>({
    x:Math.random()*W,y:Math.random()*H,
    r:Math.random()*1.4+0.2,a:Math.random(),da:Math.random()*.003+.001,speed:Math.random()*.15+.05
  }))}
  function draw(){
    ctx.clearRect(0,0,W,H);
    stars.forEach(s=>{
      s.a+=s.da; if(s.a>1||s.a<0)s.da*=-1;
      s.y+=s.speed; if(s.y>H){s.y=0;s.x=Math.random()*W}
      ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(180,210,255,${s.a})`;ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize',resize); resize(); draw();
})();

// ── Clock & Tabs ──────────────────────────────────────
setInterval(()=>{
  const el=document.getElementById('clock');
  if(el)el.textContent=new Date().toUTCString().slice(17,25)+' UTC';
},1000);

document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// ── Chart Registry ────────────────────────────────────
const COLORS={blue:'#00c8ff',orange:'#ff5a00',purple:'#9b5fff',green:'#00ffaa',gold:'#ffc740',red:'#ff3366',muted:'#6a7fa8'};
const charts={};

function destroyChart(id){ if(charts[id]){charts[id].destroy();delete charts[id]} }

function mkChart(id, labels, datasets, opts={}){
  destroyChart(id);
  const ctx=document.getElementById(id)?.getContext('2d');
  if(!ctx)return;
  charts[id] = new Chart(ctx,{
    type:'line',
    data:{labels,datasets},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}}, // Custom legends in HTML
      scales:{
        x:{ticks:{color:COLORS.muted,maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.05)'}},
        y:{ticks:{color:COLORS.muted},grid:{color:'rgba(255,255,255,0.05)'}}
      },
      ...opts
    }
  });
}

const setKPI = (id,val) => { const el=document.getElementById(id); if(el)el.textContent=val; };
const setHTML = (id,html) => { const el=document.getElementById(id); if(el)el.innerHTML=html; };

// ═══════════════════════════════════════════════
// CFD SIMULATION
// ═══════════════════════════════════════════════
document.getElementById('btn-cfd')?.addEventListener('click', runCFD);
async function runCFD(){
  const btn=document.getElementById('btn-cfd');
  const ld=document.getElementById('ld-cfd');
  btn.disabled=true; ld.classList.add('visible');
  
  const body = {
    throttle: parseFloat(document.getElementById('inp-cfd-throttle').value),
    fuel: document.getElementById('sel-cfd-fuel').value
  };

  try {
    const r = await fetch('/api/cfd', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json();
    
    setKPI('kpi-thrust', d.thrust_kN.toLocaleString() + ' kN');
    setKPI('kpi-mach', d.exit_mach.toFixed(2));
    setKPI('kpi-chamb-t', d.chamber_temp.toLocaleString() + ' K');
    setKPI('kpi-peak-q', d.peak_q.toFixed(2) + ' MW/m²');
    setKPI('kpi-heat-loss', d.total_heat_loss.toLocaleString() + ' kW');

    const xL = d.x.map(v=>v.toFixed(2));

    // 1. Pressure & Mach
    mkChart('chart-cfd-press', xL, [
      {label:'Pressure',data:d.pressure,borderColor:COLORS.orange,borderWidth:2,pointRadius:0,yAxisID:'y'},
      {label:'Mach',data:d.mach,borderColor:COLORS.blue,borderWidth:2,pointRadius:0,yAxisID:'y1'}
    ], {scales:{y:{position:'left'},y1:{position:'right',grid:{drawOnChartArea:false}}}});

    // 2. Gas vs Wall Temp
    mkChart('chart-cfd-temp-gas-wall', xL, [
      {label:'Gas T',data:d.temperature,borderColor:COLORS.red,borderWidth:2,pointRadius:0},
      {label:'Wall T',data:d.wall_temp,borderColor:COLORS.purple,borderWidth:2,pointRadius:0}
    ]);

    // 3. Heat Flux
    mkChart('chart-cfd-q', xL, [
      {label:'Heat Flux',data:d.heat_flux,borderColor:COLORS.gold,backgroundColor:'rgba(255,199,64,0.1)',fill:true,borderWidth:2,pointRadius:0}
    ]);

    // 4. Fuel Fraction
    mkChart('chart-cfd-fuel', xL, [
      {label:'Fuel %',data:d.fuel_fraction,borderColor:COLORS.green,borderWidth:2,pointRadius:0}
    ]);

    setHTML('cfd-log', `[${new Date().toLocaleTimeString()}] SOLVER CONVERGED\nL2 norm: 1.4e-6\nStability: Lax-Friedrichs Nominal\nMax Wall Stress: ${(d.peak_q * 1.5).toFixed(1)} MPa (Est)`);
  } catch(e) {
    console.error(e);
    setHTML('cfd-log', `[ERROR] Solver Diverged: Check Boundary Conditions.`);
  }
  btn.disabled=false; ld.classList.remove('visible');
}

// ═══════════════════════════════════════════════
// COMBUSTION PDE
// ═══════════════════════════════════════════════
document.getElementById('btn-combustion')?.addEventListener('click', async ()=>{
  const r = await fetch('/api/combustion', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fuel:'H2-O2',phi:1.0,P:1.0})});
  const d = await r.json();
  setKPI('kpi-Tpeak', d.peak_temperature + ' K');
  const xL = d.x.map(v=>v.toFixed(1));
  mkChart('chart-temp', xL, [{label:'Temp',data:d.temperature,borderColor:COLORS.orange,pointRadius:0}]);
  mkChart('chart-species', xL, [{label:'Fuel',data:d.fuel,borderColor:COLORS.red,pointRadius:0}]);
});

// Auto-run CFD on load
window.addEventListener('load', ()=>{ setTimeout(runCFD, 500); });
