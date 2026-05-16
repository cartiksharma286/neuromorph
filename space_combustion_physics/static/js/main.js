// ── Global Error Handling & Starfield ─────────────────
window.onerror=(msg,url,ln)=>console.error('JS Error:',msg,'at',url,':',ln);

(function(){
  const c=document.getElementById('starfield');
  if(!c)return;
  const ctx=c.getContext('2d');
  let W,H,stars=[];
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;initStars()}
  function initStars(){stars=Array.from({length:220},()=>({
    x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.4+0.2,a:Math.random(),da:Math.random()*.003+.001,speed:Math.random()*.15+.05
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
    const p=document.getElementById(btn.dataset.tab);
    if(p)p.classList.add('active');
  });
});

// ── Chart Helpers ─────────────────────────────────────
const COLORS={blue:'#00c8ff',orange:'#ff5a00',purple:'#9b5fff',green:'#00ffaa',gold:'#ffc740',red:'#ff3366',muted:'#6a7fa8'};
const charts={};

function destroyChart(id){ if(charts[id]){charts[id].destroy();delete charts[id]} }

function mkChart(id, labels, datasets, opts={}){
  destroyChart(id);
  const ctx=document.getElementById(id)?.getContext('2d');
  if(!ctx)return;
  charts[id]=new Chart(ctx,{
    type:opts.type||'line',
    data:{labels,datasets},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:labels?true:false,labels:{color:COLORS.muted,font:{size:10}}}},
      scales:opts.type==='doughnut'?{}:{
        x:{ticks:{color:COLORS.muted,maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.05)'}},
        y:{ticks:{color:COLORS.muted},grid:{color:'rgba(255,255,255,0.05)'}}
      },
      ...opts
    }
  });
}

const setKPI=(id,v)=>{const el=document.getElementById(id);if(el)el.textContent=v};
const setHTML=(id,h)=>{const el=document.getElementById(id);if(el)el.innerHTML=h};

// ═══════════════════════════════════════════════
// 1. CFD SIMULATION
// ═══════════════════════════════════════════════
document.getElementById('btn-cfd')?.addEventListener('click', runCFD);
async function runCFD(){
  const btn=document.getElementById('btn-cfd');
  const ld=document.getElementById('ld-cfd');
  btn.disabled=true; ld.classList.add('visible');
  try{
    const r=await fetch('/api/cfd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({throttle:parseFloat(document.getElementById('inp-cfd-throttle').value),fuel:document.getElementById('sel-cfd-fuel').value})});
    const d=await r.json();
    setKPI('kpi-thrust',d.thrust_kN+' kN'); setKPI('kpi-mach',d.exit_mach); setKPI('kpi-chamb-t',d.chamber_temp+' K'); setKPI('kpi-peak-q',d.peak_q+' MW/m²');
    const xL=d.x.map(v=>v.toFixed(2));
    mkChart('chart-cfd-press',xL,[{label:'Pressure',data:d.pressure,borderColor:COLORS.orange,yAxisID:'y'},{label:'Mach',data:d.mach,borderColor:COLORS.blue,yAxisID:'y1'}],{scales:{y:{position:'left'},y1:{position:'right',grid:{drawOnChartArea:false}}}});
    mkChart('chart-cfd-temp-gas-wall',xL,[{label:'Gas T',data:d.temperature,borderColor:COLORS.red},{label:'Wall T',data:d.wall_temp,borderColor:COLORS.purple}]);
    mkChart('chart-cfd-q',xL,[{label:'Heat Flux',data:d.heat_flux,borderColor:COLORS.gold,fill:true,backgroundColor:'rgba(255,199,64,0.1)'}]);
    mkChart('chart-cfd-fuel',xL,[{label:'Fuel %',data:d.fuel_fraction,borderColor:COLORS.green}]);
  }catch(e){console.error(e)}
  btn.disabled=false; ld.classList.remove('visible');
}

// ═══════════════════════════════════════════════
// 2. COMBUSTION PDE
// ═══════════════════════════════════════════════
document.getElementById('btn-combustion')?.addEventListener('click', runCombustion);
async function runCombustion(){
  const btn=document.getElementById('btn-combustion');
  const ld=document.getElementById('ld-combustion');
  btn.disabled=true; ld.classList.add('visible');
  try{
    const r=await fetch('/api/combustion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fuel:document.getElementById('sel-fuel').value,phi:parseFloat(document.getElementById('inp-phi').value),P:1.0})});
    const d=await r.json();
    setKPI('kpi-Tpeak',d.peak_temperature+' K'); setKPI('kpi-SL',d.flame_speed+' m/s'); setKPI('kpi-eta',d.combustion_efficiency+' %');
    const xL=d.x.map(v=>v.toFixed(1));
    mkChart('chart-temp',xL,[{label:'Temperature',data:d.temperature,borderColor:COLORS.orange,fill:true,backgroundColor:'rgba(255,90,0,0.1)'}]);
    mkChart('chart-species',xL,[{label:'Fuel',data:d.fuel,borderColor:COLORS.red},{label:'Oxidizer',data:d.oxidizer,borderColor:COLORS.blue},{label:'Product',data:d.products,borderColor:COLORS.green}]);
  }catch(e){console.error(e)}
  btn.disabled=false; ld.classList.remove('visible');
}

// ═══════════════════════════════════════════════
// 3. THROTTLE CONTROL
// ═══════════════════════════════════════════════
document.getElementById('btn-throttle')?.addEventListener('click', runThrottle);
async function runThrottle(){
  const btn=document.getElementById('btn-throttle');
  btn.disabled=true;
  try{
    const r=await fetch('/api/throttle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({Isp:parseFloat(document.getElementById('inp-Isp').value),m0:parseFloat(document.getElementById('inp-m0').value),mode:document.getElementById('sel-mode').value})});
    const d=await r.json();
    setKPI('kpi-dv',d.delta_v_ideal+' m/s'); setKPI('kpi-fc',(d.fuel_consumed_kg/1000).toFixed(1)+' t'); setKPI('kpi-vf',d.final_velocity_ms+' m/s');
    const tL=d.time.map(v=>v.toFixed(0));
    mkChart('chart-throttle',tL,[{label:'Throttle',data:d.throttle,borderColor:COLORS.orange,stepped:true}]);
    mkChart('chart-vel',tL,[{label:'Velocity',data:d.velocity,borderColor:COLORS.blue,yAxisID:'y'},{label:'Altitude',data:d.altitude_km,borderColor:COLORS.green,yAxisID:'y1'}],{scales:{y:{position:'left'},y1:{position:'right',grid:{drawOnChartArea:false}}}});
  }catch(e){console.error(e)}
  btn.disabled=false;
}

// ═══════════════════════════════════════════════
// 4. TRAJECTORY (HPC UPGRADE)
// ═══════════════════════════════════════════════
document.getElementById('btn-traj')?.addEventListener('click', runTraj);
async function runTraj(){
  const btn=document.getElementById('btn-traj');
  btn.disabled=true;
  try{
    const body = {
      vehicle: document.getElementById('sel-vehicle').value,
      orbit: document.getElementById('sel-orbit').value,
      payload: parseFloat(document.getElementById('inp-payload').value)
    };
    const r=await fetch('/api/trajectory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    
    setKPI('kpi-alt-traj',d.max_altitude_km+' km'); 
    setKPI('kpi-dvach',d.delta_v_achieved+' m/s');
    setKPI('kpi-maxq',d.max_q_kpa+' kPa');
    
    setHTML('orbit-status',d.orbit_achieved?'<span class="tag tag-green">Orbit Achieved</span>':'<span class="tag tag-red">Suborbital</span>');
    
    // Trajectory Path
    mkChart('chart-traj',null,[{
      label:'Flight Path (HPC)',
      data:d.x_km.map((x,i)=>({x,y:d.z_km[i]})),
      borderColor:COLORS.blue,
      showLine:true,
      pointRadius:1
    }],{type:'scatter',scales:{x:{title:{display:true,text:'Downrange (km)'}},y:{title:{display:true,text:'Altitude (km)'}}}});

    // Speed & Dynamic Pressure
    const tL=d.time.map(v=>v.toFixed(0));
    mkChart('chart-speed',tL,[
      {label:'Speed (m/s)',data:d.speed_ms,borderColor:COLORS.purple,yAxisID:'y'},
      {label:'Dynamic Pressure (kPa)',data:d.dynamic_pressure,borderColor:COLORS.gold,yAxisID:'y1'}
    ], {scales:{y:{position:'left'},y1:{position:'right',grid:{drawOnChartArea:false}}}});
    
  }catch(e){console.error(e)}
  btn.disabled=false;
}

// ═══════════════════════════════════════════════
// 5. PAYLOAD BUDGET
// ═══════════════════════════════════════════════
document.getElementById('btn-payload')?.addEventListener('click', runPayload);
async function runPayload(){
  const btn=document.getElementById('btn-payload');
  btn.disabled=true;
  try{
    const r=await fetch('/api/payload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({Isp:450,m0:5e5,m_struct:5e4,m_payload:parseFloat(document.getElementById('inp-mp').value),stages:parseInt(document.getElementById('inp-stages').value)})});
    const d=await r.json();
    mkChart('chart-pie',Object.keys(d.pie),[{data:Object.values(d.pie),backgroundColor:[COLORS.blue,COLORS.purple,COLORS.green]}],{type:'doughnut'});
    setHTML('stage-dvs',d.stage_dvs.map((v,i)=>`Stage ${i+1}: ${v} m/s`).join('<br>'));
  }catch(e){console.error(e)}
  btn.disabled=false;
}

// ═══════════════════════════════════════════════
// 6. FINITE MATH
// ═══════════════════════════════════════════════
document.getElementById('btn-finite')?.addEventListener('click', runFinite);
async function runFinite(){
  const btn=document.getElementById('btn-finite');
  btn.disabled=true;
  try{
    const r=await fetch('/api/finite',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({v_ref:500,h_ref:50000})});
    const d=await r.json();
    setHTML('mat-A',d.A.map(row=>row.map(v=>v.toFixed(3).padStart(8)).join(' ')).join('\n'));
    setHTML('eig-display',d.eigenvalues.map(e=>`λ = ${e.re} + ${e.im}i`).join('\n'));
    setHTML('stab-tag',d.stable?'<span class="tag tag-green">Stable</span>':'<span class="tag tag-red">Unstable</span>');
    setHTML('phi-matrices',d.transition_matrices.map(m=>`<div class="card"><div class="card-title">Transition Matrix Φ(t=${m.t})</div><pre class="matrix">${m.matrix.map(r=>r.map(v=>v.toFixed(3).padStart(8)).join(' ')).join('\n')}</pre></div>`).join(''));
  }catch(e){console.error(e)}
  btn.disabled=false;
}

window.addEventListener('load',()=>{ setTimeout(runCFD,500); });
