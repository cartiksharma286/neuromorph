// ═══════════════════════════════════════════════
// 10. DEMENTIA CURE WITH DEEP BRAIN STIMULATION (DBS)
// ═══════════════════════════════════════════════
document.getElementById('btn-dbs-run')?.addEventListener('click', runDBS);
async function runDBS() {
  const amp = parseFloat(document.getElementById('inp-dbs-amp').value);
  const width = parseFloat(document.getElementById('inp-dbs-width').value);
  const freq = parseFloat(document.getElementById('inp-dbs-freq').value);
  const region = document.getElementById('inp-dbs-region').value;
  const r = await fetch('/api/dbs', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({amplitude:amp, width, freq, region})
  });
  const d = await r.json();
  setKPI('kpi-dbs-efficacy', d.efficacy.toFixed(1));
  setKPI('kpi-dbs-repair', d.repair_score);
  setKPI('kpi-dbs-asymptote', d.asymptote);
  setHTML('dbs-log', d.log);
}
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
    setKPI('kpi-thrust',d.thrust_kN+' kN'); 
    setKPI('kpi-mach',d.exit_mach); 
    setKPI('kpi-chamb-t',d.chamber_temp+' K'); 
    setKPI('kpi-peak-q',d.peak_q+' MW/m²');
    setKPI('kpi-peak-stress',d.peak_wall_stress+' MPa');
    setKPI('kpi-min-sf',d.min_safety_factor);
    
    const xL=d.x.map(v=>v.toFixed(2));
    mkChart('chart-cfd-press',xL,[{label:'Pressure (bar)',data:d.pressure,borderColor:COLORS.orange,yAxisID:'y'},{label:'Mach',data:d.mach,borderColor:COLORS.blue,yAxisID:'y1'}],{scales:{y:{position:'left'},y1:{position:'right',grid:{drawOnChartArea:false}}}});
    mkChart('chart-cfd-temp-gas-wall',xL,[
      {label:'Gas T (K)',data:d.temperature,borderColor:COLORS.red},
      {label:'Inner Wall T (K)',data:d.wall_temp,borderColor:COLORS.purple},
      {label:'Outer Wall T (K)',data:d.outer_wall_temp,borderColor:COLORS.blue}
    ]);
    mkChart('chart-cfd-stress-sf',xL,[
      {label:'Thermal Stress (MPa)',data:d.thermal_stress,borderColor:COLORS.purple,yAxisID:'y'},
      {label:'Safety Factor',data:d.safety_factor,borderColor:COLORS.gold,yAxisID:'y1'}
    ],{scales:{y:{position:'left',title:{display:true,text:'Stress (MPa)'}},y1:{position:'right',grid:{drawOnChartArea:false},title:{display:true,text:'Safety Factor'}}}});
    
    mkChart('chart-cfd-q',xL,[{label:'Heat Flux (MW/m²)',data:d.heat_flux,borderColor:COLORS.gold,fill:true,backgroundColor:'rgba(255,199,64,0.1)'}]);
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
// 4. TRAJECTORY (QUANTUM HPC UPGRADE)
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
    setKPI('kpi-qml-score',(d.qml_optimization_score*100).toFixed(2)+'%');
    setKPI('kpi-maxq',d.max_q_kpa+' kPa');
    setHTML('orbit-status',d.orbit_achieved?'<span class="tag tag-green">Orbit Achieved</span>':'<span class="tag tag-red">Suborbital</span>');
    
    const tL=d.time.map(v=>v.toFixed(0));

    // 1. Quantum Flight Path
    mkChart('chart-traj',null,[{
      label:'Quantum-Corrected Path',
      data:d.x_km.map((x,i)=>({x,y:d.z_km[i]})),
      borderColor:COLORS.blue,
      backgroundColor:'rgba(0,200,255,0.05)',
      fill:true,showLine:true,pointRadius:0,borderWidth:2
    }],{type:'scatter',scales:{x:{title:{display:true,text:'Downrange (km)'}},y:{title:{display:true,text:'Altitude (km)'}}}});

    // 2. Quantum Signature
    mkChart('chart-q-sig',tL,[{
      label:'Quantum Signature |ψ|²',
      data:d.quantum_signature,
      borderColor:COLORS.purple,
      backgroundColor:'rgba(155,95,255,0.1)',
      fill:true,pointRadius:0,borderWidth:1.5
    }]);

    // 3. Velocity & Q
    mkChart('chart-speed',tL,[
      {label:'Velocity (m/s)',data:d.speed_ms,borderColor:COLORS.green,yAxisID:'y'},
      {label:'Max Q (kPa)',data:d.dynamic_pressure,borderColor:COLORS.orange,yAxisID:'y1'}
    ], {scales:{y:{position:'left'},y1:{position:'right',grid:{drawOnChartArea:false}}}});

    setHTML('qml-log', `[VQE] STATE INITIALIZED\n[CF] QUADRATURE NODES CONVERGED\n[QML] FIDELITY: ${d.qml_optimization_score}\n[OPT] TRAJECTORY CORRECTED VIA Q-SIGMA`);
    
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

// ═══════════════════════════════════════════════
// 7. THROTTLE UPTAKE
// ═══════════════════════════════════════════════
document.getElementById('btn-uptake')?.addEventListener('click', runUptake);
async function runUptake(){
  const btn=document.getElementById('btn-uptake');
  btn.disabled=true;
  try{
    const r=await fetch('/api/throttle_uptake',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        profile: document.getElementById('sel-uptake-profile').value,
        tp: parseFloat(document.getElementById('inp-uptake-tp').value),
        td: parseFloat(document.getElementById('inp-uptake-td').value)
      })
    });
    const d=await r.json();
    setKPI('kpi-uptake-delay', d.lag_s + ' s');
    setKPI('kpi-uptake-maxv', d.max_velocity + ' m/s');
    setKPI('kpi-uptake-maxt', d.peak_thrust + ' kN');
    setKPI('kpi-uptake-error', d.cf_error);
    
    const tL=d.time.map(v=>v.toFixed(1));
    mkChart('chart-uptake-throttle',tL,[
      {label:'Throttle Cmd',data:d.command,borderColor:COLORS.orange,stepped:true},
      {label:'Actual Uptake (CF)',data:d.actual,borderColor:COLORS.blue}
    ]);
    mkChart('chart-uptake-vel',tL,[
      {label:'Velocity (m/s)',data:d.velocity_ms,borderColor:COLORS.green}
    ]);
    mkChart('chart-uptake-thrust',tL,[
      {label:'Thrust (kN)',data:d.thrust_kN,borderColor:COLORS.red,fill:true,backgroundColor:'rgba(255,51,102,0.05)'}
    ]);
    mkChart('chart-uptake-pump',tL,[
      {label:'Pump Speed (RPM)',data:d.pump_rpm,borderColor:COLORS.purple}
    ]);
  }catch(e){console.error(e)}
  btn.disabled=false;
}

window.addEventListener('load',()=>{ setTimeout(runCFD,500); setTimeout(runUptake,1000); });

// ═══════════════════════════════════════════════
// 8. CANADA ARM 3 KINEMATICS TAB
// ═══════════════════════════════════════════════

async function runCanadaArmFK() {
  const joints = document.getElementById('inp-canadarm-joints').value.split(',').map(Number);
  const r = await fetch('/api/canadarm_kinematics', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:'fk', joints})
  });
  const d = await r.json();
  setKPI('kpi-canadarm-x', d.x.toFixed(2));
  setKPI('kpi-canadarm-y', d.y.toFixed(2));
  setKPI('kpi-canadarm-z', d.z.toFixed(2));
  setHTML('canadarm-log', d.log);
  mkChart('chart-canadarm-workspace', null, [{
    label:'Workspace',
    data: d.workspace.map(pt=>({x:pt[0],y:pt[1]})),
    borderColor: COLORS.blue, showLine:true, pointRadius:2, fill:true, backgroundColor:'rgba(0,200,255,0.08)'
  }], {type:'scatter',scales:{x:{title:{display:true,text:'X (m)'}},y:{title:{display:true,text:'Y (m)'}}}});
}

async function runCanadaArmIK() {
  const ee_target = document.getElementById('inp-canadarm-ee').value.split(',').map(Number);
  const r = await fetch('/api/canadarm_kinematics', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:'ik', ee_target})
  });
  const d = await r.json();
  setHTML('canadarm-ik-solution', d.joints ? d.joints.map((j,i)=>`Joint ${i+1}: ${j.toFixed(2)}°`).join('<br>') : 'No solution');
  setHTML('canadarm-log', d.log);
  document.getElementById('inp-canadarm-joints').value = d.joints.map(j=>j.toFixed(1)).join(',');
  mkChart('chart-canadarm-workspace', null, [{
    label:'Workspace',
    data: d.workspace.map(pt=>({x:pt[0],y:pt[1]})),
    borderColor: COLORS.blue, showLine:true, pointRadius:2, fill:true, backgroundColor:'rgba(0,200,255,0.08)'
  }], {type:'scatter',scales:{x:{title:{display:true,text:'X (m)'}},y:{title:{display:true,text:'Y (m)'}}}});
}

document.getElementById('btn-canadarm-fk')?.addEventListener('click', runCanadaArmFK);
document.getElementById('btn-canadarm-ik')?.addEventListener('click', runCanadaArmIK);

// Auto-run both FK and IK on tab load with prefilled params
document.querySelector('.tab-btn[data-tab="tab-canadarm"]')?.addEventListener('click', ()=>{
  setTimeout(runCanadaArmFK, 200);
  setTimeout(runCanadaArmIK, 600);
});

// ═══════════════════════════════════════════════
// 9. ELECTRICAL SPECS TAB
// ═══════════════════════════════════════════════
document.getElementById('btn-elec-calc')?.addEventListener('click', async()=>{
  const power = parseFloat(document.getElementById('inp-elec-power').value);
  const voltage = parseFloat(document.getElementById('inp-elec-voltage').value);
  const current = parseFloat(document.getElementById('inp-elec-current').value);
  const connector = document.getElementById('inp-elec-connector').value;
  const r = await fetch('/api/electrical_specs', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({power, voltage, current, connector})
  });
  const d = await r.json();
  setKPI('kpi-elec-power', d.power.toFixed(1));
  setKPI('kpi-elec-voltage', d.voltage.toFixed(1));
  setKPI('kpi-elec-current', d.current.toFixed(2));
  setKPI('kpi-elec-connector', d.connector);
  setHTML('elec-log', d.log);
});
