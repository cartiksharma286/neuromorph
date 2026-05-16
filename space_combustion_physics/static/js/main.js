// ── Starfield ─────────────────────────────────────────
(function(){
  const c=document.getElementById('starfield');
  if(!c)return;
  const ctx=c.getContext('2d');
  let W,H,stars=[];
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;initStars()}
  function initStars(){stars=Array.from({length:220},()=>({
    x:Math.random()*W,y:Math.random()*H,
    r:Math.random()*1.4+0.2,
    a:Math.random(),da:Math.random()*.003+.001,
    speed:Math.random()*.15+.05
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

// ── Clock ─────────────────────────────────────────────
function updateClock(){
  const el=document.getElementById('clock');
  if(el)el.textContent=new Date().toUTCString().slice(17,25)+' UTC';
}
setInterval(updateClock,1000);updateClock();

// ── Tabs ──────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// ── Chart helpers ─────────────────────────────────────
const COLORS={blue:'#00c8ff',orange:'#ff5a00',purple:'#9b5fff',green:'#00ffaa',gold:'#ffc740',red:'#ff3366'};
function mkChart(id,type,labels,datasets,opts={}){
  const ctx=document.getElementById(id)?.getContext('2d');
  if(!ctx)return null;
  return new Chart(ctx,{type,data:{labels,datasets},options:{
    responsive:true,maintainAspectRatio:false,animation:{duration:700},
    plugins:{legend:{labels:{color:'#6a7fa8',font:{size:11}}},tooltip:{mode:'index',intersect:false}},
    scales:type==='doughnut'?{}:{
      x:{ticks:{color:'#6a7fa8',maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.05)'}},
      y:{ticks:{color:'#6a7fa8'},grid:{color:'rgba(255,255,255,0.05)'}}
    },
    ...opts
  }});
}
function gradient(ctx,c1,c2){
  const g=ctx.createLinearGradient(0,0,0,300);
  g.addColorStop(0,c1);g.addColorStop(1,c2);return g;
}
const charts={};
function destroyChart(id){if(charts[id]){charts[id].destroy();delete charts[id]}}

function setKPI(id,val){const el=document.getElementById(id);if(el)el.textContent=val}
function setHTML(id,html){const el=document.getElementById(id);if(el)el.innerHTML=html}

// ═══════════════════════════════════════════════
// 1. COMBUSTION PDE
// ═══════════════════════════════════════════════
document.getElementById('btn-combustion')?.addEventListener('click',runCombustion);
async function runCombustion(){
  const btn=document.getElementById('btn-combustion');
  const ld=document.getElementById('ld-combustion');
  btn.disabled=true;ld.classList.add('visible');
  const body={
    fuel:document.getElementById('sel-fuel').value,
    phi:parseFloat(document.getElementById('inp-phi').value),
    P:parseFloat(document.getElementById('inp-pressure').value)
  };
  try{
    const r=await fetch('/api/combustion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    setKPI('kpi-Tpeak',d.peak_temperature.toFixed(0)+' K');
    setKPI('kpi-SL',d.flame_speed.toFixed(2)+' m/s');
    setKPI('kpi-eta',d.combustion_efficiency.toFixed(1)+'%');
    setKPI('kpi-Tad',d.adiabatic_temperature+' K');
    setKPI('kpi-xf',d.flame_position_cm.toFixed(2)+' cm');

    destroyChart('chart-temp');destroyChart('chart-species');
    const c1=document.getElementById('chart-temp').getContext('2d');
    charts['chart-temp']=new Chart(c1,{type:'line',data:{labels:d.x.map(v=>v.toFixed(1)),datasets:[{
      label:'Temperature (K)',data:d.temperature,
      borderColor:COLORS.orange,backgroundColor:'rgba(255,90,0,0.12)',
      borderWidth:2,pointRadius:0,fill:true,tension:0.4
    }]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
      plugins:{legend:{labels:{color:'#6a7fa8'}}},
      scales:{x:{ticks:{color:'#6a7fa8',maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'x (cm)',color:'#6a7fa8'}},
              y:{ticks:{color:'#6a7fa8'},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'T (K)',color:'#6a7fa8'}}}}});

    charts['chart-species']=new Chart(document.getElementById('chart-species').getContext('2d'),{type:'line',
      data:{labels:d.x.map(v=>v.toFixed(1)),datasets:[
        {label:'Fuel (Y_F)',data:d.fuel,borderColor:COLORS.red,pointRadius:0,borderWidth:2,tension:0.4,fill:false},
        {label:'Oxidizer (Y_O)',data:d.oxidizer,borderColor:COLORS.blue,pointRadius:0,borderWidth:2,tension:0.4,fill:false},
        {label:'Products (Y_P)',data:d.products,borderColor:COLORS.green,pointRadius:0,borderWidth:2,tension:0.4,fill:false}
      ]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
        plugins:{legend:{labels:{color:'#6a7fa8'}}},
        scales:{x:{ticks:{color:'#6a7fa8',maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'x (cm)',color:'#6a7fa8'}},
                y:{min:0,max:1,ticks:{color:'#6a7fa8'},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Mass Fraction',color:'#6a7fa8'}}}}});
  }catch(e){console.error(e);}
  btn.disabled=false;ld.classList.remove('visible');
}

// ═══════════════════════════════════════════════
// 2. OPTIMAL THROTTLE
// ═══════════════════════════════════════════════
document.getElementById('btn-throttle')?.addEventListener('click',runThrottle);
async function runThrottle(){
  const btn=document.getElementById('btn-throttle');
  const ld=document.getElementById('ld-throttle');
  btn.disabled=true;ld.classList.add('visible');
  const body={
    Isp:parseFloat(document.getElementById('inp-Isp').value),
    m0:parseFloat(document.getElementById('inp-m0').value),
    mode:document.getElementById('sel-mode').value
  };
  try{
    const r=await fetch('/api/throttle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    setKPI('kpi-dv',d.delta_v_ideal.toLocaleString()+' m/s');
    setKPI('kpi-mr',d.mass_ratio.toFixed(3));
    setKPI('kpi-fc',(d.fuel_consumed_kg/1000).toFixed(1)+' t');
    setKPI('kpi-vf',d.final_velocity_ms.toLocaleString()+' m/s');
    setKPI('kpi-alt-th',d.max_altitude_km.toFixed(1)+' km');

    destroyChart('chart-throttle');destroyChart('chart-vel');
    charts['chart-throttle']=new Chart(document.getElementById('chart-throttle').getContext('2d'),{type:'line',
      data:{labels:d.time.map(v=>v.toFixed(0)),datasets:[{
        label:'Throttle (0–1)',data:d.throttle,
        borderColor:COLORS.orange,backgroundColor:'rgba(255,90,0,0.12)',
        borderWidth:2.5,pointRadius:0,fill:true,stepped:true
      }]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
        plugins:{legend:{labels:{color:'#6a7fa8'}}},
        scales:{x:{ticks:{color:'#6a7fa8',maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Time (s)',color:'#6a7fa8'}},
                y:{min:0,max:1.05,ticks:{color:'#6a7fa8'},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Throttle',color:'#6a7fa8'}}}}});

    charts['chart-vel']=new Chart(document.getElementById('chart-vel').getContext('2d'),{type:'line',
      data:{labels:d.time.map(v=>v.toFixed(0)),datasets:[
        {label:'Velocity (m/s)',data:d.velocity,borderColor:COLORS.blue,pointRadius:0,borderWidth:2,tension:0.3,yAxisID:'y'},
        {label:'Altitude (km)',data:d.altitude_km,borderColor:COLORS.green,pointRadius:0,borderWidth:2,tension:0.3,yAxisID:'y1'}
      ]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
        plugins:{legend:{labels:{color:'#6a7fa8'}}},
        scales:{
          x:{ticks:{color:'#6a7fa8',maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Time (s)',color:'#6a7fa8'}},
          y:{ticks:{color:COLORS.blue},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Velocity (m/s)',color:COLORS.blue},position:'left'},
          y1:{ticks:{color:COLORS.green},grid:{drawOnChartArea:false},title:{display:true,text:'Altitude (km)',color:COLORS.green},position:'right'}
        }}});
  }catch(e){console.error(e);}
  btn.disabled=false;ld.classList.remove('visible');
}

// ═══════════════════════════════════════════════
// 3. TRAJECTORY
// ═══════════════════════════════════════════════
document.getElementById('btn-traj')?.addEventListener('click',runTrajectory);
async function runTrajectory(){
  const btn=document.getElementById('btn-traj');
  const ld=document.getElementById('ld-traj');
  btn.disabled=true;ld.classList.add('visible');
  const body={
    vehicle:document.getElementById('sel-vehicle').value,
    payload:parseFloat(document.getElementById('inp-payload').value),
    orbit:document.getElementById('sel-orbit').value
  };
  try{
    const r=await fetch('/api/trajectory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    setKPI('kpi-alt-traj',d.max_altitude_km.toLocaleString()+' km');
    setKPI('kpi-vf-traj',d.final_speed_ms.toLocaleString()+' m/s');
    setKPI('kpi-dvach',d.delta_v_achieved.toLocaleString()+' m/s');
    setKPI('kpi-dvreq',d.orbit_dv_required.toLocaleString()+' m/s');
    const oel=document.getElementById('orbit-status');
    if(oel)oel.innerHTML=d.orbit_achieved?'<span class="tag tag-green">✓ Orbit Achieved</span>':'<span class="tag tag-red">✗ Orbit Not Achieved</span>';

    destroyChart('chart-traj');destroyChart('chart-speed');
    charts['chart-traj']=new Chart(document.getElementById('chart-traj').getContext('2d'),{type:'scatter',
      data:{datasets:[{
        label:'Trajectory',
        data:d.x_km.map((x,i)=>({x,y:d.z_km[i]})),
        borderColor:COLORS.blue,backgroundColor:'rgba(0,200,255,0.5)',
        pointRadius:1.5,showLine:true,borderWidth:1.5,tension:0.3
      }]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
        plugins:{legend:{labels:{color:'#6a7fa8'}}},
        scales:{x:{ticks:{color:'#6a7fa8'},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Downrange (km)',color:'#6a7fa8'}},
                y:{ticks:{color:'#6a7fa8'},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Altitude (km)',color:'#6a7fa8'}}}}});

    charts['chart-speed']=new Chart(document.getElementById('chart-speed').getContext('2d'),{type:'line',
      data:{labels:d.time.map(v=>v.toFixed(0)),datasets:[
        {label:'Speed (m/s)',data:d.speed_ms,borderColor:COLORS.purple,pointRadius:0,borderWidth:2,tension:0.3},
        {label:'Mass (kg)',data:d.mass,borderColor:COLORS.gold,pointRadius:0,borderWidth:2,tension:0.3,yAxisID:'y1'}
      ]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
        plugins:{legend:{labels:{color:'#6a7fa8'}}},
        scales:{
          x:{ticks:{color:'#6a7fa8',maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.05)'}},
          y:{ticks:{color:COLORS.purple},grid:{color:'rgba(255,255,255,0.05)'},position:'left'},
          y1:{ticks:{color:COLORS.gold},grid:{drawOnChartArea:false},position:'right'}
        }}});
  }catch(e){console.error(e);}
  btn.disabled=false;ld.classList.remove('visible');
}

// ═══════════════════════════════════════════════
// 4. PAYLOAD
// ═══════════════════════════════════════════════
document.getElementById('btn-payload')?.addEventListener('click',runPayload);
async function runPayload(){
  const btn=document.getElementById('btn-payload');btn.disabled=true;
  const body={
    Isp:parseFloat(document.getElementById('inp-Isp-p').value),
    m0:parseFloat(document.getElementById('inp-m0-p').value),
    m_struct:parseFloat(document.getElementById('inp-ms').value),
    m_payload:parseFloat(document.getElementById('inp-mp').value),
    stages:parseInt(document.getElementById('inp-stages').value)
  };
  try{
    const r=await fetch('/api/payload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    setKPI('kpi-dv-pay',d.total_dv.toLocaleString()+' m/s');
    setKPI('kpi-mr-pay',d.mass_ratio.toFixed(3));
    setKPI('kpi-propf',d.propellant_fraction.toFixed(1)+'%');
    setKPI('kpi-payf',d.payload_fraction.toFixed(1)+'%');
    // Stage DVs
    const sdv=document.getElementById('stage-dvs');
    if(sdv)sdv.innerHTML=d.stage_dvs.map((v,i)=>`<div class="kpi" style="display:inline-block;margin:4px 8px"><div class="kpi-value" style="color:var(--purple)">${v.toLocaleString()} m/s</div><div class="kpi-label">Stage ${i+1} Δv</div></div>`).join('');

    destroyChart('chart-pie');
    charts['chart-pie']=new Chart(document.getElementById('chart-pie').getContext('2d'),{type:'doughnut',
      data:{labels:Object.keys(d.pie),datasets:[{data:Object.values(d.pie),
        backgroundColor:['rgba(0,200,255,0.7)','rgba(155,95,255,0.7)','rgba(0,255,170,0.7)'],
        borderColor:['#00c8ff','#9b5fff','#00ffaa'],borderWidth:2,hoverOffset:8}]},
      options:{responsive:true,maintainAspectRatio:false,animation:{duration:700},cutout:'62%',
        plugins:{legend:{position:'bottom',labels:{color:'#6a7fa8',padding:16}}}}});
  }catch(e){console.error(e);}
  btn.disabled=false;
}

// ═══════════════════════════════════════════════
// 5. CFD MODELING
// ═══════════════════════════════════════════════
document.getElementById('btn-cfd')?.addEventListener('click',runCFD);
async function runCFD(){
  const btn=document.getElementById('btn-cfd');
  const ld=document.getElementById('ld-cfd');
  btn.disabled=true;ld.classList.add('visible');
  const body={
    throttle:parseFloat(document.getElementById('inp-cfd-throttle').value),
    fuel:document.getElementById('sel-cfd-fuel').value
  };
  try{
    const r=await fetch('/api/cfd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    setKPI('kpi-thrust',d.thrust_kN.toLocaleString()+' kN');
    setKPI('kpi-mach',d.exit_mach.toFixed(2));
    setKPI('kpi-chamb-t',d.chamber_temp.toLocaleString()+' K');
    setKPI('kpi-peak-p',d.peak_pressure.toFixed(1)+' bar');

    destroyChart('chart-cfd-press');destroyChart('chart-cfd-temp');
    
    // Pressure & Mach Chart
    charts['chart-cfd-press']=new Chart(document.getElementById('chart-cfd-press').getContext('2d'),{type:'line',
      data:{labels:d.x.map(v=>v.toFixed(2)),datasets:[
        {label:'Pressure (bar)',data:d.pressure,borderColor:COLORS.orange,pointRadius:0,borderWidth:2,tension:0.3,yAxisID:'y'},
        {label:'Mach Number',data:d.mach,borderColor:COLORS.blue,pointRadius:0,borderWidth:2,tension:0.3,yAxisID:'y1'}
      ]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
        plugins:{legend:{labels:{color:'#6a7fa8'}}},
        scales:{
          x:{ticks:{color:'#6a7fa8',maxTicksLimit:10},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Nozzle Position x (m)',color:'#6a7fa8'}},
          y:{ticks:{color:COLORS.orange},grid:{color:'rgba(255,255,255,0.05)'},position:'left',title:{display:true,text:'Pressure (bar)',color:COLORS.orange}},
          y1:{ticks:{color:COLORS.blue},grid:{drawOnChartArea:false},position:'right',title:{display:true,text:'Mach Number',color:COLORS.blue}}
        }}});

    // Temp & Fuel Chart
    charts['chart-cfd-temp']=new Chart(document.getElementById('chart-cfd-temp').getContext('2d'),{type:'line',
      data:{labels:d.x.map(v=>v.toFixed(2)),datasets:[
        {label:'Temperature (K)',data:d.temperature,borderColor:COLORS.red,pointRadius:0,borderWidth:2,tension:0.3,yAxisID:'y'},
        {label:'Fuel Fraction',data:d.fuel_fraction,borderColor:COLORS.green,pointRadius:0,borderWidth:2,tension:0.3,yAxisID:'y1'}
      ]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
        plugins:{legend:{labels:{color:'#6a7fa8'}}},
        scales:{
          x:{ticks:{color:'#6a7fa8',maxTicksLimit:10},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Nozzle Position x (m)',color:'#6a7fa8'}},
          y:{ticks:{color:COLORS.red},grid:{color:'rgba(255,255,255,0.05)'},position:'left',title:{display:true,text:'Temperature (K)',color:COLORS.red}},
          y1:{ticks:{color:COLORS.green},grid:{drawOnChartArea:false},position:'right',title:{display:true,text:'Fuel Mass Fraction',color:COLORS.green},min:0,max:1}
        }}});
  }catch(e){console.error(e);}
  btn.disabled=false;ld.classList.remove('visible');
}

// ═══════════════════════════════════════════════
// 6. FINITE MATH
// ═══════════════════════════════════════════════
document.getElementById('btn-finite')?.addEventListener('click',runFinite);
async function runFinite(){
  const btn=document.getElementById('btn-finite');btn.disabled=true;
  const body={
    v_ref:parseFloat(document.getElementById('inp-vref').value),
    h_ref:parseFloat(document.getElementById('inp-href').value)
  };
  try{
    const r=await fetch('/api/finite',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    const fmtMat=(m,label)=>{
      const rows=m.map(row=>row.map(v=>v.toFixed(4).padStart(9)).join('  ')).join('\n');
      return `${label}\n${'─'.repeat(52)}\n${rows}`;
    };
    setHTML('mat-A',fmtMat(d.A,'A Matrix (State)'));
    setHTML('eig-display',d.eigenvalues.map(e=>`λ = ${e.re >= 0?'+':''}${e.re} ${e.im>=0?'+':'-'} ${Math.abs(e.im)}i`).join('\n'));
    const stab=document.getElementById('stab-tag');
    if(stab)stab.innerHTML=d.stable?'<span class="tag tag-green">✓ Asymptotically Stable</span>':'<span class="tag tag-red">✗ Unstable</span>';
    setKPI('kpi-cond',d.condition_number);

    const phiDiv=document.getElementById('phi-matrices');
    if(phiDiv)phiDiv.innerHTML=d.transition_matrices.map(p=>`
      <div class="card" style="margin-bottom:14px">
        <div class="card-title">Φ(t=${p.t}s) — State Transition Matrix</div>
        <div class="matrix">${fmtMat(p.matrix,'')}</div>
      </div>`).join('');
  }catch(e){console.error(e);}
  btn.disabled=false;
}

// Auto-run combustion on load
window.addEventListener('load',()=>{ setTimeout(runCombustion,600); });
