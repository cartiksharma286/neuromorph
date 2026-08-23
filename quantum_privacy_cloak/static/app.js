const $ = (id) => document.getElementById(id);
const fields = ['radius', 'layers', 'attenuation'];
fields.forEach((id) => $(id).addEventListener('input', () => { $(id + 'Out').textContent = $(id).value; }));

function drawField(points) {
  const canvas = $('fieldCanvas');
  const ctx = canvas.getContext('2d');
  const scale = Math.min(canvas.width, canvas.height) * .34;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save(); ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.strokeStyle = '#3d7475'; ctx.lineWidth = 1;
  for (let ring = 1; ring < 5; ring++) { ctx.beginPath(); ctx.arc(0, 0, scale * ring / 4, 0, Math.PI * 2); ctx.stroke(); }
  ctx.beginPath();
  points.forEach((point, index) => { const x = point.x * scale / 2, y = point.y * scale / 2; index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath(); ctx.fillStyle = 'rgba(54,183,187,.23)'; ctx.fill(); ctx.strokeStyle = '#8fe1d2'; ctx.lineWidth = 2; ctx.stroke();
  ctx.restore();
}

function drawConvergence(samples) {
  const canvas = $('convergenceCanvas');
  const ctx = canvas.getContext('2d');
  const width = canvas.width, height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#376163'; ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach((fraction) => { ctx.beginPath(); ctx.moveTo(0, height * fraction); ctx.lineTo(width, height * fraction); ctx.stroke(); });
  const maxLoss = Math.max(...samples.map((sample) => sample.loss));
  ctx.beginPath();
  samples.forEach((sample, index) => { const x = index * width / (samples.length - 1); const y = height - sample.loss / maxLoss * (height - 12) - 6; index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = '#ee8068'; ctx.lineWidth = 3; ctx.stroke();
  $('convergenceStatus').textContent = `loss ${samples[samples.length - 1].loss}`;
}

function renderCharacteristics(target, values) {
  target.innerHTML = Object.entries(values).map(([key, value]) => `<span><small>${key.replaceAll('_', ' ')}</small><b>${value}</b></span>`).join('');
}

async function simulate() {
  const button = $('simulate'); button.disabled = true; button.textContent = 'Computing field...';
  try {
    const response = await fetch('/api/simulate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ radius: $('radius').value, layers: $('layers').value, attenuation: $('attenuation').value, seed: $('seed').value }) });
    if (!response.ok) throw new Error('Simulation request failed');
    const data = await response.json();
    drawField(data.points); drawConvergence(data.convergence);
    $('visibility').textContent = `${data.visibility}%`; $('scattering').textContent = `${data.scattering} dB`; $('confidence').textContent = `${data.qml_confidence}%`;
    $('scheduleId').textContent = data.schedule_id;
    $('primeList').innerHTML = data.prime_schedule.map((prime, index) => `<span>${String(index + 1).padStart(2, '0')} / ${prime}</span>`).join('');
    renderCharacteristics($('cloakCharacteristics'), data.cloak_characteristics);
  } catch (error) { $('convergenceStatus').textContent = error.message; } finally { button.disabled = false; button.innerHTML = 'Run field simulation <span>↗</span>'; }
}

async function createSession() {
  const response = await fetch('/api/session', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ subject: $('subject').value, seed: $('seed').value }) });
  const data = await response.json();
  renderCharacteristics($('sessionCharacteristics'), data.session_characteristics);
  $('sessionResult').textContent = `SESSION ${data.session_id}\nCOMMITMENT ${data.commitment}\nCREATED ${data.created_at}\n\n${data.key_exchange} · ${data.signature}\n${data.privacy_basis}`;
}

$('simulate').addEventListener('click', simulate);
$('createSession').addEventListener('click', createSession);
simulate();
