const $ = (id) => document.getElementById(id);

// Tab navigation handler
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');
    const targetId = btn.getAttribute('data-tab');
    const targetContent = document.getElementById(targetId);
    if (targetContent) targetContent.classList.add('active');
  });
});

const fields = ['radius', 'layers', 'attenuation'];
fields.forEach((id) => {
  const el = $(id);
  if (el) {
    el.addEventListener('input', () => {
      const out = $(id + 'Out');
      if (out) out.textContent = el.value;
    });
  }
});

function drawField(points) {
  const canvas = $('fieldCanvas');
  if (!canvas) return;
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
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width, height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#376163'; ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach((fraction) => { ctx.beginPath(); ctx.moveTo(0, height * fraction); ctx.lineTo(width, height * fraction); ctx.stroke(); });
  const maxLoss = Math.max(...samples.map((sample) => sample.loss));
  ctx.beginPath();
  samples.forEach((sample, index) => { const x = index * width / (samples.length - 1); const y = height - sample.loss / maxLoss * (height - 12) - 6; index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = '#ee8068'; ctx.lineWidth = 3; ctx.stroke();
  const statusEl = $('convergenceStatus');
  if (statusEl) statusEl.textContent = `loss ${samples[samples.length - 1].loss}`;
}

function renderCharacteristics(target, values) {
  if (!target) return;
  target.innerHTML = Object.entries(values).map(([key, value]) => `<span><small>${key.replaceAll('_', ' ')}</small><b>${value}</b></span>`).join('');
}

async function simulate() {
  const button = $('simulate');
  if (button) { button.disabled = true; button.textContent = 'Computing field...'; }
  try {
    const response = await fetch('/api/simulate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        radius: $('radius') ? $('radius').value : 1.0,
        layers: $('layers') ? $('layers').value : 12,
        attenuation: $('attenuation') ? $('attenuation').value : 1.0,
        seed: $('seed') ? $('seed').value : 1009
      })
    });
    if (!response.ok) throw new Error('Simulation request failed');
    const data = await response.json();
    drawField(data.points);
    drawConvergence(data.convergence);
    if ($('visibility')) $('visibility').textContent = data.visibility_index < 0.001 ? '<0.001' : data.visibility_index.toFixed(6);
    if ($('scattering')) $('scattering').textContent = `${data.scattering} dB`;
    if ($('confidence')) $('confidence').textContent = `${data.qml_confidence}%`;
    if ($('scheduleId')) $('scheduleId').textContent = data.schedule_id;
    if ($('primeList')) $('primeList').innerHTML = data.prime_schedule.map((prime, index) => `<span>${String(index + 1).padStart(2, '0')} / ${prime}</span>`).join('');
    renderCharacteristics($('cloakCharacteristics'), data.cloak_characteristics);
  } catch (error) {
    if ($('convergenceStatus')) $('convergenceStatus').textContent = error.message;
  } finally {
    if (button) { button.disabled = false; button.innerHTML = 'Run field simulation <span>↗</span>'; }
  }
}

async function createSession() {
  const response = await fetch('/api/session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      subject: $('subject') ? $('subject').value : 'Ontario Clinical Telemetry',
      seed: $('seed') ? $('seed').value : 1009
    })
  });
  const data = await response.json();
  renderCharacteristics($('sessionCharacteristics'), data.session_characteristics);
  const resEl = $('sessionResult');
  if (resEl) {
    resEl.textContent = `SESSION ${data.session_id}\nCOMMITMENT ${data.commitment}\nCREATED ${data.created_at}\n\n${data.key_exchange} · ${data.signature}\n${data.privacy_basis}`;
  }
}

// ==================== TAB 2: POST-QUANTUM CRYPTOGRAPHY ====================
async function evaluatePqc() {
  const btn = $('evaluatePqcBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Evaluating Lattice...'; }
  try {
    const response = await fetch('/api/pqc/evaluate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        level: $('pqcLevel') ? $('pqcLevel').value : 'ML-KEM-768',
        nodes: $('pqcNodes') ? $('pqcNodes').value : 8,
        seed: $('pqcSeed') ? $('pqcSeed').value : 1009
      })
    });
    const data = await response.json();
    if ($('pqcSecurityBits')) $('pqcSecurityBits').textContent = `${data.quantum_security_bits} Bits`;
    if ($('pqcDecryptionFailure')) $('pqcDecryptionFailure').textContent = `≤ ${data.decryption_failure_bound}`;
    if ($('pqcShannonEntropy')) $('pqcShannonEntropy').textContent = `${data.shannon_entropy_bits} Bits`;
    if ($('pqcMeshEdges')) $('pqcMeshEdges').textContent = `${data.mesh_combinatorics.key_exchange_edges} Links`;

    const polyEl = $('polySampleTable');
    if (polyEl) {
      polyEl.innerHTML = `RING: <b>${data.ring}</b> (Modulus q = ${data.modulus_q}, Dim n = ${data.dimension_n}, Rank k = ${data.module_rank_k})
--------------------------------------------------------------------------------
A[0] (Uniform ∈ Z_q):  [${data.polynomial_samples.matrix_row_a0.join(', ')}]
s (Secret ∈ χ_${data.noise_eta}):      [${data.polynomial_samples.secret_vector_s.join(', ')}]
e (Noise ∈ χ_${data.noise_eta}):       [${data.polynomial_samples.noise_vector_e.join(', ')}]
t = A·s + e (mod q):   [${data.polynomial_samples.public_key_t.join(', ')}]
--------------------------------------------------------------------------------
Min-Entropy H_min(X|E): ${data.min_entropy_bits} bits | PQC Schedule: ${data.schedule_id}`;
    }

    const meshEl = $('meshCombinatoricsBox');
    if (meshEl) {
      meshEl.innerHTML = `• Participant Clinical Nodes: <b>${data.mesh_combinatorics.participant_nodes} nodes</b>
• Complete Key Mesh Edges: <b>${data.mesh_combinatorics.key_exchange_edges} links (K_${data.mesh_combinatorics.participant_nodes})</b>
• Cayley Spanning Trees: <b>${data.mesh_combinatorics.cayley_spanning_trees}</b>
• Partition Entropy: <b>${data.mesh_combinatorics.partition_entropy_bits} bits</b>
• Active Coordination Primes: <b>[${data.recurrent_primes.slice(0, 5).join(', ')}...]</b>`;
    }
  } catch (err) {
    console.error(err);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = 'Evaluate Lattice Combinatorics <span>↗</span>'; }
  }
}

// ==================== TAB 3: CANADIAN PATIENT PRIVACY ====================
async function cloakPatient() {
  const btn = $('cloakPatientBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Shielding Patient Telemetry...'; }
  try {
    const payload = {
      patient_id: $('patientIdInput') ? $('patientIdInput').value : 'CAN-ON-4892-X',
      modality: $('patientModality') ? $('patientModality').value : '7T Neuroimaging MRI + WGS Biobank',
      age: $('patientAge') ? $('patientAge').value : 58,
      systolic_bp: $('patientBp') ? $('patientBp').value : 134.5,
      biomarker: $('patientBio') ? $('patientBio').value : 42.8,
      postal_prefix: $('patientPostal') ? $('patientPostal').value : 'M5S',
      genomic_variant: $('patientGenomic') ? $('patientGenomic').value : 'APOE-ε4/ε4 / rs429358(C)',
      epsilon: $('patientEpsilon') ? $('patientEpsilon').value : 0.50
    };
    const response = await fetch('/api/patient/cloak', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await response.json();

    const rawBox = $('rawPatientBox');
    if (rawBox) {
      rawBox.innerHTML = `<b>PATIENT ID:</b> ${data.raw_record.patient_id}
<b>CLINICAL MODALITY:</b> ${data.modality}
<b>EXACT AGE:</b> ${data.raw_record.age} yrs
<b>SYSTOLIC BLOOD PRESSURE:</b> ${data.raw_record.systolic_bp_mmhg} mmHg
<b>SERUM BIOMARKER LEVEL:</b> ${data.raw_record.biomarker_ng_ml} ng/mL
<b>RESIDENCE POSTAL FSA:</b> ${data.raw_record.postal_code} (Direct Geo-linkable)
<b>GENOMIC SEQUENCE:</b> ${data.raw_record.genomic_variant}`;
    }

    const cloakedBox = $('cloakedPatientBox');
    if (cloakedBox) {
      cloakedBox.innerHTML = `<b>POST-QUANTUM TOKEN:</b> <span style="color:#0284c7;">${data.patient_token}</span>
<b>AGE BRACKET:</b> ${data.cloaked_record.age_bracket} yrs (Generalization)
<b>PERTURBED BP:</b> ${data.cloaked_record.systolic_bp_sanitized} mmHg (DP noise: ${data.cloaked_record.dp_noise_added_bp > 0 ? '+' : ''}${data.cloaked_record.dp_noise_added_bp})
<b>PERTURBED BIOMARKER:</b> ${data.cloaked_record.biomarker_sanitized} ng/mL (DP noise: ${data.cloaked_record.dp_noise_added_bio > 0 ? '+' : ''}${data.cloaked_record.dp_noise_added_bio})
<b>MASKED FSA:</b> ${data.cloaked_record.geographic_fsa_masked}
<b>GENOMIC SHA3-256 HASH:</b> ${data.cloaked_record.genomic_hash}
<b>PHYSICAL METAMATERIAL ENCLAVE:</b> ${data.metamaterial_enclave_cloaking.rf_attenuation_db} attenuation (Vis: ${data.metamaterial_enclave_cloaking.residual_visibility_index})`;
    }

    if ($('cardEpsilon')) $('cardEpsilon').textContent = `ε = ${data.privacy_guarantees.differential_privacy_epsilon}, δ = ${data.privacy_guarantees.differential_privacy_delta}`;
    if ($('cardKAnon')) $('cardKAnon').textContent = `${data.privacy_guarantees.k_anonymity_achieved}, ${data.privacy_guarantees.l_diversity_achieved}`;
    if ($('cardReident')) $('cardReident').textContent = `${data.privacy_guarantees.reidentification_probability}`;
    if ($('cardAttenuation')) $('cardAttenuation').textContent = `${data.metamaterial_enclave_cloaking.rf_attenuation_db} (${data.metamaterial_enclave_cloaking.residual_visibility_index})`;
  } catch (err) {
    console.error(err);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = 'Apply Canadian Patient Shielding & Cloaking <span>↗</span>'; }
  }
}

// ==================== TAB 4: AUDIT COMPLIANCE ====================
async function loadComplianceAudit() {
  const btn = $('runAuditBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Auditing...'; }
  try {
    const response = await fetch('/api/compliance/audit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ framework: 'all' })
    });
    const data = await response.json();

    const canBody = $('canadianAuditBody');
    if (canBody && data.canadian_compliance) {
      canBody.innerHTML = data.canadian_compliance.statutes
        .map((s) => `
        <tr>
          <td><strong>${s.name}</strong><br><small style="color:#64748b;">${s.authority}</small></td>
          <td>${s.section}</td>
          <td>${s.technical_control}</td>
          <td><b style="color:#0284c7;">${s.compliance_score}%</b></td>
          <td><span class="badge green-bg">${s.status}</span></td>
        </tr>`)
        .join('');
    }

    const euBody = $('europeanAuditBody');
    if (euBody && data.european_compliance) {
      euBody.innerHTML = data.european_compliance.statutes
        .map((s) => `
        <tr>
          <td><strong>${s.name}</strong></td>
          <td><small style="color:#64748b;">${s.authority}</small></td>
          <td>${s.section}</td>
          <td>${s.technical_control}</td>
          <td><b style="color:#16a34a;">${s.compliance_score}%</b></td>
          <td><span class="badge green-bg">${s.status}</span></td>
        </tr>`)
        .join('');
    }
  } catch (err) {
    console.error(err);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = 'Re-evaluate Multi-Jurisdiction Audit <span>↻</span>'; }
  }
}

// ==================== NEWLY DEVELOPED PRIVACY ALGORITHMS ====================

// 01: NTT Polynomial Multiplier
async function runNtt() {
  const btn = $('runNttBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Running NTT...'; }
  try {
    const response = await fetch('/api/algorithms/ntt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        degree: $('nttDegree') ? $('nttDegree').value : 256,
        seed: $('nttSeed') ? $('nttSeed').value : 1009
      })
    });
    const data = await response.json();
    const box = $('nttResultsBox');
    if (box) {
      box.innerHTML = `
RING: <b>${data.ring}</b> (Modulus q = ${data.modulus_q}, Root of Unity ζ = ${data.root_of_unity_zeta})
EXECUTION LATENCY: <b>${data.latency_microseconds} μs</b> (Simulated NTT Kernel)
--------------------------------------------------------------------------------
Poly A (Sample): [${data.poly_a_sample.join(', ')}...]
Poly s (Secret): [${data.poly_s_sample.join(', ')}...]
Product t (NTT): [${data.product_poly_t.join(', ')}...]
--------------------------------------------------------------------------------
MAX COEFF: ${data.max_coefficient} | AVG COEFF: ${data.average_coefficient}
${data.theorem_tail_bound}`;
    }
  } catch (err) {
    console.error(err);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = 'Run NTT <span>↗</span>'; }
  }
}

// 02: Rényi Differential Privacy (RDP)
async function runRdp() {
  const btn = $('runRdpBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Tracking RDP...'; }
  try {
    const response = await fetch('/api/algorithms/rdp', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        epochs: $('rdpEpochs') ? $('rdpEpochs').value : 20,
        sigma: $('rdpSigma') ? $('rdpSigma').value : 1.5,
        delta: $('rdpDelta') ? parseFloat($('rdpDelta').value) : 1e-6
      })
    });
    const data = await response.json();
    const box = $('rdpResultsBox');
    if (box) {
      box.innerHTML = `
RDP COMPOSITION: <b>${data.epochs} epochs</b> (Gaussian σ = ${data.gaussian_sigma})
OPTIMAL DIVERGENCE ORDER: <b>α = ${data.optimal_alpha}</b>
TOTAL SPENT PRIVACY: <b>ε = ${data.optimal_standard_epsilon}</b> (δ = ${data.target_delta})
MUTUAL INFORMATION BOUND: <b>I(X; Y) ≤ ${data.mutual_information_leakage_bound_bits}</b>
--------------------------------------------------------------------------------
STATUS: ${data.privacy_guarantee}`;
    }
  } catch (err) {
    console.error(err);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = 'Track RDP <span>↗</span>'; }
  }
}

// 03: Stirling Partition & k-Anonymity
async function runStirling() {
  const btn = $('runStirlingBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Partitioning...'; }
  try {
    const response = await fetch('/api/algorithms/stirling', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        cohort_size: $('stirlingCohort') ? $('stirlingCohort').value : 48,
        clusters: $('stirlingClusters') ? $('stirlingClusters').value : 4
      })
    });
    const data = await response.json();
    const box = $('stirlingResultsBox');
    if (box) {
      box.innerHTML = `
COHORT SIZE: <b>N = ${data.cohort_size_N} patients</b> → <b>m = ${data.partition_clusters_m} clusters</b>
STIRLING S(N, m): <b>${data.stirling_number_approx}</b> partition ways (log10 ≈ ${data.log10_stirling})
GUARANTEED ANONYMITY: <b>${data.guaranteed_k_anonymity}</b> | <b>${data.achieved_l_diversity}</b>
SHANNON PARTITION ENTROPY: <b>${data.shannon_partition_entropy}</b>
RE-IDENTIFICATION BOUND: <b>${data.reidentification_probability_bound}</b>`;
    }
  } catch (err) {
    console.error(err);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = 'Partition <span>↗</span>'; }
  }
}

// 04: Simplicial Metamaterial Tensors
async function runTensors() {
  const btn = $('runTensorsBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Solving...'; }
  try {
    const response = await fetch('/api/algorithms/tensors', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        inner_a: $('tensorA') ? $('tensorA').value : 0.5,
        outer_b: $('tensorB') ? $('tensorB').value : 1.5,
        layers: $('tensorLayers') ? $('tensorLayers').value : 12
      })
    });
    const data = await response.json();
    const box = $('tensorsResultsBox');
    if (box) {
      const sampleLayers = data.tensor_layers.slice(0, 3);
      const layerRows = sampleLayers.map((l) => `L${l.layer_index} (r'=${l.r_prime}): ε^rr=${l.eps_rr}, ε^θθ=${l.eps_thetatheta}, n=${l.refractive_index}`).join('\n');
      box.innerHTML = `
ANNULAR REGION: [a=${data.inner_radius_a}, b=${data.outer_radius_b}] across <b>${data.layer_count} shells</b>
TOTAL ATTENUATION: <b>${data.total_scattering_attenuation_db}</b>
CF STABILITY SCORE: <b>S_CF = ${data.cf_stability_score}</b> (Terms: [${data.continued_fraction_terms.join(', ')}])
--------------------------------------------------------------------------------
${layerRows}
... (${data.layer_count - 3} additional shells solved)`;
    }
  } catch (err) {
    console.error(err);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = 'Solve Tensors <span>↗</span>'; }
  }
}

if ($('simulate')) $('simulate').addEventListener('click', simulate);
if ($('createSession')) $('createSession').addEventListener('click', createSession);
if ($('evaluatePqcBtn')) $('evaluatePqcBtn').addEventListener('click', evaluatePqc);
if ($('cloakPatientBtn')) $('cloakPatientBtn').addEventListener('click', cloakPatient);
if ($('runAuditBtn')) $('runAuditBtn').addEventListener('click', loadComplianceAudit);
if ($('runNttBtn')) $('runNttBtn').addEventListener('click', runNtt);
if ($('runRdpBtn')) $('runRdpBtn').addEventListener('click', runRdp);
if ($('runStirlingBtn')) $('runStirlingBtn').addEventListener('click', runStirling);
if ($('runTensorsBtn')) $('runTensorsBtn').addEventListener('click', runTensors);

simulate();
evaluatePqc();
cloakPatient();
loadComplianceAudit();
runNtt();
runRdp();
runStirling();
runTensors();

