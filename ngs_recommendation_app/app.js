const form = document.getElementById("recommendationForm");
const resultPanel = document.getElementById("resultPanel");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  resultPanel.innerHTML = `
    <div class="loading-card">
      <div class="spinner"></div>
      <h2>Assembling the recommendation</h2>
      <p>Inferring a quantum-like state and draft rationale from your wet-lab and informatics preferences.</p>
    </div>
  `;

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    renderResult(data);
  } catch (error) {
    resultPanel.innerHTML = `
      <div class="loading-card error-card">
        <h2>Recommendation unavailable</h2>
        <p>${error.message}</p>
      </div>
    `;
  }
});

function renderResult(data) {
  const { summary, recommendation, qml, llm_rationale, evidence } = data;
  resultPanel.innerHTML = `
    <div class="result-card hero-result">
      <div class="result-header">
        <div>
          <p class="eyebrow">${summary.title}</p>
          <h2>${summary.headline}</h2>
        </div>
        <div class="coherence-pill">Coherence ${Math.round(qml.coherence * 100)}%</div>
      </div>
      <p class="lead">${summary.description}</p>
    </div>

    <div class="result-grid">
      <article class="result-card">
        <h3>Suggested workflow</h3>
        <p>${recommendation.workflow}</p>
      </article>
      <article class="result-card">
        <h3>Sequencing platform</h3>
        <p>${recommendation.platform}</p>
      </article>
      <article class="result-card">
        <h3>QC strategy</h3>
        <p>${recommendation.qc}</p>
      </article>
      <article class="result-card">
        <h3>Analysis stack</h3>
        <p>${recommendation.analysis}</p>
      </article>
      <article class="result-card">
        <h3>Reporting output</h3>
        <p>${recommendation.reporting}</p>
      </article>
    </div>

    <div class="result-grid compact">
      <article class="result-card">
        <h3>Quantum-inspired state</h3>
        <div class="meter-row"><span>Coherence</span><div class="meter"><i style="width:${qml.coherence * 100}%"></i></div></div>
        <div class="meter-row"><span>Entropy</span><div class="meter"><i style="width:${(1 - qml.entropy) * 100}%"></i></div></div>
      </article>
      <article class="result-card">
        <h3>LLM-ready rationale</h3>
        <ul>
          ${llm_rationale.map((item) => `<li>${item}</li>`).join("")}
        </ul>
      </article>
      <article class="result-card">
        <h3>Evidence signals</h3>
        <ul>
          ${evidence.map((item) => `<li>${item}</li>`).join("")}
        </ul>
      </article>
    </div>
  `;
}
