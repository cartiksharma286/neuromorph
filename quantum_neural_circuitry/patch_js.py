with open("static/script.js", "r") as f:
    js = f.read()

# Make sure tabs logic applies to the new tab implicitly, standard behavior relies on class "tab" and data-tab

js_code = """
    // QCF Logic
    const btnApplyQcf = document.getElementById('btn-apply-qcf');
    if (btnApplyQcf) {
        btnApplyQcf.addEventListener('click', async () => {
            const out = document.getElementById('qcf-output');
            out.textContent = "Processing Quantum Statistical Distributions...";
            try {
                const response = await fetch('/api/quantum_cf/apply');
                const data = await response.json();
                out.textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                out.textContent = "Error: " + err;
            }
        });
    }
"""

if "btnApplyQcf" not in js:
    js = js + "\n" + js_code

with open("static/script.js", "w") as f:
    f.write(js)
