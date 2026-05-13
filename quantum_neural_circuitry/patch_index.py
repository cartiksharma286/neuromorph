import re

with open("static/index.html", "r") as f:
    content = f.read()

# Add a tab link
tab_link = '<div class="tab" data-tab="qcf">Quantum Stats & Continued Fractions</div>'
if 'data-tab="qcf"' not in content:
    content = content.replace('<div class="tab" data-tab="quantum-stat">Quantum Theoretic Statistics</div>', '<div class="tab" data-tab="quantum-stat">Quantum Theoretic Statistics</div>\n            ' + tab_link)

# Add tab content
tab_content = """
            <!-- QCF Tab -->
            <div class="tab-content" id="qcf-tab">
                <h3>Quantum Statistical Distributions & Continued Fractions</h3>
                <div class="card">
                    <p>Improve neural circuitry utilizing statistical distributions integrated via continued fraction expansions.</p>
                    <button id="btn-apply-qcf" class="btn">Apply QCF Improvement</button>
                    <pre id="qcf-output" style="margin-top: 15px; background: #1a1a2e; padding: 10px; border-radius: 5px; color: #4deeea; overflow-x: auto;"></pre>
                </div>
            </div>
"""

if 'id="qcf-tab"' not in content:
    content = content.replace('<!-- Quantum Theoretical Statistic Tab -->', tab_content + '\n            <!-- Quantum Theoretical Statistic Tab -->')

with open("static/index.html", "w") as f:
    f.write(content)
