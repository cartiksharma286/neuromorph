with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js = f.read()

handlers = """
// Auto trigger bindings
document.addEventListener("DOMContentLoaded", () => {
    const dr = document.getElementById('decline-range');
    const da = document.getElementById('dementia-dbs-amp');
    const ga = document.getElementById('gen-ai-prompt');
    
    if (dr) dr.addEventListener('input', runDementiaStaging);
    if (da) da.addEventListener('input', runDementiaStaging);
    if (ga) ga.addEventListener('change', runDementiaStaging);
    
    // Attempt init run
    setTimeout(runDementiaStaging, 500); 
});
"""

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js + "\n" + handlers)
