import re

html_file = 'dbs/templates/index.html'
js_file = 'dbs/static/js/main.js'

with open(html_file, 'r') as f:
    html = f.read()

html = re.sub(r'\s*<button class="tab-btn" onclick="switchTab\('"'sunnybrook'"', event\)">Sunnybrook Projections 2036</button>', '', html)
html = re.sub(r'\s*<div id="sunnybrook-sidebar" class="tab-content">.*?</div>\s*</div>', '', html, flags=re.DOTALL)
html = re.sub(r'\s*<div id="sunnybrook-main" class="tab-content".*?</div>\s*</div>\s*</div>\s*</div>', '', html, flags=re.DOTALL)

with open(html_file, 'w') as f:
    f.write(html)

with open(js_file, 'r') as f:
    js = f.read()

js = re.sub(r'function simulateSunnybrook\(\).*?\}\n', '', js, flags=re.DOTALL)
js = re.sub(r'function renderSunnybrookChart\(\).*?\}\n', '', js, flags=re.DOTALL)

with open(js_file, 'w') as f:
    f.write(js)

print("Sunnybrook tab removed.")
