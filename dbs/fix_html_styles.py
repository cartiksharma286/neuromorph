with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    text = f.read()

text = text.replace('id="dementia-sidebar" class="tab-content" style="display:none;"', 'id="dementia-sidebar" class="tab-content"')
text = text.replace('id="dementia-main" class="tab-content" style="display:none; height: 100%;"', 'id="dementia-main" class="tab-content" style="height: 100%;"')

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(text)
