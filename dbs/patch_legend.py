import re

file_path = "templates/index.html"
with open(file_path, "r") as f:
    text = f.read()

legend_html = """
                    <div style="font-size: 10px; color: var(--accent-pink); margin-bottom: 5px;">CORTICAL CURRENT DENSITY (BEM)</div>
                    <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.6); padding: 5px; border-radius: 3px; font-size: 8px; color: white; z-index: 10; display: flex; flex-direction: column; gap: 3px; border: 1px solid rgba(255,255,255,0.2);">
                        <div style="margin-bottom: 2px;">BEM E-Field (V/mm)</div>
                        <div style="display: flex; align-items: center; gap: 5px;"><div style="width:10px; height:10px; background: red;"></div> High (> 2.0)</div>
                        <div style="display: flex; align-items: center; gap: 5px;"><div style="width:10px; height:10px; background: yellow;"></div> Medium (1.0)</div>
                        <div style="display: flex; align-items: center; gap: 5px;"><div style="width:10px; height:10px; background: green;"></div> Low (0.5)</div>
                        <div style="display: flex; align-items: center; gap: 5px;"><div style="width:10px; height:10px; background: blue;"></div> Min (0.0)</div>
                    </div>
"""

new_text = text.replace(
    '<div style="font-size: 10px; color: var(--accent-pink); margin-bottom: 5px;">CORTICAL CURRENT DENSITY</div>',
    legend_html
)

with open(file_path, "w") as f:
    f.write(new_text)

print("Added BEM Legend to Cortical Canvas")

