
import os

path = 'index.html'
with open(path, 'r') as f:
    content = f.read()

# Replace Branding
replacements = {
    'Powered by Google Quantum AI': 'Powered by NVIDIA NVQLink',
    'Sycamore-optimized': 'NVQLink-optimized',
    'Run Quantum Neural Repair': 'Run NVQLink Neural Repair',
    'using Google Quantum VQE': 'using NVQLink VQE',
    '#4285f4': '#76b900',  # Replace Google Blue with NVIDIA Green
    '#ea4335': '#000000',  # Replace Google Red with Black (Gradient end)
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(path, 'w') as f:
    f.write(content)
print("Updated index.html to NVQLink branding")
