#!/usr/bin/env python3
"""
Generate comprehensive PDF report for Knee Vascular Coil Project
"""
import sys
sys.path.append('/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim')

from generate_pdf import md_to_pdf
import os

# Define paths
base_dir = '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim'
md_file = os.path.join(base_dir, 'KNEE_COIL_PROJECT_SUMMARY.md')
pdf_file = os.path.join(base_dir, 'Knee_Vascular_Coil_Project_Report.pdf')

print("=" * 80)
print("GENERATING KNEE VASCULAR COIL PROJECT PDF REPORT")
print("=" * 80)
print(f"\nSource: {md_file}")
print(f"Output: {pdf_file}")
print("\nProcessing...")

try:
    # Generate PDF from markdown
    md_to_pdf(md_file, pdf_file)
    
    print("\n" + "=" * 80)
    print("✓ PDF REPORT GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nReport saved to:")
    print(f"  {pdf_file}")
    
    # Check file size
    if os.path.exists(pdf_file):
        size_kb = os.path.getsize(pdf_file) / 1024
        print(f"\nFile size: {size_kb:.1f} KB")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
