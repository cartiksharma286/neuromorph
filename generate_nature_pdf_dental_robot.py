#!/usr/bin/env python3
"""
Entry point to generate the Nature Preprint PDF for the Dental Image-Guided Surgery Robot.
"""
import os
import sys

# Add dental_robot_surgery module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dental_robot_surgery.generate_nature_pdf import generate_dental_robot_nature_pdf

if __name__ == '__main__':
    pdf_path = generate_dental_robot_nature_pdf()
    print(f"Preprint generated: {pdf_path}")
