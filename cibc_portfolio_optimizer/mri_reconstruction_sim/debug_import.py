
import sys
import os

print("Starting extended debug import...")

def try_import(module_name):
    try:
        __import__(module_name)
        print(f"{module_name} imported successfully")
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        import traceback
        traceback.print_exc()

try_import('simulator_core')
try_import('circuit_schematic_generator')
try_import('generate_pdf')
try_import('app_enhanced')
