#!/bin/bash

# DICOM Viewer Launcher Script
# Launches the enhanced DICOM viewer with proper environment setup

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DICOM Neuroimage Viewer Launcher${NC}"
echo -e "${BLUE}========================================${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found: $(python3 --version)${NC}"

# Check required DICOM file
DICOM_FILE="$SCRIPT_DIR/neurovascular_brain_with_tumor.dcm"
if [ ! -f "$DICOM_FILE" ]; then
    echo -e "${RED}✗ Sample DICOM file not found:${NC}"
    echo -e "  $DICOM_FILE"
    echo ""
    echo "Creating sample DICOM file..."
    python3 "$SCRIPT_DIR/create_sample_dicom.py"
fi

echo -e "${GREEN}✓ Sample DICOM file ready${NC}"
echo -e "  File: $(basename $DICOM_FILE)"
echo -e "  Size: $(du -h "$DICOM_FILE" | cut -f1)"

# Check if enhanced viewer exists
VIEWER_FILE="$SCRIPT_DIR/dicom_viewer_enhanced.py"
if [ ! -f "$VIEWER_FILE" ]; then
    echo -e "${RED}✗ Viewer script not found:${NC}"
    echo -e "  $VIEWER_FILE"
    exit 1
fi

echo -e "${GREEN}✓ Viewer script ready${NC}"

# List dependencies
echo ""
echo -e "${BLUE}Checking dependencies...${NC}"

python3 << 'EOF'
import sys
import importlib

dependencies = {
    'PyQt5': 'PyQt5.QtWidgets',
    'ITK': 'itk',
    'VTK': 'vtk',
    'pydicom': 'pydicom',
    'NumPy': 'numpy',
    'PIL': 'PIL'
}

missing = []
for name, module in dependencies.items():
    try:
        importlib.import_module(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (missing)")
        missing.append(name)

if missing:
    print(f"\nError: Missing dependencies: {', '.join(missing)}")
    print("\nInstall with:")
    for dep in missing:
        if dep == 'PyQt5':
            print(f"  pip3 install PyQt5")
        elif dep == 'ITK':
            print(f"  pip3 install itk")
        elif dep == 'VTK':
            print(f"  pip3 install vtk")
        elif dep == 'pydicom':
            print(f"  pip3 install pydicom")
        elif dep == 'PIL':
            print(f"  pip3 install Pillow")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Dependency check failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Launching DICOM Viewer...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Controls:"
echo "  Left/Right Arrow: Navigate slices"
echo "  Up/Down Arrow: Adjust Gaussian smoothing"
echo "  Ctrl+O: Open DICOM file"
echo "  Ctrl+D: Run auto demo"
echo "  Ctrl+Q: Quit"
echo ""
echo "Features:"
echo "  • Real-time image processing (Gaussian, Median, Threshold)"
echo "  • 2D/3D visualization (Volume & Surface rendering)"
echo "  • Interactive slice navigation"
echo "  • High-intensity tumor detection"
echo "  • Neurovascular structure visualization"
echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Launch the viewer
python3 "$VIEWER_FILE"
