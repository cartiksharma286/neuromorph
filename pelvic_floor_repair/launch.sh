#!/bin/bash

# Gynecological Repair & Pelvic Floor Reconstruction System
# Launch Script

echo "🏥 Launching Pelvic Floor Reconstruction Application..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Navigate to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Clear any cached files
echo "🧹 Cleaning up..."
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

# Display system info
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🏥 PELVIC FLOOR RECONSTRUCTION SYSTEM"
echo "════════════════════════════════════════════════════════════════"
echo "🔧 Service:          Gynecological Repair Assistant"
echo "🧠 AI Engine:        LLM Design Assistant v1.0"
echo "📊 Analysis Mode:    Combinatorial Implant Design"
echo "🎨 Visualization:    3D Chamber & Implant Models"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Start the Flask application
echo "🚀 Starting Flask Server..."
echo "📡 Access the application at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
